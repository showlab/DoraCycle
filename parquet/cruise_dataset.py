import os
import gc
import math
import torch
import pickle
import random
import warnings
import copy
import time
import numpy as np
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import barrier, is_initialized
from torch.utils.data import _utils
from functools import partial
from cruise.utilities.distributed import DIST_ENV
from cruise.data_module.u13.u13_config_manager import U13ConfigManager

try:
    from torch.utils.data.dataloader import BytedDataLoader as DataLoader
except ImportError:
    from torch.utils.data import DataLoader
from typing import Callable, List, Optional, Union, Dict, Tuple
from .utils import LazyLoader
from .hybrid_dataset import HybridDataset, DatasetCallback, DatasetFinished, DatasetPolled
from .utils import get_row_group_infos, get_kv_keys
from .utils import get_dataset_size, get_free_disk_space, download_hdfs_data, shard_source, split_by_row_group
from .utils import get_rank, get_world, get_pickle_length, get_length_from_sources
from .utils.files_utils import aggregate_parent_urls
from ..utilities.report_usage import report_usage
from ..utilities.data_io_metrics import CruiseLoaderIOMetrics
from ..utilities.rank_zero import once_only, rank_zero_warn
from ..utilities.logger import get_cruise_logger


logger = get_cruise_logger()

pq = LazyLoader('pq', globals(), 'pyarrow.parquet')
byted_dataloader = LazyLoader('dataloader', globals(), 'dataloader')

kv_ds_length_info = dict()
STATE_SKIP_SAVE = [
    "torch_iter", "torch_loader", "kv_loader", "data_iters",
    "stop_queue", "transform_fn", "batch_transform_fn",
    "post_process", "_first_batch", "worker_dismiss_block_signals",
    "worker_state_queues", "resume_buffer", "change_sharing_strategy",
    "u13_config_manager"
]
STATE_PICKLE_SAVE = ["processor", "decode_fn"]

__all__ = ['DistributedCruiseDataLoader']

VALID_DATA_RETRY = 100


def simple_collate(x):
    return x


def get_fake_gpu_trans():
    return simple_collate


def set_seed(seed, worker_id):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_config(seed, change_sharing_strategy, worker_id):
    # users may customize their gc behavior, once they set gc.disabled()
    # in the main process, which will make dataloader subprocess memory
    # leak, so we must enable gc in loader.
    if not gc.isenabled():
        gc.enable()
    set_seed(seed, worker_id)
    if change_sharing_strategy:
        torch.multiprocessing.set_sharing_strategy("file_system")


class DistributedCruiseDataLoader:
    r"""
    Cruise Data Loader, an user-friendly data loader, support hybrid multiple tasks.

        Args:
            data_sources (Union[List[List[str]], List[List[Tuple[str, str]]]]):
                We support two types of data sources input:
                    (1) List[List[str]]: data sources paths like [[kv1, kv2, kv3], [parquet1, parquet2]]
                    (2) List[List[Tuple[str, str]]]: for each input data source, the data source (actual
                        video data) and metadata source (like video caption, resolution, fps) are packed
                        as a tuple. For this input type, we only support parquet file format and metadata
                        file path should be placed behind of the data file path. Example: [[(parquet_data1,
                        parquet_metadata1), (parquet_data2, parquet_metadata2)]]
            parquet_cache_on (bool, optional):
                Enable parquet dataset caching mode. Defaults to True.
            no_sharding (bool, optional):
                Do not perform dataset sharding. Defaults to False.
            fast_resume (bool, optional):
                resume the loader state in a faster way, which might sacrifice the resume precision
            synthetic_sample (bool, optional):
                if True, the prefetch processes will use fist sample as synthetic data and skip
                the following data reading process. That data will be transformed/processed later
                within whole pipeline.
            shuffle_buffer_size (int, optional, default=100):
                if `shuffle` is True and loader loading on iterable Datasets,
                i.e Parquet Dataset or TFRecord Dataset, these Datasets will open a sample buffer to shuffle data,
                the size of that sample buffer is equal to `shuffle_buffer_size`.
                The unit is per data, not per batch. Every dataset worker has a shuffle buffer.
            rank (int, optional, default=None):
                if params `rank` and `world_size` are not None and `no_sharding` is `False`,
                `DistributedCruiseDataLoader.rank` and `DistributedCruiseDataLoader.world` will be overwriten.
            world_size (int, optional, default=None):
                used with `rank`.
            row_group_shuffle (bool, optional, default=False):
                if is True, enable shuffle by row groups in parquet. It only works when `shuffle` is `True`.
            data_padding (bool):
                if True, cruise will pad additional batch to ranks so that each rank provides the same batches of data.
            bitwise_resume (bool, optional, default=False):
                if is True, loader will resume in a precise way and generate loader ckpt, which also includes random
                state for shuffle. If enable this feature, `fast_resume` will be overwritten. Currently only supports
                for parquet dataset. Be advised the saving interval should be larger than (prefetch_ratio + 1) * num_workers
            auto_source_len (bool, optional, default=False):
                if is True, we will not pre calculate the length of the dataset, but will directly shuffle
                according to the dataset's custom sampler policy. At this time, the length of the dataset is unknown,
                and when predefined_steps is -1, the dataloader will only complete the entire dataset once,
                and the steps of the dataloader on different ranks may be inconsistent
            torch_num_threads (int, optional, default=0):
                Configure torch num threads. If it is less than or equal to 0, it will not be explicitly configured,
                but will use the default value (1) that comes with torch
            diff_seed (bool, optional, default=False):
                make random seeds different across rank, to make multiplx data more average.
            stream (bool, optional, default=False):
                Let the dataset read samples one by one as the input for the callback.
            cache_processed (bool, optional, default=False):
                Let the dataset cache its output. Usually work with `persistent_workers` to make sure validation loader
                result is the same across different epochs.
            persistent_workers (bool, optional, default=False):
                keep torch loader workers alive when we re-iter loader. This may save some time for loader initialization.
    """

    def __init__(self,
                 data_sources: Union[List[List[str]], List[List[Tuple[str, str]]]],
                 keys_or_columns: List[List[str]],
                 batch_sizes: List,
                 num_workers: int,
                 num_readers: Optional[Union[int, List[int]]],
                 decode_fn_list: List[Callable],
                 processor,
                 predefined_steps: Union[int, str] = None,
                 source_types: Optional[List[str]] = None,
                 seed=0,
                 last_step: int = 0,
                 kv_loader_state: Dict = {},
                 shuffle: bool = False,
                 task_id_list: List = [],
                 use_gpu: bool = False,
                 enable_borrower: bool = False,
                 multiplex_weights: list = [],
                 use_all_gather: bool = False,
                 remain_sample_idx: bool = False,
                 transform_output_many: bool = False,
                 drop_last: bool = True,
                 key_mapping: List[Dict] = None,
                 local_fetch_path: str = None,
                 pin_memory: bool = True,
                 parquet_cache_on: bool = True,
                 use_arnold: bool = True,
                 no_sharding: bool = False,
                 shuffle_buffer_size: int = 100,
                 synthetic_sample: bool = False,
                 synthetic_batch: bool = False,
                 fast_resume: bool = False,
                 rank: int = None,
                 world_size: int = None,
                 row_group_shuffle: bool = False,
                 data_padding: bool = False,
                 remain_filename: bool = False,
                 bitwise_resume: bool = False,
                 auto_source_len: bool = False,
                 torch_num_threads: int = 0,
                 callbacks: List[DatasetCallback] = [],
                 diff_seed: bool = False,
                 source_meta: Optional[dict] = None,
                 same_seed_across_epoch: bool = False,
                 stream: bool = False,
                 cache_processed: bool = False,
                 persistent_workers: bool = False,
                 raise_when_subprocess_error: bool = True,
                 **kwargs):

        assert len(data_sources) != 0, 'data_sources is empty, can not fetch data'
        self.orig_data_sources = copy.deepcopy(data_sources)
        for i in range(len(data_sources)):
            if len(data_sources[i]) == 0:
                logger.warning(f"data index {i} in data_sources is empty, be aware")
        u13_id_fields = kwargs.get('u13_id_fields', None)
        u13_id_labels = kwargs.get('u13_id_labels', None)
        self.u13_config_manager = U13ConfigManager(data_sources, source_types, u13_id_fields, u13_id_labels)
        self.data_sources, self.source_types, self.u13_filters = self.u13_config_manager.get_sources_with_u13_info()
        self.magnus_datasource_count = self.u13_config_manager.magnus_datasource_count
        self.u13_datasource_count = self.u13_config_manager.u13_datasource_count
        self.data_sources_count = len(self.source_types)
        self.data_sources, self.source_types, self.magnus_table_infos = self.plan_magnus(self.data_sources,
                                                                                         self.source_types)
        self.diff_seed = diff_seed
        if not task_id_list:
            self.task_id_list = [None for _ in self.data_sources]
        else:
            assert len(task_id_list) == len(self.data_sources), 'task_id_list should have equal length as data_sources'
            self.task_id_list = task_id_list
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        if self.num_workers <= 0:
            self.num_workers = 0
            warnings.warn("Detected Dataloader num_worker = 0, which might affect the data loading performance.")
        self.num_readers = num_readers
        self.return_keys = keys_or_columns
        self.enable_borrower = enable_borrower
        self.multiplex_weights = multiplex_weights
        self.source_meta = source_meta if source_meta else {}
        for i, data_source in enumerate(self.data_sources):
            if i not in self.source_meta or not self.source_meta[i]:
                self.source_meta[i] = None
            else:
                assert len(self.source_meta[i]) == len(data_source), \
                    f"source_meta should have the same length as {i}th data source"

        self.remain_sample_idx = remain_sample_idx
        self.remain_filename = remain_filename
        self.transform_output_many = transform_output_many
        self.drop_last = drop_last
        self.no_sharding = no_sharding
        self.cache_processed = cache_processed
        self.fast_resume = fast_resume
        self.bitwise_resume = bitwise_resume
        assert not (self.bitwise_resume and self.enable_borrower), "bitwise_resume and enable_borrower \
            shall not both be True"
        # only train loader cfg can update DIST_ENV config
        # tell trainer to save loader ckpt for each rank
        if shuffle or bitwise_resume:
            DIST_ENV.bitwise_ckpt = bitwise_resume
        # tell trainer to update loader chkpt prefix for each rank
        if shuffle or auto_source_len:
            DIST_ENV.dataset_use_auto_len = auto_source_len
        self.row_group_shuffle = row_group_shuffle
        self.data_padding = data_padding
        self.callbacks = callbacks
        self.stream = stream
        self.auto_source_len = auto_source_len
        if self.no_sharding:
            self.rank = 0
            self.world = 1
            if self.num_workers > 1:
                msg = f"Dataloader num_workers adjusted from {self.num_workers} to 1 with `no_sharding=True`"
                warnings.warn(msg)
                self.num_workers = 1
        else:
            self.rank = get_rank()
            self.world = get_world()
            if rank is not None and world_size is not None:
                self.rank = rank
                self.world = world_size
        self.use_all_gather = use_all_gather and self.world > 1
        if self.use_all_gather:
            assert is_initialized(), "Users should make sure dist group is initialized"
        self.kwargs = kwargs

        self.repeat = True
        # use length == -1 to indicate loading all the data as one epoch
        if predefined_steps == -1:
            self.repeat = False
        self.repeat = self.repeat or self.data_padding

        self.kv_downsample_ratio = kwargs.get("downsample_ratio", -1.0)

        self.source_lens = [[]] * len(self.data_sources)
        triplet_sampling = kwargs.get("triplet_sampling", False)
        if triplet_sampling:
            self.length = self._init_triplet_length(predefined_steps)
        else:
            self.length = self._init_length(predefined_steps, drop_last)
        self.predefined_steps = predefined_steps if isinstance(predefined_steps, int) else self.length
        self.resume = True if last_step > 0 else False
        self.reset_resume_step = False
        self.seed = seed
        self.same_seed_across_epoch = same_seed_across_epoch
        self.epoch = 0
        self.step = last_step

        # Zhi: bsz is per rank now by removing this
        # for i, batch in enumerate(self.batch_sizes):
        #     self.batch_sizes[i] = batch // self.world
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.use_gpu = use_gpu
        self.transform_fn = processor.transform
        self.batch_transform_fn = processor.batch_transform
        self.post_process = getattr(processor, 'post_transform', None)
        self.processor = processor
        self.decode_fn = decode_fn_list
        self.torch_num_threads = torch_num_threads
        self.persistent_workers = persistent_workers
        self.raise_when_subprocess_error = raise_when_subprocess_error
        # only used for arnold dataset
        self.transform_replace_all = kwargs.get("transform_replace_all", False)

        if self.data_padding:
            self.remain_sample_idx = True
            if hasattr(self.processor, "modal_keys") and isinstance(self.processor.modal_keys, dict):
                # pass the crs_sample_idx to trainer so that trainer could check if there is repeated data
                self.processor.modal_keys["crs_sample_idx"] = ["crs_sample_idx"]
        self.stop_queue = mp.Queue()
        if key_mapping:
            self.key_mapping = []
            for each in key_mapping:
                if not each:
                    self.key_mapping.append({})
                    continue
                num_keys = len(each['previous'])
                mappings = {each['previous'][i]: each['current'][i] for i in range(num_keys)}
                self.key_mapping.append(mappings)
        else:
            self.key_mapping = [None] * len(self.data_sources)
        self.pin_memory = pin_memory
        self.parquet_cache_on = parquet_cache_on

        local_fetch_success = False
        if local_fetch_path:
            if not os.path.exists(local_fetch_path):
                os.makedirs(local_fetch_path, exist_ok=True)
            ds_total_size = 0
            for i, data_source in enumerate(self.data_sources):
                for dataset in data_source:
                    ds_total_size += get_dataset_size(dataset, self.source_types[i])
            available = get_free_disk_space(local_fetch_path)
            if available * 0.5 > ds_total_size / 1024 ** 3:
                dataset_mappings = download_hdfs_data(self.data_sources, source_types, local_fetch_path)
                if torch.distributed.is_initialized():
                    barrier()
                local_fetch_success = True
                for i, data_source in enumerate(self.data_sources):
                    for j, dataset in enumerate(data_source):
                        self.data_sources[i][j] = dataset_mappings[dataset]

        # We found that torch iter_loader is better after fetching data to the local disk
        if int(os.getenv("CRUISE_LOADER_USE_ARNOLD_DATASET", "1")) and not local_fetch_success and use_arnold:
            self.kv_source_idx = [i for i in range(len(self.data_sources)) if self.source_types[i] == "kv"]
            self.iter_source_idx = [i for i in range(len(self.data_sources)) if self.source_types[i] != "kv"]
            self.key_mapping = [self.key_mapping[i]
                                for i in range(len(self.data_sources)) if self.source_types[i] != "kv"]
        else:
            self.kv_source_idx = []
            self.iter_source_idx = list(range(len(self.data_sources)))
        self.synthetic_sample = synthetic_sample
        if synthetic_sample:
            warnings.warn(f"Dataloader is runing on synthetic samples, data loading processes is skipped.")
        self.synthetic_batch = synthetic_batch
        if synthetic_batch:
            warnings.warn(f"Dataloader is runing on synthetic batch, transform/batch_transform processes is skipped.")
        self.kv_loader = None
        self.torch_iter = None
        self.kv_loader_state = kv_loader_state
        self.torch_loader = None

        self.all_worker_states = {}
        self.resume_buffer = [[] for _ in range(max(self.num_workers, 1))]
        self.worker_on_duty = 0
        self._finished_worker = [False for _ in range(max(self.num_workers, 1))]
        self.change_sharing_strategy = False

        if not self.shuffle and self.iter_source_idx:
            # create a loader instance when shuffle is False to save time
            self.torch_loader = self._create_iter_loader()

        if self.kv_source_idx:
            self._create_kv_loader()

        self.loader_prob = []
        if self.multiplex_weights:
            kv_weights = sum([self.multiplex_weights[i] for i in self.kv_source_idx])
            iter_weights = 1 - kv_weights
            if kv_weights:
                self.loader_prob.append(kv_weights)
            if iter_weights:
                self.loader_prob.append(iter_weights)
        self.ask_subprocess_stop = False

        self.report_usage_point()

    def __del__(self):
        self.terminate()

    @once_only
    def report_usage_point(self):
        additional_info = {
            'data_sources': str(list(aggregate_parent_urls(self.data_sources))),
            'orig_data_sources': str(list(self.orig_data_sources)),
            'batch_sizes': str(self.batch_sizes),
            'num_workers': str(self.num_workers),
            'num_readers': str(self.num_readers),
            'source_types': str(self.source_types),
            'return_keys': str(self.return_keys),
            'enable_borrower': str(self.enable_borrower),
            'multiplex_weights': str(self.multiplex_weights),
            'source_meta': str(self.source_meta),
            'remain_sample_idx': str(self.remain_sample_idx),
            'transform_output_many': str(self.transform_output_many),
            'drop_last': str(self.drop_last),
            'no_sharding': str(self.no_sharding),
            'fast_resume': str(self.fast_resume),
            'bitwise_resume': str(self.bitwise_resume),
            'rank': str(get_rank()),
            'magnus_table_infos': str(self.magnus_table_infos)
        }
        extra_info = {
            'dataset_count': self.data_sources_count,
            'magnus_dataset_count': self.magnus_datasource_count,
            'u13_filter_applied_dataset_count': self.u13_datasource_count
        }
        report_usage(self.__class__.__name__, extra_info=extra_info, additional_info=additional_info)

    def _create_kv_loader(self):
        kv_sources = [self.data_sources[i] for i in self.kv_source_idx]
        kv_u13_filters = [self.u13_filters[i] for i in self.kv_source_idx]
        kv_batch_sizes = [self.batch_sizes[i] for i in self.kv_source_idx]
        # for arnold dataset, multiple kv dataset use the same kv num_readers, here we just get the maximum
        kv_num_readers = max([self.num_readers[i] for i in self.kv_source_idx])
        kv_task_id_list = [self.task_id_list[i] for i in self.kv_source_idx]
        kv_ds_split_num = self.kwargs.get("dataset_split_num", 4)
        kv_epochs_for_reader = self.kwargs.get("epochs_for_reader", 5)
        kv_decode_fn = None if not self.decode_fn else [self.decode_fn[i] for i in self.kv_source_idx]
        kv_return_keys = None if not self.return_keys else [self.return_keys[i] for i in self.kv_source_idx]
        triplet_sampling = self.kwargs.get("triplet_sampling", False)
        triplet_meta_dict_path = self.kwargs.get("triplet_meta_dict_path", "")
        triplet_meta_dict_format = self.kwargs.get("triplet_meta_dict_format", "pickle")
        triplet_p = self.kwargs.get("triplet_p", 1)
        triplet_k = self.kwargs.get("triplet_k", 1)
        from .arnold_dataset import ArnoldDataset
        size_guarantee = not self.shuffle
        self.kv_loader = ArnoldDataset(
            kv_sources, kv_batch_sizes, kv_task_id_list, self.num_workers, kv_num_readers, world_size=self.world,
            rank=self.rank, shuffle=self.shuffle, return_keys=kv_return_keys, decode_fn=kv_decode_fn,
            trans_fn=(self.transform_fn, self.batch_transform_fn, self.post_process),
            dataset_split_num=kv_ds_split_num, pin_memory=self.pin_memory, epochs_for_reader=kv_epochs_for_reader,
            remain_sample_idx=self.remain_sample_idx, resume_state=self.kv_loader_state,
            transform_replace_all=self.transform_replace_all,
            triplet_info=(triplet_sampling, triplet_meta_dict_path, triplet_meta_dict_format, triplet_p, triplet_k),
            size_guarantee=size_guarantee, synthetic_sample=self.synthetic_sample,
            transform_many=self.transform_output_many, downsample_ratio=self.kv_downsample_ratio,
            remain_filename=self.remain_filename, u13_filters=kv_u13_filters)

    def _create_iter_loader(self):
        shard_data, shard_rank_info = self.shard_data_sources()
        iter_sources = [shard_data[i] for i in self.iter_source_idx]
        iter_shard_rank_info = [shard_rank_info[i] for i in self.iter_source_idx]
        iter_source_types = [self.source_types[i] for i in self.iter_source_idx]
        iter_u13_filters = [self.u13_filters[i] for i in self.iter_source_idx]
        iter_batch_sizes = [self.batch_sizes[i] for i in self.iter_source_idx]
        iter_num_readers = [self.num_readers[i] for i in self.iter_source_idx]
        iter_task_id_list = [self.task_id_list[i] for i in self.iter_source_idx]
        iter_return_keys = None if not self.return_keys else [self.return_keys[i] for i in self.iter_source_idx]
        iter_decode_fn = None if not self.decode_fn else [self.decode_fn[i] for i in self.iter_source_idx]
        iter_multiplex_weights = [] if not self.multiplex_weights else [self.multiplex_weights[i] for i in self.iter_source_idx]
        if iter_multiplex_weights:  # normalize the probability to make them sum to 1
            weight_sum = sum(iter_multiplex_weights)
            iter_multiplex_weights = [i / weight_sum for i in iter_multiplex_weights]
        batch_shuffle = self.kwargs.get("batch_shuffle", False)
        loader_kwargs = {}
        if "prefetch_factor" in self.kwargs and self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.kwargs.get("prefetch_factor", 2)
        self.worker_state_queues = [mp.Queue(1) for _ in range(max(self.num_workers, 1))]
        self.worker_to_block_signals = [mp.Event() for _ in range(max(self.num_workers, 1))]
        self.worker_dismiss_block_signals = [mp.Event() for _ in range(max(self.num_workers, 1))]
        falcon_config = self.kwargs.get("falcon_config", {}).copy()
        if self.source_meta:
            falcon_config["dataset_length"] = [self.source_meta.get(i) for i in self.iter_source_idx]

        if self.same_seed_across_epoch:
            seed = self.seed
        else:
            seed = self.seed + self.epoch
        if self.diff_seed:
            seed = seed + self.rank
        iter_dataset = HybridDataset(iter_sources, iter_source_types, iter_batch_sizes, iter_num_readers,
                                     iter_return_keys, iter_decode_fn,
                                     (self.transform_fn, self.batch_transform_fn, self.post_process),
                                     self.shuffle,
                                     seed,
                                     self.step, shard_rank_info=iter_shard_rank_info,
                                     task_id_list=iter_task_id_list, multiplex_weights=iter_multiplex_weights,
                                     remain_sample_idx=self.remain_sample_idx, repeat=self.repeat,
                                     stop_queue=self.stop_queue, key_mapping_list=self.key_mapping,
                                     drop_last=self.drop_last, batch_shuffle=batch_shuffle,
                                     parquet_cache_on=self.parquet_cache_on,
                                     shuffle_buffer_size=self.shuffle_buffer_size,
                                     fast_resume=self.fast_resume, synthetic_sample=self.synthetic_sample,
                                     transform_many=self.transform_output_many, batch_padding=self.data_padding,
                                     remain_filename=self.remain_filename, state_queues=self.worker_state_queues,
                                     to_blocks=self.worker_to_block_signals,
                                     dismiss_blocks=self.worker_dismiss_block_signals,
                                     worker_states=self.all_worker_states, bitwise_resume=self.bitwise_resume,
                                     falcon_config=falcon_config, callbacks=self.callbacks,
                                     u13_filters=iter_u13_filters,
                                     torch_num_threads=self.torch_num_threads, diff_seed=self.diff_seed, stream=self.stream,
                                     cache_processed=self.cache_processed)

        torch_loader = DataLoader(iter_dataset, num_workers=self.num_workers, collate_fn=simple_collate,
                                  batch_size=None, pin_memory=self.pin_memory,
                                  persistent_workers=self.persistent_workers,
                                  worker_init_fn=partial(set_config, self.seed, self.change_sharing_strategy), **loader_kwargs)
        if self.enable_borrower and hasattr(torch_loader, "_enable_borrower"):
            torch_loader.enable_borrower()
        return torch_loader

    def __iter__(self):
        """Get the dataset iterator."""
        if self.stop_queue is not None and not self.stop_queue.empty():
            self.stop_queue.get()
        self.ask_subprocess_stop = False
        if self.resume:
            self.resume = False

            # Reconstruct self.torch_loader at the beginning of the next epoch to reset resume_step.
            self.reset_resume_step = True
        else:
            self.all_worker_states = {}
            self.resume_buffer = [[] for _ in range(max(self.num_workers, 1))]
            if self.shuffle:
                self._shuffle()
            self.step = 0
            self.worker_on_duty = 0

            if self.reset_resume_step:
                if not self.shuffle and self.iter_source_idx:
                    self.torch_loader = self._create_iter_loader()
                self.reset_resume_step = False

        if self.iter_source_idx:
            if self.shuffle or self.torch_loader is None:
                torch_loader = self._create_iter_loader()
            else:
                torch_loader = self.torch_loader
            self.torch_iter = iter(torch_loader)
        self.data_iters = []
        if self.kv_loader is not None:
            self.data_iters.append(self.kv_loader)
        if self.torch_iter is not None:
            self.data_iters.append(self.torch_iter)

        # TODO add gpu loader
        # if not self.use_gpu:
        #     self.loader = iter(torch_loader)
        # else:
        #     # right now just use fake_gpu_trans as a place holder
        #     gpu_loader = GPULoader(torch_loader, get_fake_gpu_trans, step=self.length)
        #     self.loader = iter(gpu_loader)
        self._loader_alive = True

        self.io_metrics = CruiseLoaderIOMetrics()
        return self

    def combine_two_loader(self, data):
        item1 = data[0]
        item2 = data[1]
        if isinstance(item1, list):
            return item1 + item2
        if isinstance(item1, dict):
            res = {}
            for k in item1.keys():
                res[k] = self.combine_two_loader([item1[k], item2[k]])
            return res

    def _get_data_from_loaders(self):
        if self.multiplex_weights:
            data_iter = random.choices(self.data_iters, weights=self.loader_prob, k=1)[0]
            return next(data_iter)
        else:
            if self.kv_loader is not None:
                kv_data = next(self.kv_loader)
                if not self.torch_iter:
                    return kv_data
            if self.torch_iter is not None:
                iter_data = next(self.torch_iter)
                if not self.kv_loader:
                    return iter_data
            data = self.combine_two_loader([kv_data, iter_data])
            return data

    def __next__(self):
        # encase self.length = -1
        if self.length > 0 and self.step >= self.length:
            # if self.use_gpu:
            #     # call __next__ one more time to actually stop gpu loader
            #     next(self.loader)
            self.stop_queue.put(1)
            self.ask_subprocess_stop = True
            if self.torch_iter is not None and self.num_workers > 0:
                self.shutdown_torch_iter()
            self.step = 0
            self.worker_on_duty = 0
            logger.info(f"rank {DIST_ENV.rank} loader.__next__ raise Stopiteration")
            raise StopIteration
        # update step only after successfully getting the data
        # otherwise, the recorded step count might be larger than the actual iterated step
        try:
            start_time = time.time()
            if self.synthetic_batch and hasattr(self, "_first_batch"):
                # get the first batch of data only, data will not be processed/transformed
                data = self._first_batch
            elif self.bitwise_resume:
                data = None
                _got_data = False
                assert len(self.resume_buffer) > 0, "self.resume_buffer should not be empty"
                # if `resume_buffer` has data, use it immediately, otherwise, gather data
                # from loader
                # make sure each unfinished worker loads at least one sample
                _retry_0 = 0
                while any([len(x) == 0 for wid, x in enumerate(self.resume_buffer) if not self._finished_worker[wid]]):
                    try:
                        if _retry_0 >= VALID_DATA_RETRY and _retry_0 % 100 == 0:
                            for wid, buffer in enumerate(self.resume_buffer):
                                if len(buffer) == 0:
                                    self._finished_worker[wid] = True
                            alive_workers = getattr(self.torch_iter, "_workers_status", [False])
                            warnings.warn(f"[bitwise] rank {DIST_ENV.rank} loader detected too many ({_retry_0}) \
                                retry on collecting data, alive_workers: {alive_workers}, \
                                self._finished_worker: {self._finished_worker}, possible hang detected.")
                        _retry_0 += 1
                        _data, _wid = None, None
                        _retry_1 = 0
                        while not self._is_valid_data(_data):
                            if _retry_1 >= VALID_DATA_RETRY and _retry_1 % 100 == 0:
                                warnings.warn(f"[bitwise] rank {DIST_ENV.rank} loader detected too many ({_retry_1}) \
                                    retry on getting valid data, possible hang detected, \
                                    data sample: {self._inspect_invalid_data(data)}")
                            _data, _wid = self._get_data_from_loaders()
                            _retry_1 += 1
                        if isinstance(_data, DatasetFinished):
                            self._finished_worker[_wid] = True
                        else:
                            self.resume_buffer[_wid].append((_data, _wid))
                    except StopIteration:
                        logger.info(f"[bitwise] rank {DIST_ENV.rank} loader.__next__ got Stopiteration")
                        self._loader_alive = False
                        break
                    except Exception as e:
                        warnings.warn(f"[bitwise] rank {DIST_ENV.rank} loader.__next__ exception: {str(e)}")
                        if self.raise_when_subprocess_error:
                            raise e
                if all([len(x) == 0 for x in self.resume_buffer]) and all(self._finished_worker):
                    logger.info(f"[bitwise] rank {DIST_ENV.rank} loader raise Stopiteration")
                    raise StopIteration
                for i in range(len(self.resume_buffer)):
                    cur_worker_on_duty = (self.worker_on_duty + i) % len(self.resume_buffer)
                    if self.resume_buffer[cur_worker_on_duty]:
                        data, _ = self.resume_buffer[cur_worker_on_duty].pop(0)
                        if not hasattr(self, "_first_batch"):
                            self._first_batch = data
                        self.worker_on_duty = (cur_worker_on_duty + 1) % len(self.resume_buffer)
                        _got_data = True
                        break
                if (not _got_data) and (not self._loader_alive):
                    logger.info(f"[bitwise] rank {DIST_ENV.rank} loader raise Stopiteration")
                    raise StopIteration
                assert _got_data, f"[bitwise] not valid data, \
                    self._finished_worker: {self._finished_worker}, \
                    self.resume_buffer: {[len(x) == 0 for x in self.resume_buffer]}, \
                    self._loader_alive: {self._loader_alive}"
            else:
                data = self._get_data_from_loaders()
                if not hasattr(self, "_first_batch"):
                    self._first_batch = data
            self.step += 1
            exe_time = time.time() - start_time
            self.io_metrics.add_data_point('exe_time', exe_time)
            self.io_metrics.add_one_step()
            return data
        except StopIteration:
            logger.info(f"rank {DIST_ENV.rank} loader.__next__ got Stopiteration")
            if self.torch_iter is not None and self.num_workers > 0 and not self.persistent_workers:
                # when we reach here, dataset processes are already joined/exited
                self.torch_iter = None
            self.step = 0
            self.worker_on_duty = 0
            self._finished_worker = [False for _ in range(max(self.num_workers, 1))]
            raise StopIteration

    def __len__(self):
        """Get the dataset length."""
        return self.length

    def shard_data_sources(self):
        shard_data = []
        shard_rank_info = []
        row_group_infos = {}
        parquet_sources = [x for i, x in enumerate(self.data_sources)
                           if self.source_types[i] == "parquet"]
        if self.row_group_shuffle:
            row_group_infos = get_row_group_infos(parquet_sources, int(self.rank), int(self.world))
        for i, data_source in enumerate(self.data_sources):
            if self.source_types[i] == "kv" and self.kv_source_idx:  # kv type data would use arnold dataset to loader data
                shard_data.append(None)
                shard_rank_info.append((0, 0))
            elif self.source_types[i] == 'webdataset':  # webdataset will shard in every dataset and won't need in-file reshard
                shard_data.append(data_source)
                shard_rank_info.append((int(self.rank), int(self.world)))
            elif self.source_types[i] == 'falcon' and self.auto_source_len:
                shard_data.append(data_source)
                shard_rank_info.append((int(self.rank), int(self.world)))
            else:
                source_meta = self.source_meta[i] if self.source_meta else None
                source_data, shard_rank, shard_world, returned_source_meta = shard_source(
                    data_source, self.rank, self.world, self.num_workers, self.source_types[i],
                    source_meta, self.drop_last, self.length, self.batch_sizes[i], self.use_all_gather)
                if row_group_infos:
                    row_group_shuffle_seed = self.seed if self.shuffle else None
                    source_data = split_by_row_group(source_data, row_group_infos, row_group_shuffle_seed)
                shard_data.append(source_data)
                shard_rank_info.append((shard_rank, shard_world))
                if not self.source_meta or i not in self.source_meta:
                    self.source_meta[i] = returned_source_meta
        return shard_data, shard_rank_info

    def _init_length(self, predefined_steps, drop_last=True):
        if predefined_steps and isinstance(predefined_steps, int) and predefined_steps > 0:
            # Users have predefined lengths for all data sources
            return predefined_steps
        if self.auto_source_len:
            length = predefined_steps if predefined_steps and isinstance(predefined_steps, int) else -1
            return length
        else:
            num_steps = []
            for k, data_source in enumerate(self.data_sources):
                if self.source_meta[k]:
                    source_len = sum(self.source_meta[k])
                else:
                    dist_flag = self.world > 1 and self.use_all_gather
                    source_len, source_meta = get_length_from_sources(
                        data_source, self.source_types[k], dist_flag, rank=self.rank, world=self.world)
                    # should prune files that have zero length
                    data_source = [source for source in data_source
                                   if source_meta[source[1] if isinstance(source, tuple)
                                                  and self.source_types[k] == "parquet" else source] > 0]
                    if len(data_source) < len(self.data_sources[k]):
                        zero_files = set(self.data_sources[k]) - set(data_source)
                        msg = f'cruise loader auto filters files with zero length: {zero_files}'
                        warnings.warn(msg)
                        self.data_sources[k] = data_source
                    source_meta = [source_meta[url[1] if isinstance(url, tuple)
                                               and self.source_types[k] == "parquet" else url] for url in data_source]
                    self.source_meta[k] = source_meta
                global_bsz = self.batch_sizes[k] * self.world
                # To align the steps calculated by arnold dataset
                # ArnoldDataset will pad each sub dataset to match the batch size
                # TODO discuss if the padding is appropriate when reading multiple dataset
                # since if the number of dataset is large, a lot of data would be padded.
                if self.source_types[k] == "kv" and int(os.getenv("CRUISE_LOADER_USE_ARNOLD_DATASET", "1")):
                    downsample_ratio = 1.0 if self.kv_downsample_ratio <= 0 else self.kv_downsample_ratio
                    tmp_num_steps = [math.ceil(int(i * downsample_ratio) / global_bsz) for i in source_meta]
                    all_steps = sum(tmp_num_steps)
                    # calculate the average steps for each dataloader process
                    kv_ds_split_num = self.kwargs.get("dataset_split_num", 4)
                    avg_steps_per_proc = all_steps / len(tmp_num_steps) / kv_ds_split_num
                    if avg_steps_per_proc < 10:  # hardcode the threshold for now
                        rank_zero_warn(
                            "KV subdatasets are too small and might cause frequent process switch, which would do harm to performance. \
                            Suggest to use parquet dataset instead.")

                    num_steps.append(all_steps)
                elif self.source_types[k] == "kv":
                    if drop_last:
                        num_steps.append(source_len // global_bsz)
                    else:
                        num_steps.append(math.ceil(source_len / global_bsz))
                else:
                    num_worker = self.num_workers if self.num_workers > 0 else 1
                    data_per_procs = source_len // (self.world * num_worker)
                    remain = source_len - self.world * num_worker * data_per_procs if not drop_last else 0
                    remain_per_rank = (remain + self.world - 1) // self.world

                    # get actual remain data for each rank
                    actual_remain_per_rank = []
                    for r in range(self.world):
                        if remain >= remain_per_rank:
                            actual_remain_per_rank.append(remain_per_rank)
                            remain -= remain_per_rank
                        elif remain > 0:
                            actual_remain_per_rank.append(remain)
                            remain = 0
                        else:
                            actual_remain_per_rank.append(0)

                    batch_per_rank = []
                    for r in range(self.world):
                        cur_rank_data_per_procs = [data_per_procs] * num_worker
                        for i in range(actual_remain_per_rank[r]):
                            cur_rank_data_per_procs[i] += 1
                        cur_rank_batch_per_procs = [
                            (i + self.batch_sizes[k] - 1) // self.batch_sizes[k] for i in cur_rank_data_per_procs
                        ]
                        batch_per_rank.append(sum(cur_rank_batch_per_procs))
                    # use rank 0's result as the step
                    num_steps.append(batch_per_rank[0])
                    # check if there exists potential hang issue
                    batch_per_rank_mismatch = any(i != batch_per_rank[0] for i in batch_per_rank)
                    if batch_per_rank_mismatch and not self.repeat and not self.data_padding:
                        rank_zero_warn(
                            f"Different ranks will have different steps to run for data source {k}! \
                            The step info for each rank would be {batch_per_rank}. \
                            \nPlease set data_padding to avoid hang issue.")

            if predefined_steps == 'min':
                return min(num_steps)
            else:
                return max(num_steps)

    def _init_triplet_length(self, predefined_steps):
        meta_dict_path = self.kwargs.get("triplet_meta_dict_path", "")
        meta_dict_format = self.kwargs.get("triplet_meta_dict_format", "pickle")
        batch_category = self.kwargs.get("triplet_p", 1)
        if predefined_steps and isinstance(predefined_steps, int) and predefined_steps > 0:
            # Users have predefined lengths for all data sources
            return predefined_steps
        if isinstance(meta_dict_path, str):
            meta_dict_path = [meta_dict_path]
        category_num = 0
        if meta_dict_format == 'kv':
            for path in meta_dict_path:
                keys = get_kv_keys(path)
                category_num += len(keys)
        elif meta_dict_format == 'pickle':
            ctx = mp.get_context("spawn")
            with ctx.Pool(1) as p:
                category_num = p.map(get_pickle_length, [meta_dict_path, ])[0]
        else:
            raise RuntimeError('triplet dataset format must be kv or pickle, please check the input')

        if self.drop_last:
            return category_num // (batch_category * self.world)
        else:
            return math.ceil(category_num / (batch_category * self.world))

    def _shuffle(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1  # incre seed so each time shuffle will have different order
        for i, data_source in enumerate(self.data_sources):
            if self.source_meta and self.source_meta[i]:
                tmp = list(zip(self.data_sources[i], self.source_meta[i]))
                rng.shuffle(tmp)
                self.data_sources[i], self.source_meta[i] = zip(*tmp)
                if isinstance(self.data_sources[i], tuple):
                    self.data_sources[i] = list(self.data_sources[i])
            else:
                rng.shuffle(data_source)

    @staticmethod
    def plan_magnus(data_sources, source_types):
        data_sources_new = []
        source_types_new = []
        magnus_table_infos = []

        for data_source, source_type in zip(data_sources, source_types):
            is_magnus_source_type = source_type in ['magnus', 'magnus_plan_only']
            if is_magnus_source_type:
                from ..utilities.magnus import build_magnus_read_confs, plan_tasks, \
                    BaseCruiseMagnusTask, \
                    underlying_file_format, plan_file_urls
                from ..utilities.magnus_properties import DR_METRIC_FIELD, DR_METRIC_LABEL
                from ..utilities.magnus.config import MagnusReadConf
                read_confs: List[MagnusReadConf] = build_magnus_read_confs(data_source)

                if source_type == 'magnus':
                    aggregated_tasks: List[BaseCruiseMagnusTask] = []
                    for read_conf in read_confs:
                        tasks, table_info = plan_tasks(read_conf)
                        aggregated_tasks.extend(tasks)
                        magnus_table_infos.append(table_info)
                    data_sources_new.append(aggregated_tasks)
                    source_types_new.append('magnus_task')
                elif source_type == 'magnus_plan_only':
                    aggregated_urls: List[str] = []
                    for read_conf in read_confs:
                        urls, table_info = plan_file_urls(read_conf)
                        aggregated_urls.extend(urls)
                        magnus_table_infos.append(table_info)
                    data_sources_new.append(aggregated_urls)

                    file_formats = [underlying_file_format(read_conf.table) for read_conf in read_confs]
                    file_format = file_formats[0]
                    assert all(format_ == file_format for format_ in file_formats), "All formats should be the same"
                    source_types_new.append(file_format)
            else:
                data_sources_new.append(data_source)
                source_types_new.append(source_type)
                magnus_table_infos.append(None)
        return data_sources_new, source_types_new, magnus_table_infos

    def __getstate__(self):
        d = self.__dict__.copy()
        if self.kv_loader is not None:
            d["kv_loader_state"] = self.kv_loader.state
        for member in STATE_SKIP_SAVE:
            if member in d:
                d.pop(member)
        for member in STATE_PICKLE_SAVE:
            if member in d:
                try:
                    d[member] = pickle.dumps(d[member])
                except pickle.PicklingError:
                    warnings.warn(
                        f'Dataloader member {member} cannot be pickled,'
                        'we will pop this member. Please re-initialize it'
                        'when loading'
                    )
        if self.bitwise_resume:
            self.all_worker_states.update(self.gather_worker_states())
        d['all_worker_states'] = self.all_worker_states
        if 'worker_to_block_signals' in d:
            d.pop('worker_to_block_signals')
        return d

    def __setstate__(self, state):
        # for case of pickle loading, if set bitwise_resume in init fn, ignore ckpt value
        if hasattr(self, "bitwise_resume") and "bitwise_resume" in state:
            state.pop("bitwise_resume")
        for member in STATE_PICKLE_SAVE:
            if member in state:
                try:
                    self.__dict__[member] = pickle.loads(state[member])
                    state.pop(member)
                except Exception:
                    # legacy mode
                    pass
        self.__dict__.update(state)
        # because in seedrl dataloader, they init DistributedCruiseDataLoader with rank 0 in initializing
        # so in resume, they need to keep consistency
        if os.getenv("SEED_RL_ON_RAY", "FALSE") == "TRUE":
            self.rank = 0
            self.world = 1
        else:
            self.rank = get_rank()
            self.world = get_world()
        self.kv_loader = None
        self.torch_iter = None
        self.torch_loader = None
        self.transform_fn = self.processor.transform
        self.batch_transform_fn = self.processor.batch_transform
        self.post_process = getattr(self.processor, 'post_transform', None)
        if self.kv_source_idx:
            self._create_kv_loader()
        self.resume = True
        self.stop_queue = mp.Queue()
        self.resume_buffer = self.all_worker_states.get('resume_buffer', [[] for _ in range(max(self.num_workers, 1))])
        if DIST_ENV.bitwise_ckpt and not self.bitwise_resume:
            warnings.warn("users change to bitwise loader, we force bitwise_resume to True")
            self.bitwise_resume = True
        if not self.bitwise_resume:
            logger.info(f"[bitwise resume] rank: {DIST_ENV.rank} disable bitwise loader resume")
            self.all_worker_states = {}
        else:
            logger.info(f"[bitwise resume] rank: {DIST_ENV.rank} enable bitwise loader resume")
        if not self.shuffle and self.iter_source_idx:
            # create a loader instance when shuffle is False to save time
            self.torch_loader = self._create_iter_loader()

    def restore(self, step, epoch, kv_loader_state={}):
        if step > 0:
            self.resume = True
        self.step = step
        self.epoch = epoch
        self.kv_loader_state = kv_loader_state

    def terminate(self):
        """Shutdown the dataloader and its related threads/processes."""
        # in case users do not trigger stop iteration
        if not self.ask_subprocess_stop:
            self.stop_queue.put(1)
            self.ask_subprocess_stop = True
            if self.torch_iter is not None:
                try:
                    while True:
                        next(self.torch_iter)
                except StopIteration:
                    logger.info(f"rank {DIST_ENV.rank} loader.terminate got Stopiteration")
                    pass
        if self.torch_iter is not None and self.num_workers > 0:
            self.shutdown_torch_iter(force=True)
        if self.kv_loader is not None:
            self.kv_loader.terminate()

    def shutdown_torch_iter(self, force=False):
        if not force and self.persistent_workers:
            return
        # Below code is copied from _shutdown_workers method in torch, but extend timeout for join
        python_exit_status = _utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            return
        if not self.torch_iter._shutdown:
            self.torch_iter._shutdown = True
            try:
                if hasattr(self.torch_iter, '_pin_memory_thread'):
                    self.torch_iter._pin_memory_thread_done_event.set()
                    self.torch_iter._worker_result_queue.put((None, None))
                    self.torch_iter._pin_memory_thread.join()
                    self.torch_iter._worker_result_queue.cancel_join_thread()
                    self.torch_iter._worker_result_queue.close()

                # Exit workers now.
                self.torch_iter._workers_done_event.set()
                for worker_id in range(len(self.torch_iter._workers)):
                    if self.torch_iter._persistent_workers or self.torch_iter._workers_status[worker_id]:
                        self.torch_iter._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self.torch_iter._workers:
                    w.join(timeout=300)
                for q in self.torch_iter._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                if self.torch_iter._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self.torch_iter))
                    self.torch_iter._worker_pids_set = False
                for w in self.torch_iter._workers:
                    if w.is_alive():
                        w.terminate()

    def gather_worker_states(self):
        '''
        This function is to gather worker states for bitwise resuming feature,
        it will 1)temporally block Dataloader workers, then 2)collect buffered data samples
        and  dataset states from them. After gathering all the worker states, it will
        3)release the block so the loader can continue on data-loading.
        '''
        all_worker_states = {}
        if not hasattr(self, "worker_dismiss_block_signals"):
            return self.all_worker_states
        # to block all the workers
        for dismiss_block in self.worker_dismiss_block_signals:
            dismiss_block.clear()
        for to_blocks in self.worker_to_block_signals:
            to_blocks.set()
        back_signal_cnt = set()
        # collect data samples from dataset buffer
        logger.info(f"[bitwise] rank {DIST_ENV.rank} starts workers polling")
        _retry = 0
        # to get poll from all alive workers
        while self._loader_alive:
            if _retry >= VALID_DATA_RETRY and _retry % 100 == 0:
                warnings.warn(f"[bitwise] rank {DIST_ENV.rank} loader detected too many ({_retry}) \
                    on dataset polling, possible hang detected")
            try:
                data, wid = self._get_data_from_loaders()
                if isinstance(data, DatasetPolled):
                    back_signal_cnt.add(wid)
                    workers_status = getattr(self.torch_iter, '_workers_status', [True])
                    alive_worker_polled = [i in back_signal_cnt for i, status in enumerate(workers_status) if status]
                    # if we saw all alive worker has polled
                    if all(alive_worker_polled):
                        break
                elif isinstance(data, DatasetFinished):
                    self._finished_worker[wid] = True
                else:
                    self.resume_buffer[wid].append((data, wid))
                _retry += 1
            except StopIteration:
                logger.info(f"rank {DIST_ENV.rank} loader.gather_worker_states.p0 got Stopiteration")
                self._loader_alive = False
                break
        logger.info(f"[bitwise] rank {DIST_ENV.rank} finished workers polling")
        # collect dataset states from workers
        data_usage_state = {}
        if self._loader_alive:
            logger.info(f"[bitwise] rank {DIST_ENV.rank} starts gathering worker states")
            for wid, state_queue in enumerate(self.worker_state_queues):
                state = {}
                while True:
                    try:
                        workers_status = getattr(self.torch_iter, '_workers_status', [True])
                        if not workers_status[wid]:
                            warnings.warn(f"[bitwise resume] dataloader worker {wid} is dead \
                                and will be skipped in loader ckpt")
                            break
                        state = state_queue.get(timeout=5)
                        break
                    except Exception:
                        warnings.warn(f"[bitwise resume] dataloader trying to get state from worker {wid}, \
                            possible hang detected")
                if not state:
                    continue
                all_worker_states[state['wid']] = state['worker_state']

                if 'data_usage' in state:
                    cur_data_usage = state['data_usage']
                    for url, start, end in cur_data_usage:
                        if url not in data_usage_state:
                            data_usage_state[url] = []
                        data_usage_state[url].append((start, end))
            logger.info(f"[bitwise] rank {DIST_ENV.rank} finished gathering worker states")
        reshard_info = {}
        if data_usage_state:
            reshard_info['data_usage_state'] = data_usage_state
        if reshard_info:
            all_worker_states['reshard_info'] = reshard_info

        all_worker_states['resume_buffer'] = copy.deepcopy(self.resume_buffer)
        # unblock all the workers
        for to_blocks in self.worker_to_block_signals:
            to_blocks.clear()
        for dismiss_block in self.worker_dismiss_block_signals:
            dismiss_block.set()
        # dump signal values from workers, in case that two calls of gather_worker_states gets too
        # closed so that some worker might stay unchanged in polling status, which will cause hang
        # in gather states
        if self.num_workers > 0:
            for _ in self.worker_state_queues:
                try:
                    data, wid = self._get_data_from_loaders()
                    if isinstance(data, DatasetFinished):
                        self._finished_worker[wid] = True
                    if not isinstance(data, (DatasetPolled, DatasetFinished)):
                        self.resume_buffer[wid].append((data, wid))
                except StopIteration:
                    logger.info(f"rank {DIST_ENV.rank} loader.gather_worker_states.p1 got Stopiteration")
                    self._loader_alive = False
                    break
        return all_worker_states

    def _is_valid_data(self, data):
        if data is None or isinstance(data, (DatasetPolled)):
            return False
        if isinstance(data, (list, tuple)) and isinstance(data[0], DatasetPolled):
            return False
        return True

    def _inspect_invalid_data(self, data):
        if isinstance(data, (list, tuple)):
            return f"{type(data)}({[self._inspect_invalid_data(x) for x in data]})"
        else:
            return f"{type(data)}"


if __name__ == '__main__':
    dataloader = DistributedCruiseDataLoader()
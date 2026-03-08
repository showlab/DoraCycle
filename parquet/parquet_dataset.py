#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
CruiseParquetDataset
使用Cruise工具读取parquet数据文件 支持resume、queue shuffling 的功能 参数基本与DistLineReadingDatasetV2对齐
'''
import os
import random
import glob
import torch
from torch.utils.data import IterableDataset
import warnings
# from data_utils import hlist_files, torch_io_load, local_rank_zero_only
from parquet.data_utils import hlist_files, torch_io_load, local_rank_zero_only
try:
    # sys.path.append('/opt/tiger/cruise/')
    from cruise.data_module.hybrid_dataset import DistIterableDataset
    from cruise.data_module.cruise_loader import shard_source
except Exception as e:
    warnings.warn('cruise is not installed, if you are using CruiseParquetDataset, please install if from https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag, otherwise, ignore this warning')
class CruiseParquetDataset(IterableDataset):  # pylint: disable=W0223
    """
    iterate Parquet Dataset.
    TODO(shibiao): Test resume logics.
    """
    def __init__(self,
                 data_path: str,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = False,
                 repeat: bool = False,
                 verbose: bool = True,
                 buffer_size: int = 1,
                 meta_data_path: str = None,
                 state_path: str = None,
                 parquet_cache_on: bool = False,
                 seed: int = 42,
                 num_workers: int = 1,
                 ):
        """
        data_path: 数据文件夹路径，会list出这个文件夹下面的所有file。支持多个文件夹，用 `,` 隔开
        rank: 在多机多卡的情况下，需要根据rank来划分
        world_size: 在多机多卡的情况下，需要根据world_size来划分
        repeat: 是否重复，如果重复的话，在遍历完一遍数据后，会继续重新遍历
        shuffle: 是否shuffle，按file shuffle；以及如果有buffer的话，对buffer shuffle
        verbose: 是否打印一些log
        buffer_size: 是否构造一个buffer 来预存一些数据。这个的好处是配合shuffle可以做到一定程度的打散。1表示不buffer
        meta_data_path: 记录数据meta 信息的config 路径，主要用来load 每个文件的行数
        state_path: 记录 data offset，对于resume 有用
        parquet_cache_on: 是否打开本地cache功能
        """
        super().__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.files = hlist_files(data_path.split(','))
        files = [f for f in self.files if f.find('_SUCCESS') < 0]
        self.files = [f for f in files if not f.endswith('snappy.parquet')]
        # import ipdb
        # ipdb.set_trace()
        # this is for refinedweb dataset
        if len(self.files) == 0:
            self.files = glob.glob(data_path)
        # import ipdb
        # ipdb.set_trace()
        self.files.sort()
        self.is_hdfs = data_path.startswith('hdfs')
        self.data_path = data_path
        self.repeat = repeat
        print(
            '[DATA]--all dataset containing {} files.'.format(len(self.files)))
        if len(self.files) % self.world_size != 0:
            print('[DATA]--Whole dataset file num %s cannot split to worldsize %s ' %
                     (len(self.files), self.world_size))
        self.verbose = verbose
        self.load_data_offsets(state_path)
        # local data buffer
        self.buffer = []
        self.buffer_size = buffer_size
        self.parquet_cache_on = parquet_cache_on
        self._seed = seed
        # here we call `shard_source` in case self.files were not registered in rh2
        cur_rank_files, _, _, _ = shard_source(
            self.files, self.rank, self.world_size, num_workers, "parquet", None, drop_last=False)
        print(cur_rank_files)
        self.cur_rank_files = cur_rank_files
        assert len(self.cur_rank_files[0]) > 0, "Parquet files number too few, need to increase parquet files number"
    def load_data_offsets(self, training_state_path=None):
        """ 加载 data offset """
        self.data_offsets = {}
        if training_state_path is not None:
            training_state = torch_io_load(training_state_path, map_location='cpu')
            self.data_offsets = training_state['data_offsets']
            self._seed = training_state['seed']
            data_offsets_basename = {os.path.basename(k): v for k, v in self.data_offsets.items()}
            local_rank_zero_only(log.info)(f'[Resuming] data offsets: {data_offsets_basename}')
    def generate(self, seed=-1):
        """
        # TODO(shibiao): Add more comments
        """
        if seed > 0:
            self._seed = seed
        if self.shuffle:
            self.files = self.sort_and_shuffle(self.files, self._seed)
        else:
            self.files.sort()
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1
        wid = 0
        if worker_info is not None:
            num_workers = worker_info.num_workers
            wid = worker_info.id
        # print('--------- num workers: ', num_workers)
        # Use 'cur_worker_hash' to replace 'filepath'
        current_worker_hash = str(hash((self.data_path, self.rank, wid)))
        while True:
            if self.shuffle:  # Shuffle parquet source in each epoch
                # Add seed here to make resume available even after shuffle
                random.Random(self._seed).shuffle(self.cur_rank_files)
            # if self.verbose:
            #     print(
            #         f"[DataLoader] --> Rank:[{self.rank}]  Workers:[{worker_info.id if worker_info else 0}] process file: {len(self.cur_rank_files)} :{self.cur_rank_files[:3]}  ..."
            #     )
            prev_offset = self.data_offsets.get(current_worker_hash, 0)
            # Param 'shuffle' in 'DistIterableDataset' will do buffer shuffle only
            # 'DistIterableDataset' will do second sharding inside itself
            pq_dataset = DistIterableDataset(
                self.cur_rank_files, url_format='parquet',
                repeat=False, batch_size=1, shuffle=self.shuffle,
                shuffle_buffer_size=self.buffer_size, parquet_cache_on=self.parquet_cache_on,
                resume_step=prev_offset, seed=self._seed)
            for data_idx, data in enumerate(pq_dataset):
                yield data[0], current_worker_hash, data_idx, self._seed
            if not self.repeat:
                break
            self._seed += 1
    def __iter__(self):
        return self.generate()

    def reset(self, seed):
        del self.buffer
        self.buffer = []
        self._seed = seed
        return self.generate()
    def sort_and_shuffle(self, data, seed):
        data.sort()
        random.Random(seed).shuffle(data)
        return data
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
CruiseParquetDataset

使用Cruise工具读取parquet数据文件 支持resume、queue shuffling 的功能 参数基本与DistLineReadingDatasetV2对齐
'''

import io, os, sys
import random
import torch
from torch.utils.data import IterableDataset
import warnings
from io import BytesIO
import numpy as np
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms as T
import collections
from data_utils import resize_crop
from torchvision.transforms import Compose, Normalize, ColorJitter
import torchvision.transforms.functional as F
import tensorflow as tf
from decord import VideoReader
from data_utils import hlist_files, torch_io_load, local_rank_zero_only
import time
import pandas as pd
import json
try:
    sys.path.append('/opt/tiger/cruise/')
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
        self.files = [f for f in self.files if f.find('_SUCCESS') < 0]
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
        self.cur_rank_files = cur_rank_files
        
        assert all([len(x) for x in self.cur_rank_files]), "Parquet files number too few, need to increase parquet files number"

    def load_data_offsets(self, training_state_path=None):
        """ 加载 data offset """
        self.data_offsets = {}
        if training_state_path is not None:
            training_state = torch_io_load(training_state_path, map_location='cpu')
            self.data_offsets = training_state['data_offsets']
            self._seed = training_state['seed']
            data_offsets_basename = {os.path.basename(k): v for k, v in self.data_offsets.items()}
            local_rank_zero_only(print)(f'[Resuming] data offsets: {data_offsets_basename}')

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
        
        # Use 'cur_worker_hash' to replace 'filepath'
        current_worker_hash = str(hash((self.data_path, self.rank, wid)))
        while True:
            if self.shuffle:  # Shuffle parquet source in each epoch
                # Add seed here to make resume available even after shuffle
                random.Random(self._seed).shuffle(self.cur_rank_files)
            if self.verbose:
                print(
                    f"[DataLoader] --> Rank:[{self.rank}]  Workers:[{worker_info.id if worker_info else 0}] process file: {len(self.cur_rank_files[wid])} :{self.cur_rank_files[wid][:3]}  ..."
                )
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


class CommonMsDatasetParquet(CruiseParquetDataset):
    def __init__(self,
                config,
                data_path,
                rank: int = 0,
                world_size: int = 1,
                shuffle=False,
                repeat=False,
                num_workers=2,
                **kwargs
                ):

        super().__init__(data_path, rank, world_size, shuffle, repeat, verbose=True, buffer_size=0, meta_data_path=None, state_path=None, num_workers=num_workers)
        if config['data_loader'].get('image_h', None) == None:
            self.frame_size = (config['arch']['args']['tokenizer_args']['image_size'], config['arch']['args']['tokenizer_args']['image_size'])
        else:
            self.frame_size = (config['data_loader']['image_h'], config['data_loader']['image_w'])

    def __iter__(self):
        '''
        data keys: width, height,  web_text_t5_feat_npz: binary 
                recaption_t5_feat_npz: binary
                sdxl_vae_256_feat_npy:binary
        '''
        for example in self.generate():
            data, current_worker_hash, data_idx, seed = example
            try:
                if data['img'] is None:
                    continue

                img = tf.image.decode_jpeg(data['img'], channels=3, dct_method='INTEGER_ACCURATE').numpy()
                img = torch.as_tensor(img).float().div_(255).permute(2, 0, 1)

                if random.random() < 0.5:
                    img = F.hflip(img)

                img, _, _ = resize_crop(img, self.frame_size[0], self.frame_size[1])

                aug = Compose([Normalize(0.5, 0.5)])

                img = aug(img)

                ret = {'images': img}
                yield ret
            
            except Exception as e:
                print('parquet dataset iter error', e)
                continue

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)

        for k, v in batched.items():
            batched[k] = torch.stack(v, dim=0)
        return [batched['images']]

class VideoDatasetParquet(CruiseParquetDataset):
    def __init__(self,
                data_path,
                rank: int = 0,
                world_size: int = 1,
                shuffle=False,
                repeat=False,
                num_workers=2,
                frame_augmentation=None,
                random_sample=False,
                **kwargs
                ):

        super().__init__(data_path, rank, world_size, shuffle, repeat, verbose=True, buffer_size=1000, meta_data_path=None, state_path=None, num_workers=num_workers)
        self.num_frames = 17
        self.frame_augmentation = frame_augmentation
        self.random_sample = random_sample
        self.frame_size = (256,256)
        self.frame_interval = 1
        # self.caption_file = "<YOUR_WEBVID_CSV_PATH>"
        self.caption_file = "<YOUR_VIDEO_CAPTION_JSON_PATH>"
        with open(self.caption_file, 'r', encoding='utf-8') as json_file:
            self.captions = json.load(json_file)
        # import pdb; pdb.set_trace()
        
        
        # df = pd.read_csv(self.caption_file)
        # import pdb; pdb.set_trace()
        # df2 = pd.read_csv('<YOUR_WEBVID_TEST_CSV_PATH>')
        # self.df = pd.concat([df1, df2], ignore_index=True)
        # import pdb; pdb.set_trace()
    
    def get_caption(self,videopath):
        # 查找对应的行
        if videopath in self.captions:
            # print("hh----------",videopath)
            return self.captions[videopath]
        else:
            # print("ee----------",videopath)
            return None
    # def get_caption(self,videopath):
    #     try:
    #         value = df.loc[df['key'] == videopath, 'value'].values[0]
    #         print("hhh---------",videopath)
    #         return value
    #     except IndexError:
    #         print("eee---------",videopath)
    #         return None
            

    


    def __iter__(self):

        for example in self.generate():

            data, current_worker_hash, data_idx, seed = example
            try:
                # start_time = time.time()
                if data['images'] is None:
                    continue

                images = data['images']
                frames = VideoReader(io.BytesIO(images))
                num_raw_imgs = len(frames)

                if num_raw_imgs < self.num_frames:
                    print(f'num of raw imgs ({num_raw_imgs}) < num_frames (17), skip and continue, video is {filename}')
                    continue
                
                if self.random_sample:
                    # TODO add random sample by random frame interval
                    pass
                else:
                    max_offset = max(num_raw_imgs - (self.num_frames - 1) * self.frame_interval, 1)
                    start = np.random.randint(0, max_offset)
                    sample_frame_idx = list(range(start, start + self.num_frames * self.frame_interval, self.frame_interval))
                frames = frames.get_batch(sample_frame_idx).asnumpy()

                images = torch.as_tensor(frames).float().div_(255).permute(0, 3, 1, 2)

                if random.random() < 0.5:
                    images = F.hflip(images)
                
                if images.shape[-1] != self.frame_size[1] or images.shape[-2] != self.frame_size[0]:
                    images, _, _ = resize_crop(images, self.frame_size[0], self.frame_size[1])
                
                if self.frame_augmentation:
                    images = self.frame_augmentation(images)

                filename = data['filename']
                caption = self.get_caption(filename)

                ret = {'images': images,"filename":filename,'caption':caption}
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"代码运行时间: {elapsed_time:.4f} 秒")
                yield ret
            
            except Exception as e:
                print('parquet dataset iter error', e)
                continue

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)

        for k, v in batched.items():
            if k=="images":
                batched[k] = torch.stack(v, dim=0)
        return [batched['images'],batched['filename'],batched['caption']]


def create_video_parquet_dataloader(path, device, batch_size, shuffle, repeat, is_train=True):

    dataset = VideoDatasetParquet(
                                    data_path=path,
                                    rank=int(os.environ.get('RANK', '0')),
                                    world_size=int(os.environ.get("WORLD_SIZE", "1")),
                                    shuffle=shuffle,
                                    repeat=repeat,
                                    num_workers=4
                                    )
    # import pdb; pdb.set_trace()

    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                        batch_size=batch_size, 
                                        collate_fn=dataset.collate_fn,
                                        num_workers=4,
                                        pin_memory=True,
                                        drop_last=False,
                                        prefetch_factor=6,
                                        pin_memory_device=str(device)
                                        )

    return data_loader


def create_parquet_dataloader(config, path, device, batch_size, shuffle, repeat, is_train=True):

    dataset = CommonMsDatasetParquet(config=config,
                                    data_path=path,
                                    rank=int(os.environ.get('RANK', '0')),
                                    world_size=int(os.environ.get("WORLD_SIZE", "1")),
                                    shuffle=shuffle,
                                    repeat=repeat,
                                    num_workers=4
                                    )

    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                        batch_size=batch_size, 
                                        collate_fn=dataset.collate_fn,
                                        num_workers=4,
                                        pin_memory=True,
                                        drop_last=True,
                                        prefetch_factor=24,
                                        pin_memory_device=str(device)
                                        )

    return data_loader


if __name__ == '__main__':
    import json

    # config = json.load(open('baseline_debug.json'))
    # config['trainer']['num_data_workers'] = 2
    # config['trainer']['prefetch_factor'] = 6
    # path = "<YOUR_PARQUET_PATH>"
    # path = "<YOUR_PARQUET_PATH>"
    # device = 'cuda:0'
    # batch_size = 64
    # data_loader = create_parquet_dataloader(config=config, path=path, device=device, batch_size=batch_size, shuffle=True, repeat=True)
    # loader_iter = iter(data_loader)

    # nums = 0
    # for index in range(20):
    #     data = next(loader_iter)
    #     print(data[0].shape)
    #     nums += data[0].shape[0]
    #     print(data[0].shape)
    #     print('running', nums)
    # print('finish', nums)



    # path = "<YOUR_VIDEO_PARQUET_PATH>"
    path = "<YOUR_VIDEO_PARQUET_PATH>"
    # path = "<YOUR_VIDEO_PARQUET_PATH>"
    # path = "<YOUR_VIDEO_PARQUET_PATH>"
    # path = "<YOUR_VIDEO_PARQUET_PATH>"
    # path = "<YOUR_VIDEO_PARQUET_PATH>"
    # path = "<YOUR_VIDEO_PROCESSED_PATH>"

    # path = "<YOUR_REFINEDWEB_PARQUET_PATH>"
    device = 'cuda:1'
    batch_size = 1
    data_loader = create_video_parquet_dataloader( path=path, device=device, batch_size=batch_size, shuffle=True, repeat=False)
    loader_iter = iter(data_loader)
    import time

    while(1):
        start_time = time.time()
        video_data = next(loader_iter)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"代码运行时间: {elapsed_time:.4f} 秒")
        import pdb; pdb.set_trace()

    # data = video_data[0][0].unsqueeze(0)
    # filename = video_data[1]
    # import pdb; pdb.set_trace()


    # from torchvision.transforms.functional import to_pil_image, to_tensor
    # import numpy as np
    # import os
    # import imageio

    # for idx, vv in enumerate(data): # vv.shape f c h w
    #     pil_images = [to_pil_image(video) for video in vv] #  video.shape c h w
    #     import pdb; pdb.set_trace()
        # save_path = '/opt/tiger/tokenizer/video_tokenizer/data/debug/test44.mp4'
        # writer = imageio.get_writer(save_path, fps=8, quality=6)
        # for img in pil_images:
        #     writer.append_data(np.array(img))
        # writer.close()








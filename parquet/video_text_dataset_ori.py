import argparse
import collections
import io
import random
import warnings

import numpy as np
import torch
import json
import torchvision.transforms.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from cruise.data_module.tools import dump_processor_cfg, create_cruise_loader
from cruise.data_module.utils import parse_data_source
from parquet.data_utils import resize_crop
from decord import VideoReader
import collections
import random
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from parquet.parquet_dataset import CruiseParquetDataset
from torchvision import transforms
from training.data import remove_prefix

warnings.filterwarnings("ignore",
                        message="Palette images with Transparency expressed in bytes should be converted to RGBA images")


class VideoTextDataset(CruiseParquetDataset):
    def __init__(self,
                data_path,
                rank: int = 0,
                world_size: int = 1,
                shuffle=True,
                repeat=True,
                buffer_size=1000,
                num_workers=1,
                num_frames=17, frame_size=(256, 256), frame_augmentation=None, random_sample=False,
                frame_interval=1, is_captioning=False,
                caption_path='<YOUR_VIDEO_CAPTION_JSON_PATH>',
                **kwargs
                ):
        super().__init__(data_path, rank, world_size, shuffle, repeat, verbose=True, buffer_size=buffer_size, meta_data_path=None, state_path=None, num_workers=num_workers)

        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_augmentation = frame_augmentation
        self.random_sample = random_sample
        self.frame_interval = frame_interval
        self.is_captioning = is_captioning
        with open(caption_path, 'r', encoding='utf-8') as json_file:
            self.captions = json.load(json_file)
        print('------ Video caption loaded. ------')


    def get_caption(self, video_path):
        if video_path in self.captions:
            return self.captions[video_path]
        else:
            return ''

    def __iter__(self):
        for example in self.generate():
            try:
                data, current_worker_hash, data_idx, seed = example

                if data['images'] is None:
                    continue

                images = data['images']
                filename = data['filename']
                frames = VideoReader(io.BytesIO(images))
                num_raw_imgs = len(frames)

                if num_raw_imgs < self.num_frames:
                    print(f'num of raw imgs ({num_raw_imgs}) < num_frames ({self.num_frames}), skip and continue,'
                          f' video is {filename}')
                    continue

                if self.random_sample:
                    # TODO add random sample by random frame interval
                    pass
                else:
                    max_offset = max(num_raw_imgs - (self.num_frames - 1) * self.frame_interval, 1)
                    start = np.random.randint(0, max_offset)
                    sample_frame_idx = list(
                        range(start, start + self.num_frames * self.frame_interval, self.frame_interval))

                frames = frames.get_batch(sample_frame_idx).asnumpy()
                images = torch.as_tensor(frames).float().div_(255).permute(3, 0, 1, 2)

                # if random.random() < 0.5:
                #     images = F.hflip(images)

                if images.shape[-1] != self.frame_size[1] or images.shape[-2] != self.frame_size[0]:
                    images, _, _ = resize_crop(images, self.frame_size[0], self.frame_size[1])

                if self.frame_augmentation:
                    images = self.frame_augmentation(images)

                caption = self.get_caption(filename)
                if isinstance(caption, list):
                    caption = caption[0]

                ret = {'images': images, "filename": filename, "input_ids": caption}
                yield ret

            except Exception as e:
                # print(e)
                continue

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity', "filename"):
                batched[k] = torch.stack(v, dim=0)

        return batched

if __name__ == '__main__':

    # dataset = ImgaeTextDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)
    # dataset = ImgaeTextDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)
    # dataset = ImgaeTextDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)
    # dataset = ImgaeTextDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)
    # dataset = ImgaeTextDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)
    # dataset = ImageTextDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=64)
    dataset = ImageTextDataset("<YOUR_VIDEO_PARQUET_PATH>", num_workers=0)

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset, batch_size=10,
                                  sampler=None, collate_fn=dataset.collate_fn,
                                  # num_workers=2)
                                  num_workers=0)
    for i, batch in enumerate(train_dataloader):
        print(i)
        # continue
        import ipdb
        ipdb.set_trace()
    # for idx, item in enumerate(dataset):
    #     print(item['image'].shape, item['input_ids'])
    #     import ipdb
    #     ipdb.set_trace()
    #     print(item)
    #     if idx > 100:
    #         break
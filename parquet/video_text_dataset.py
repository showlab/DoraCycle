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
from torchvision.transforms import Compose, Normalize, ColorJitter

warnings.filterwarnings("ignore",
                        message="Palette images with Transparency expressed in bytes should be converted to RGBA images")


class VideoTextProcessor:
    @dump_processor_cfg()
    def __init__(self, num_frames=17, frame_size=(256, 256),
                 frame_augmentation=Compose([Normalize(0.5, 0.5)]),
                 random_sample=False, frame_interval=1, is_captioning=False,
                 caption_path='<YOUR_VIDEO_CAPTION_JSON_PATH>'):
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

    def transform(self, data):

        try:
            if data['images'] is None:
                return None

            images = data['images']
            filename = data['filename']
            frames = VideoReader(io.BytesIO(images))
            num_raw_imgs = len(frames)

            if num_raw_imgs < self.num_frames:
                print(f'num of raw imgs ({num_raw_imgs}) < num_frames ({self.num_frames}), skip and continue,'
                      f' video is {filename}')
                return None

            if self.random_sample:
                # TODO add random sample by random frame interval
                pass
            else:
                max_offset = max(num_raw_imgs - (self.num_frames - 1) * self.frame_interval, 1)
                start = np.random.randint(0, max_offset)
                sample_frame_idx = list(range(start, start + self.num_frames * self.frame_interval, self.frame_interval))

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
            return ret

        except Exception as e:
            print(e)
            return None

def create_videotext_dataloader(train_shards_path_or_url, batch_size, num_frames=17, frame_size=(256, 256),
                                num_workers=32, num_readers=32, predefined_steps=-1, drop_last=False, shuffle=True,
                                shuffle_buffer_size=1000,
                                frame_augmentation=Compose([Normalize(0.5, 0.5)]),
                                random_sample=False, frame_interval=1, is_captioning=False
                                ):
    files = parse_data_source(train_shards_path_or_url)[0]
    dataloader = create_cruise_loader(
        files, 'parquet',
        batch_sizes=batch_size,
        num_workers=num_workers,
        num_readers=num_readers,
        processors=VideoTextProcessor(num_frames=num_frames, frame_size=frame_size,
                                      frame_augmentation=frame_augmentation, random_sample=random_sample,
                                      frame_interval=frame_interval, is_captioning=is_captioning),
        # predefined_steps = self.hparams.data_size // self.hparams.train_batch_size // self.trainer.world_size
        predefined_steps=predefined_steps,
        drop_last=drop_last,
        shuffle=shuffle,
        dump_config=True,
        bitwise_resume=True,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=np.random.randint(0, 100000),
    )

    return dataloader


def example():

    # args.data_path = "<YOUR_VIDEO_PARQUET_PATH>"
    args.data_path = "<YOUR_VIDEO_PARQUET_PATH>"

    print('data path \n', args.data_path)
    files = parse_data_source(args.data_path)[0]
    loader = create_cruise_loader(
        files, 'parquet',
        batch_sizes=args.batch_size,
        num_workers=args.num_workers,
        num_readers=args.num_readers,
        processors=VideoTextProcessor(),
        predefined_steps=-1,  # self.hparams.data_size // self.hparams.train_batch_size // self.trainer.world_size,
        drop_last=False,
        shuffle=True,
        dump_config=True,
        bitwise_resume=True,
        shuffle_buffer_size=1000,
    )
    for i, data in enumerate(loader):
        print(data['input_ids'])
        import ipdb
        ipdb.set_trace()
        pixel_values = data['images']  # (b, 3, h, w)
        # for i in range(len(data['input_ids'])):
        #     print(data['input_ids'][i])
        if i % 100 == 0:
            print(i, pixel_values.shape)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-readers", type=int, default=16)
    parser.add_argument("--data-path", type=str, default=["<YOUR_PARQUET_PATH_1>", "<YOUR_PARQUET_PATH_2>"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    example()

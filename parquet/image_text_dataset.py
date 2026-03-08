import argparse
import collections
import random
import warnings
from io import BytesIO

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from cruise.data_module.tools import dump_processor_cfg, create_cruise_loader
from cruise.data_module.utils import parse_data_source
from training.data import remove_prefix
from training.utils import image_transform

warnings.filterwarnings("ignore",
                        message="Palette images with Transparency expressed in bytes should be converted to RGBA images")


class ImageTextProcessor:
    @dump_processor_cfg()
    def __init__(self, image_size=256, is_captioning=False, aes_score=None):
        self.image_size = image_size
        self.is_captioning = is_captioning
        self.image_transform = image_transform
        self.aes_score = aes_score

    def transform(self, data):
        try:
            if self.aes_score is not None:
                if data['aes'] < self.aes_score:
                    return None

            img = Image.open(BytesIO(data['img'])).convert("RGB")

            mode = img.mode
            if mode == 'RGBA':
                img = img.convert('RGBA')
                img = np.array(img)[:, :, :3]
            elif img.mode == "P" and "transparency" in img.info:
                img = img.convert('RGBA')
                img = np.array(img)[:, :, :3]
            elif mode == 'RGB':
                img = img.convert('RGB')
                img = np.array(img)
            elif mode == 'L':
                img = np.array(img.convert('L'))
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            else:
                return None

            img = Image.fromarray(img)

            img = self.image_transform(img, resolution=self.image_size)

            if 'recaption' in data.keys() and data['recaption'] is not None:
                text = data['recaption'].replace('\n', '')
                if not self.is_captioning:
                    text = remove_prefix(text)
                    if random.random() < 0.5:
                        text = text.split('.')[0]
            else:
                text = ''

            sample_id = data['crs_sample_idx']

            ret = {'images': img, 'input_ids': text, 'key': sample_id, 'aes_score': data['aes']}
            return ret

        except Exception as e:
            print(e)
            return None

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('key', 'input_ids'):
                batched[k] = torch.stack(v, dim=0)
        return batched


def create_imagetext_dataloader(train_shards_path_or_url, batch_size, image_size=256, num_workers=64, num_readers=32,
                                predefined_steps=-1, drop_last=False, shuffle=True, shuffle_buffer_size=1000,
                                is_captioning=False
                                ):
    files = parse_data_source(train_shards_path_or_url)[0]
    dataloader = create_cruise_loader(
        files, 'parquet',
        batch_sizes=batch_size,
        num_workers=num_workers,
        num_readers=num_readers,
        processors=ImageTextProcessor(image_size=image_size, is_captioning=is_captioning),
        # predefined_steps = self.hparams.data_size // self.hparams.train_batch_size // self.trainer.world_size
        predefined_steps=predefined_steps,
        drop_last=drop_last,
        shuffle=shuffle,
        dump_config=True,
        bitwise_resume=False,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=np.random.randint(0, 100000),
    )

    return dataloader


def example():
    # args.data_path = [f"<YOUR_PARQUET_PATH_PATTERN>/{i:03d}/*.parquet" for i in range(60)]
    # args.data_path = [f"<YOUR_PARQUET_PATH_PATTERN>/{i:02d}/*.parquet" for i in range(11)]
    args.data_path = ["<YOUR_PARQUET_DATA_PATH>"]
    # args.data_path = "<YOUR_PARQUET_DATA_PATH>"

    print('data path \n', args.data_path)
    files = parse_data_source(args.data_path)[0]
    loader = create_cruise_loader(
        files, 'parquet',
        batch_sizes=args.batch_size,
        num_workers=args.num_workers,
        num_readers=args.num_readers,
        processors=ImageTextProcessor(aes_score=6.0),
        predefined_steps=-1,  # self.hparams.data_size // self.hparams.train_batch_size // self.trainer.world_size,
        drop_last=False,
        shuffle=True,
        dump_config=True,
        bitwise_resume=True,
        shuffle_buffer_size=1000,
    )
    for i, data in enumerate(loader):
        pixel_values = data['images']  # (b, 3, h, w)
        # for i in range(len(data['input_ids'])):
        #     print(data['input_ids'][i])
        print(i, pixel_values.shape, data['aes_score'].mean())
        # import ipdb
        # ipdb.set_trace()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--num-readers", type=int, default=64)
    parser.add_argument("--data-path", type=str, default=["<YOUR_PARQUET_PATH_1>", "<YOUR_PARQUET_PATH_2>"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    example()

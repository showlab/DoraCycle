import os
import random
from PIL import Image
import numpy as np
import torch
# from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
# from torch.utils.data import Dataset
# from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms as T
# from diffusion.data.builder import get_data_path, DATASETS
# from diffusion.utils.logger import get_root_logger
import json
from parquet.parquet_dataset import CruiseParquetDataset
# from parquet_dataset import CruiseParquetDataset
import requests
import logging
from io import BytesIO
import collections
import warnings

# 定义一个警告处理函数
def handle_warning(message, category, filename, lineno, file=None, line=None):
    # 检查特定的警告消息
    if "Palette images with Transparency expressed in bytes should be converted to RGBA images" in str(message):
        # 处理警告的逻辑
        pass
        # 其他处理逻辑...
    else:
        # 其他警告的处理逻辑
        warnings.defaultaction(message, category, filename, lineno, file, line)

warnings.showwarning = handle_warning


class Laion2bDataset(CruiseParquetDataset):
    def __init__(self,
                data_path,
                rank: int = 0,
                world_size: int = 1,
                shuffle=True,
                repeat=True,
                transform=None,
                buffer_size=1000,
                num_workers=1,
                **kwargs
                ):
        super().__init__(data_path, rank, world_size, shuffle, repeat, verbose=True, buffer_size=buffer_size, meta_data_path=None, state_path=None, num_workers=num_workers)

    def image_transform(sefl, image, resolution=256):
        image = T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR)(image)
        # get crop coordinates
        c_top, c_left, _, _ = T.RandomCrop.get_params(image, output_size=(resolution, resolution))
        image = T.functional.crop(image, c_top, c_left, resolution, resolution)
        image = T.ToTensor()(image)
        return image

    def __iter__(self):
        for example in self.generate():
            try:
                data, current_worker_hash, data_idx, seed = example
                img = Image.open(BytesIO(data['IMAGE']))
                mode = img.mode
                if mode == 'RGBA':
                    img = img.convert('RGBA')
                    img = np.array(img)[:, :, :3]
                elif mode == 'RGB':
                    img = img.convert('RGB')
                    img = np.array(img)
                elif mode == 'L':
                    img = np.array(img.convert('L'))
                    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                else:
                    raise Exception('Unsupported mode')

                img = Image.fromarray(img)

                img = self.image_transform(img)

                text = data['TEXT']
                if text is None:
                    text = ''

                sample_id = data['SAMPLE_ID']
                similarity = data['similarity']

                ret = {'image': img, 'input_ids': text, 'key': sample_id, 'similarity': similarity}
                yield ret

            except Exception as e:
                # print('internal dataset iter error', e)
                # import ipdb
                # ipdb.set_trace()
                continue

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)

        return batched

if __name__ == '__main__':
    transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # dataset = Laion2bDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)
    dataset = Laion2bDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)
    # dataset = Laion2bDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)

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
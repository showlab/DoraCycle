import collections
import random
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from parquet.parquet_dataset import CruiseParquetDataset
from torchvision import transforms
from training.data import remove_prefix


class SFTDataset(CruiseParquetDataset):
    def __init__(self,
                data_path,
                rank: int = 0,
                world_size: int = 1,
                shuffle=True,
                repeat=True,
                buffer_size=1000,
                num_workers=1,
                image_size=256,
                **kwargs
                ):
        super().__init__(data_path, rank, world_size, shuffle, repeat, verbose=True, buffer_size=buffer_size, meta_data_path=None, state_path=None, num_workers=num_workers)
        self.image_size = image_size

    def image_transform(sefl, image, resolution=256):
        image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)(image)
        image = transforms.CenterCrop((resolution, resolution))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
        return image

    def __iter__(self):
        for example in self.generate():
            try:
                data, current_worker_hash, data_idx, seed = example
                img = Image.open(BytesIO(data['img']))
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

                img = self.image_transform(img, resolution=self.image_size)

                if 'recaption' in data.keys() and data['recaption'] is not None:
                    text = data['recaption'].replace('\n', '')
                    text = remove_prefix(text)
                    if data['from_dataset'] != 'journeydb' and random.random() < 0.5:
                        text = text.split('.')[0]

                elif 'caption' in data.keys():
                    text = ''
                else:
                    raise Exception('No recaption or caption')

                if not isinstance(text, str):
                    text = ''

                sample_id = data['sample_id']

                ret = {'image': img, 'input_ids': text, 'key': sample_id}
                yield ret

            except Exception as e:
                # print('internal dataset iter error', e)
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

    # dataset = SFTDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)
    # dataset = SFTDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)
    dataset = SFTDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)
    # dataset = SFTDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)
    # dataset = SFTDataset("<YOUR_PARQUET_DATA_PATH>", num_workers=0)

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
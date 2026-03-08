from cruise.data_module.utils import parse_data_source
from cruise.data_module.tools import dump_processor_cfg, create_cruise_loader
import torch
from torchvision import transforms as T
from torchvision.transforms.v2.functional import InterpolationMode
from PIL import Image
from io import BytesIO
import collections
import argparse
import os
class QualityDataProcessor:
    # 装饰器用于导出yaml，如果不需要导出yaml可以不用
    @dump_processor_cfg()
    def __init__(self, image_key):
        self.image_key = image_key
        self.op = T.Compose([
            T.Resize(490, interpolation=InterpolationMode.BICUBIC, max_size=None),
            T.CenterCrop(490),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    def transform(self, data_dict):
        try:
            img = self.op(
                Image.open(BytesIO(data_dict[self.image_key])).convert("RGB")
            )
        except:
            return {}
        ret_dict = {'img': img}
        for k, v in data_dict.items():
            if k in [self.image_key, 'width', 'height']:
                continue
            ret_dict[k] = v
        return ret_dict
    def batch_transform(self, batch_data):
        batched = collections.defaultdict(list)
        for data in batch_data:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k == 'img':
                batched[k] = torch.stack(v, dim=0)
        return batched

def example():
    processors = QualityDataProcessor('img')
    # args.data_path = [f"<YOUR_PARQUET_PATH_PATTERN>/{i:03d}/*.parquet" for i in range(60)]
    args.data_path = [f"<YOUR_PARQUET_PATH_PATTERN>/{i:02d}/*.parquet" for i in range(10)]
    files = parse_data_source(args.data_path)[0]
    loader = create_cruise_loader(
        files, 'parquet',
        batch_sizes=args.batch_size,
        num_workers=args.num_workers,
        num_readers=args.num_readers,
        processors=processors,
        predefined_steps=-1,
        drop_last=False,
        shuffle=False,
        dump_config=True,
        bitwise_resume=True
    )
    for data in loader:
        pixel_values = data['img']  # (b, 3, h, w)
        uids = data['image_phash']
        print(uids)
        import ipdb
        ipdb.set_trace()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--num-readers", type=int, default=64)
    parser.add_argument("--data-path", type=str, default=["<YOUR_PARQUET_PATH_1>", "<YOUR_PARQUET_PATH_2>"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    example()
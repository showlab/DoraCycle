import json
import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import DatasetFolder, default_loader
from torchvision import transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import collections
# import sys
# sys.path.append("..")


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class OpenImagesDataset(DatasetFolder):
    def __init__(
        self,
        root: str,
        caption_path: str = "<YOUR_OPENIMAGES_CAPTION_JSON_PATH>",
        loader: Callable[[str], Any] = default_loader,
        image_size=256,
        word_dropout_prob=0.0,
        without_flickr30k=False,
        only_flickr30k=False,
        flickr30k_path: str = "<YOUR_FLICKR30K_IMAGES_PATH>",
    ):

        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        # self.transform = transforms.Compose([
        #     transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.RandomCrop((image_size, image_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # ])
        self.root = root
        self.samples = os.listdir(self.root)
        with open(caption_path, 'r') as f:
            self.caption_dict = json.load(f)
        self.loader = loader
        self.word_dropout_prob = word_dropout_prob
        if without_flickr30k:
            self.samples = list(set(self.samples) - set(os.listdir(flickr30k_path)))
        if only_flickr30k:
            self.samples = os.listdir(flickr30k_path)

        print(f"Openimages dataset loaded. {self.__len__()} images.")

    def __len__(self):
        return len(self.samples)

    def word_dropout(self, caption):
        caption = caption.split(' ')
        num_elements = len(caption)
        num_to_remove = int(self.word_dropout_prob * num_elements)
        indices_to_remove = random.sample(range(num_elements), num_to_remove)
        remaing_elements = [elem for idx, elem in enumerate(caption) if idx not in indices_to_remove]
        removed_elements = [elem for idx, elem in enumerate(caption) if idx in indices_to_remove]

        return ' '.join(remaing_elements), removed_elements

    def __getitem__(self, idx):

        try:
            path = self.samples[idx]
            image = self.loader(os.path.join(self.root, path))
            image = self.transform(image)
            input_ids = self.caption_dict[path.split('.')[0]]
            if self.word_dropout_prob > 0.0:
                input_ids, removed = self.word_dropout(input_ids)
            return {'images': image, 'input_ids': input_ids}

        except Exception as e:
            print(e)
            return self.__getitem__(idx+1)

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)

        return batched


def create_openvid1m_dataloader():
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader
    dataset = OpenImagesDataset(
        "<YOUR_OPENIMAGES_ROOT>",
        "<YOUR_OPENIMAGES_CAPTION_JSON_PATH>",
        image_size=256,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=int(os.environ.get("WORLD_SIZE", "1")),
        rank=int(os.environ.get("RANK", "0")),
        shuffle=True,
        seed=10086
        )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        num_workers=32,
        pin_memory=True,
        drop_last=True,
        # prefetch_factor=512,
    )

    return loader

if __name__ == '__main__':
    # from omegaconf import DictConfig, ListConfig, OmegaConf
    # # yaml_conf = OmegaConf.load('../configs/laion12m_phismall_6x8_gpus_with_lm_zero_grad_sp_head_muse_setting_predict_next_multidata_fp16.yaml')
    # yaml_conf = OmegaConf.load('./configs/phi_6x8_gpus_pretraining_stage_imagenet.yaml')
    # config = OmegaConf.merge(yaml_conf)
    # preproc_config = config.dataset.preprocessing
    # dataset_config = config.dataset.params
    # dataset_cls = OpenImagesDataset(
    #     "<YOUR_OPENIMAGES_ROOT>",
    #     "<YOUR_OPENIMAGES_CAPTION_JSON_PATH>",
    #     image_size=preproc_config.resolution,
    # )
    # print("Length: ", len(dataset_cls))
    # from torch.utils.data import DataLoader
    # from torch.utils.data.distributed import DistributedSampler
    #
    # # if accelerator.num_processes > 1:
    # #     sampler = DistributedSampler(dataset_lm)
    # # else:
    # #     sampler = None
    #
    # train_dataloader = DataLoader(dataset_cls, batch_size=config.training.batch_size,
    #                                  sampler=None, collate_fn=dataset_cls.collate_fn,
    #                                  shuffle=True,  num_workers=0)
    # # num_workers=0)
    #
    # # train_dataloader = DataLoader(dataset_cls, batch_size=config.training.batch_siz,
    # #                               sampler=sampler, collate_fn=dataset_cls.collate_fn,
    # #                               num_workers=dataset_config.num_workers)
    # #                             # num_workers=0)

    train_dataloader = create_openvid1m_dataloader()
    for i, batch in enumerate(train_dataloader):
        print(i)



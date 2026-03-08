import os
import collections
from typing import Any, Callable
import torch
from datasets import load_dataset
from torchvision import transforms
from collections import defaultdict
import random

def image_transform(image, resolution=256, normalize=True):    
    # If image has a single channel, convert it to 3 channels by copying
    if image.mode == 'L':
        # print(f"image.mode {image.mode}")
        image = image.convert('RGB')
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image

class CusDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str = "explanation_1",
        train_split: str = "train",
        special_mark: str = "SKS_STYLE",
        image_size=512,
        sample_mode='image',
        sub_ratio=1.0,
        paired_ratio=1.0,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.transform = image_transform
        self.sub_ratio = sub_ratio
        self.paired_ratio = paired_ratio
        self.special_mark = special_mark
        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_name, split)[train_split]
        # print(dataset)
        data = [item for item in dataset]
        
        # Subsample data based on sub_ratio
        if sub_ratio < 1.0:
            data = random.sample(data, int(len(data) * sub_ratio))
        
        # Process data based on paired_ratio
        processed_data = []
        for item in data:
            # print(item)
            image = item.get('image')
            synopses = item.get('image_description')
            
            if random.random() < paired_ratio:
                processed_data.append({'key_frames': image, 'synopses': synopses})  # Fully paired data
            else:
                if sample_mode == 'image':
                    processed_data.append({'key_frames': image, 'synopses': None})  # Image only, no text
                elif sample_mode == 'text':
                    if len(synopses) == 0:
                        continue
                    processed_data.append({'key_frames': None, 'synopses': synopses})  # Text only, no image
                else:
                    processed_data.append({'key_frames': image, 'synopses': synopses})  # Default paired data
        
        self.data = processed_data
        self.sample_mode = sample_mode
        self.sample_len = len(self.data)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item.get('key_frames')
        synopsis = item.get('synopses')

        # Load image
        if image is not None:
            image = self.transform(image, resolution=self.image_size)  # Apply resizing and cropping transformation

        # If synopsis is a string, use it directly
        input_ids = None
        if synopsis is not None:
            if isinstance(synopsis, list) and len(synopsis) > 0:
                input_ids = random.choice(synopsis)
            else:
                input_ids = synopsis

        return {
            'images': image if image is not None else None,
            'input_ids': self.special_mark + " " + input_ids if input_ids is not None else None,
        }

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)

        for k, v in batched.items():
            if k == 'images':
                batched[k] = [item for item in v if item is not None]
                if len(batched[k]) > 0:
                    batched[k] = torch.stack(batched[k], dim=0)
                else:
                    batched[k] = None

        return batched

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # Load dataset from Hugging Face
    dataset_name = "<YOUR_CUS_DATA_PATH>"
    dataset = CusDataset(dataset_name=dataset_name, image_size=512, sub_ratio=0.5, paired_ratio=0.8, sample_mode='image')

    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # Load one sample per batch
        shuffle=False, 
        collate_fn=dataset.collate_fn
    )

    # Iterate through the first batch of the dataset
    for batch in dataloader:
        print("Batch of data:")
        print("Images:", batch['images'])  # Print image tensor
        print("Input IDs:", batch['input_ids'])  # Print text descriptions
        break  # Print only the first batch's content

import torch
import os
from PIL import Image
import numpy as np
import config
import utils
from torch.utils.data import Dataset

class HorseZebraDataset(Dataset):
    def __init__(self, horse_root, zebra_root, transform=None):
        self.horse_root = horse_root
        self.zebra_root = zebra_root
        self.transform = transform

        self.horse_images = os.listdir(horse_root)
        self.zebra_images = os.listdir(zebra_root)

        self.length_dataset = max(len(self.horse_images), len(self.zebra_images))
        self.horse_length = len(self.horse_images)
        self.zebra_length = len(self.zebra_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, idx):
        zebra_image = self.zebra_images[idx % self.length_dataset]
        horse_image = self.horse_images[idx % self.length_dataset]

        zebra_path = os.path.join(self.zebra_root, zebra_image)
        horse_path = os.path.join(self.horse_root,horse_image)

        zebra_image = np.array(Image.open(zebra_path).convert('RGB'))
        horse_image = np.array(Image.open(horse_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=zebra_image,image0=horse_image)
            zebra_image = augmentations['image']
            horse_image = augmentations['image0']

        return zebra_image, horse_image

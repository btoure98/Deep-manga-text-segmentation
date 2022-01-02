from __future__ import print_function, division
import cv2
import os
import random
import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


class MangaDataset(Dataset):
    def __init__(self, img_paths, target_paths, transform=None, transform_mask=None):
        self.img_paths = img_paths
        self.target_paths = target_paths
        self.transform = transform
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        target_path = self.target_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        target = cv2.imread(target_path, cv2.IMREAD_COLOR)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        _, target = cv2.threshold(target, 100, 255, cv2.THRESH_OTSU)
        if self.transform:
            image = self.transform(image)
            target = self.transform_mask(target)
        return image, target

    def viz_random(self):
        index = random.randint(0, self.__len__())
        sample_img, sample_target = self.__getitem__(index)
        _ , axs = plt.subplots(1,2)
        axs[0].imshow(sample_img)
        axs[1].imshow(sample_target)
        plt.show()
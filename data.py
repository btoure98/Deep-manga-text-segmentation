from __future__ import print_function, division

import cv2
import random

import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()


class MangaDataset(Dataset):
    def __init__(self, img_paths, target_paths, bbox_paths=None, transform=None, transform_mask=None):
        self.img_paths = img_paths
        self.target_paths = target_paths
        self.bbox_paths = bbox_paths
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
        if self.bbox_paths:
            bbox_path = self.bbox_paths[idx]
            bbox_mask = cv2.imread(bbox_path, cv2.IMREAD_COLOR)
            bbox_mask = cv2.cvtColor(bbox_mask, cv2.COLOR_BGR2GRAY)
            _, bbox_mask = cv2.threshold(bbox_mask, 100, 255, cv2.THRESH_OTSU)
            return image, target, bbox_mask

        return image, target

    def viz_random(self):
        index = random.randint(0, self.__len__())
        sample_img, sample_target = self.__getitem__(index)
        _, axs = plt.subplots(1, 2)
        axs[0].imshow(sample_img)
        axs[1].imshow(sample_target)
        plt.show()

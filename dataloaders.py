import os
import cv2
import torch

from torch.utils.data import Dataset
from torchvision import datasets

from file_sorter import FileSorter

class ImageLoader(Dataset):
    def __init__(self, configs, extention="jpg", transform=None, target_transform=None):
        #self.image_dir = configs.source
        self.configs = configs
        self.extention = extention
        self.image_list = os.listdir(self.configs.source)
        self.transform = transform
        self.target_transform = target_transform
        self.file_sorter = FileSorter()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        _, image_file_paths = self.file_sorter.get_sorted_names(self.configs.source, "*." + self.extention)
        image = cv2.imread(image_file_paths[idx])
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        
        return image, image_file_paths[idx]
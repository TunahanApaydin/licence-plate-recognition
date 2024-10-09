import os
import cv2
import torch

from torch.utils.data import Dataset
from torchvision import datasets

from file_sorter import FileSorter

class ImageLoader(Dataset):
    def __init__(self, configs, extention="jpg", transform=None, target_transform=None):
        self.configs = configs
        self.extention = extention
        self.image_list = os.listdir(self.configs.source)
        self.transform = transform
        self.target_transform = target_transform
        self.file_sorter = FileSorter()
        _, self.image_file_paths = self.file_sorter.get_sorted_names(self.configs.source, "*." + self.extention)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_file_paths[idx])
        width = image.shape[1] 
        height = image.shape[0] 

        return image, (int(height), int(width)), self.image_file_paths[idx]

class VideoLoader(object):
    def __init__(self, configs, extention="mp4"):
        self.configs = configs
        self.extention = extention
        self.video_list = os.listdir(self.configs.source)
        self.capture = None
        self.width = None  
        self.height = None
        self.file_sorter = FileSorter()
        _, self.video_file_paths = self.file_sorter.get_sorted_names(self.configs.source, "*." + self.extention)
        self.__video_capture(self.video_file_paths[0])
        
    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        
        self.capture.grab()
        ret, frame = self.capture.retrieve()
        
        while not ret:
            self.capture.release()

            self.count += 1
            if self.count == len(self.video_list): # TODO: get only video file count
                raise StopIteration
            
            self.__video_capture(self.video_file_paths[self.count])
            ret, frame = self.capture.read()
            
        return frame, (int(self.height), int(self.width)), self.video_file_paths[self.count]
        # capture = cv2.VideoCapture(self.video_file_paths[idx])
        # width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  
        # height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # assert capture.isOpened(), "Failed to open {}".format(self.configs.source)
        
        # #print(video)
        # while capture.isOpened():
        #     ret, image = capture.read()
        #     cv2.imwrite("test_videos/res/" + str(counter) + ".jpg", image)
        #     counter += 1
        #     print(counter)
        #     if not ret:
        #         capture.release()
        #         print("Break")
        #         break
                
        #     return image, (int(height), int(width)), self.video_file_paths[idx]
    
    def __video_capture(self, video_path):
        self.capture = cv2.VideoCapture(video_path)
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)  
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
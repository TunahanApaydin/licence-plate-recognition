import os
import cv2
from torch.utils.data import Dataset
from file_sorter import get_sorted_file_paths

class ImageLoader(Dataset):
    def __init__(self, configs):
        self.configs = configs
        self.image_list = os.listdir(self.configs.source)
        _, self.image_file_paths = get_sorted_file_paths(self.configs.source, self.configs.image_extensions)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_file_paths[idx])
        width = image.shape[1] 
        height = image.shape[0] 

        return image, (int(height), int(width)), self.image_file_paths[idx]

class VideoLoader(object):
    def __init__(self, configs):
        self.configs = configs
        self.video_list = os.listdir(self.configs.source)
        self.capture = None
        self.width = None  
        self.height = None

        _, self.video_file_paths = get_sorted_file_paths(self.configs.source, self.configs.video_extensions)
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
    
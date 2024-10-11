import yaml
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import os
from PIL import Image
from scipy import special
import colorsys
import random
import argparse
import sys
import time
import platform

from file_sorter import FileSorter

class YOLOv9ONNXInference(object):
    
    def __init__(self, configs) -> None:
        self.inference_session = None
        self.model_inputs = None
        self.input_names = None
        self.input_shape = None
        self.model_output = None
        self.output_names = None
        self.model_input_height = 640
        self.model_input_width = 640
        self.image_input_height = None
        self.image_input_width = None
        self.configs = configs
        
    
    def preprocess(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.model_input_width, self.model_input_height))

        # Scale input pixel value to 0 to 1
        input_image = resized / 255.0
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        
        return input_tensor
    
    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y 

    def postprocess(self, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.configs.conf_thres, :]
        scores = scores[scores > self.configs.conf_thres]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Rescale box
        boxes = predictions[:, :4]
        
        input_shape = np.array([self.model_input_width, self.model_input_height, self.model_input_width, self.model_input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_input_width, self.image_input_height, self.image_input_width, self.image_input_height])
        boxes = boxes.astype(np.int32)
        
        # xmins = [box[0] for box in boxes]
        # ymins = [box[1] for box in boxes]
        # xmaxs = [box[2] for box in boxes]
        # ymaxs = [box[3] for box in boxes]
            
        # pred = list(zip(xmins, ymins, xmaxs, ymaxs, scores, class_ids))
            
        #indices = nms(boxes, scores, class_ids, 0.45)
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.configs.conf_thres, nms_threshold=self.configs.iou_thres)
        detections = []
        for bbox, score, label in zip(self.xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
            detections.append({
                "class_index": label,
                "confidence": score,
                "box": bbox,
                "class_name": "lp"
            })
        return detections
    
    def read_class_names():
        '''loads class name from a file'''
        with open('inference.yaml', 'r') as inference_config_file:
            inference_configs = yaml.load(inference_config_file, Loader=yaml.SafeLoader)
            
        names = {}
        classes = inference_configs["class_names"]
        for id, name in enumerate(classes):
            names[id] = name 
        # names = {}
        # with open(class_file_name, 'r') as data:
        #     for ID, name in enumerate(data):
        #         names[ID] = name.strip('\n')
        return names
    
    def draw_bbox(self, img, detections):
        for detection in detections:
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = detection['box'].astype(int)
            class_id = detection['class_index']
            confidence = detection['confidence']

            # Retrieve the color for the class ID
            color = (255,255,0)

            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Create the label text with class name and score
            label = "lp"

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )

            # Draw the label text on the image
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img
    
    def onnx_session(self):
        #session_opt = ort.SessionOptions()
        
        if self.configs.device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            
        self.inference_session = ort.InferenceSession(self.configs.weight, providers=providers)
        
        self.model_inputs = self.inference_session.get_inputs()
        self.input_names = [self.model_inputs[i].name for i in range(len(self.model_inputs))]
        self.input_shape = self.model_inputs[0].shape
        self.model_output = self.inference_session.get_outputs()
        self.output_names = [self.model_output[i].name for i in range(len(self.model_output))]
        #self.model_input_height, self.model_input_width = self.input_shape[2:] # TODO: find a way to get model input shape
        
    # def check_model_file(self):
    #     if not os.path.exists(self.inference_configs["weights"]):
    #         raise Exception("The specified file: {} was not found!".format(self.inference_configs["weights"]))
        
    #     extension = os.path.splitext(self.inference_configs["weights"])[-1].lower()
    #     assert extension == ".onnx" , "{}, is an unknown file format. Use .onnx format".format(extension)
    
    @classmethod
    def read_config_file(cls, file_path: str) -> None:
        with open(file_path, 'r') as inference_config_file:
            cls.inference_configs = yaml.load(inference_config_file, Loader=yaml.SafeLoader)
    
    def run(self): 
        # Validate model file path
        #self.check_model_file()
    
        #self.onnx_session()
        
        if self.configs.image:
            if not os.path.isdir(self.configs.source):
                raise FileNotFoundError("The specified file: {} was not found!".format(self.configs.source))
            
            # TODO: write dataloader instead of this
            file_sorter = FileSorter()
            _, image_file_paths = file_sorter.get_sorted_names(self.configs.source, "*.jpg")
            
            for image_path in image_file_paths:
                total_start = time.time()
                
                image_name = os.path.basename(image_path)
                image = cv2.imread(image_path)
                self.image_input_height, self.image_input_width, _ = image.shape
                
                preprocess_start = time.time()
                input_tensor = self.preprocess(image)
                preprocess_end = time.time()
                
                inference_start = time.time()
                outputs = self.inference_session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
                inference_end = time.time()
                
                postprocess_start = time.time()
                detection = self.postprocess(outputs=outputs)
                postprocess_end = time.time()
                
                draw_start = time.time()
                osd_image = self.draw_bbox(image, detection)
                draw_end = time.time()
                
                #cv2.imwrite(self.inference_configs["result_save_path"] + image_name, osd_image.astype(np.uint8))
                
                total_end = time.time()
                
                print(image_name)
                preprocess_time = preprocess_end - preprocess_start
                inference_time = inference_end - inference_start
                postprocess_time = postprocess_end - postprocess_start
                draw_time = draw_end - draw_start
                total_time = total_end - total_start
                
                print("-*-"*50)
                print("preprocess_time: ", preprocess_time)
                print("inference_time: ", inference_time)
                print("postprocess_time: ", postprocess_time)
                print("draw_time: ", draw_time)
                print("total_time: ", total_time)
                print("FPS: ", 1.0/total_time)

# if __name__ == "__main__":
#     inference_conf_path = "inference.yaml"
#     YOLOv9ONNXInference().read_config_file(file_path=inference_conf_path)
    
#     inference = YOLOv9ONNXInference()
#     inference.run()
    
#     print("spgsdgb")
    
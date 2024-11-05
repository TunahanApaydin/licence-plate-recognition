import cv2
import copy
import numpy as np
import multiprocessing
import onnxruntime as ort
from tracking import Tracking

class YOLOv9ONNXInference(object):
    def __init__(self, configs, ocr) -> None:
        self.configs = configs
        
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

        self.inf_input_q = multiprocessing.Queue()
        self.inf_res_q = multiprocessing.Queue()
        
        print("Inference module INIT")
        
        self.tracking = Tracking(self.configs, ocr)
    
    def start_processes(self):
        inf_process = multiprocessing.Process(target=self.inference) 
        inf_process.start()
        
        print("inference function started in process")

    
    def __preprocess(self, frame, frame_shape):

        self.image_input_height, self.image_input_width = frame_shape
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (self.model_input_width, self.model_input_height))

        input_frame = resized / 255.0
        input_frame = input_frame.transpose(2,0,1)
        input_tensor = input_frame[np.newaxis, :, :, :].astype(np.float32)
        
        return input_tensor
    
    def inference(self):
        self.onnx_session()
        while True:
            if self.inf_input_q.qsize() > 0:
                
                message_dict = self.inf_input_q.get().copy()
                message_dict_copy = copy.deepcopy(message_dict)
                input_tensor = self.__preprocess(message_dict_copy["frame"], message_dict_copy["frame_shape"])


                outputs = self.inference_session.run(self.output_names, {self.input_names[0]: input_tensor})[0]

                if np.any(outputs):
                    message_dict_copy["detect_bboxes"], message_dict_copy["detection_scores"] = self.__postprocess(message_dict_copy["frame_shape"], outputs)

                    if message_dict_copy["detect_bboxes"].any() and message_dict_copy["detection_scores"].any():
                        self.tracking.tracking_input_q.put(message_dict_copy)
                else:
                    pass
            else:
                pass
                
    def __postprocess(self, frame_shape, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.configs.conf_thres, :]
        scores = scores[scores > self.configs.conf_thres]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = predictions[:, :4]
        scale_x = frame_shape[1] / self.model_input_width
        scale_y = frame_shape[0] / self.model_input_height

        x_center_orig = boxes[:, 0] * scale_x
        y_center_orig = boxes[:, 1] * scale_y
        width_orig = boxes[:, 2] * scale_x
        height_orig = boxes[:, 3] * scale_y

        x1 = x_center_orig - (width_orig / 2)
        y1 = y_center_orig - (height_orig / 2)
        x2 = x_center_orig + (width_orig / 2)
        y2 = y_center_orig + (height_orig / 2)
        
        xyxy_boxes = np.column_stack((x1, y1, x2, y2)).astype(int) 
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.configs.conf_thres, nms_threshold=self.configs.iou_thres)
        
        detections = []
        for bbox, score, label in zip(xyxy_boxes[indices], scores[indices], class_ids[indices]):
            detections.append({
                "class_index": label,
                "confidence": score,
                "box": bbox,
                "class_name": "lp"
            })

        scores = np.array([])
        bboxes = np.array([])
        if detections:
            scores = np.array([detection["confidence"] for detection in detections])
            bboxes = np.array([detection["box"] for detection in detections])

        return bboxes, scores

            
    def onnx_session(self):
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
        
    
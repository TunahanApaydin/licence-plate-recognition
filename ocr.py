import re
import os
import cv2
import easyocr
import numpy as np


class OCR(object):
    def __init__(self, configs) -> None:
        self.configs = configs
        self.reader = easyocr.Reader(lang_list=self.configs.lang_list) # this needs to run only once to load the model into memory

    def preprocess(self, frame: np.ndarray, tracked_bboxes: list) -> np.ndarray:
        if tracked_bboxes:
            for bbox in tracked_bboxes:
                x1, y1, w, h = bbox
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                
                xyxy = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                ocr_input = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

                return ocr_input
        
    def ocr(self, ocr_input: np.ndarray) -> list:
        if np.any(ocr_input):
            ocr_results = self.reader.readtext(ocr_input,
                                               allowlist = self.configs.allowlist,
                                               detail = 1,
                                               text_threshold = self.configs.text_threshold,
                                               min_size = self.configs.min_size,
                                               ycenter_ths = self.configs.ycenter_ths,
                                               height_ths = self.configs.height_ths,
                                               width_ths = self.configs.width_ths,
                                               contrast_ths = self.configs.contrast_ths,
                                               adjust_contrast = self.configs.adjust_contrast,
                                               decoder=self.configs.decoder)
            
            # for (bbox, text, prob) in results:
            #     print(f'Text: {text}, Probability: {prob}, Bbox: {bbox}')
                
            return ocr_results

    def postprocess(self, ocr_results: list, ocr_input) -> str:
        licence_plate_texts = []
        if ocr_results:
            for (bbox, text, prob) in ocr_results:
                xmin = int(bbox[0][0])
                ymin = int(bbox[0][1])
                xmax = int(bbox[2][0])
                ymax = int(bbox[2][1])
                
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0  
                    
                bbox_area = ocr_input[ymin:ymax, xmin:xmax]
                #cv2.imwrite("ocr_results/"+ text + ".jpg", bbox_area)
                
                bbox_height = ymax - ymin
                if (bbox_height >= self.configs.min_bbox_height) and (len(text) <= self.configs.max_detected_text_length): 
                    #print(text)
                    licence_plate_texts.append(text)
            
            licence_plate_text = "".join(licence_plate_texts)[:self.configs.max_lp_text_length]
            #print("LP: {}".format(licence_plate_text))  
            if (len(licence_plate_text) >= self.configs.min_lp_text_length) and (len(licence_plate_text) <= self.configs.max_lp_text_length):
                # numbers = re.findall(r'[0-9]+', licence_plate_text)
                # alphabets = re.findall(r'[a-zA-Z]+', licence_plate_text)
                if licence_plate_text[:2] in self.configs.provincial_lp_codes:
                    if re.fullmatch(self.configs.lp_pattern_regex, licence_plate_text): #Turkish LP template check (99 Y 9999, 99 Y 99999, 99 YY 999, 99 YY 9999, 9 YYY 99, 99 YYY 999)
                        print("LP: {}".format(licence_plate_text)) 
                        return licence_plate_text

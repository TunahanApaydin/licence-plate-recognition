import re
import copy
import easyocr
import numpy as np
import multiprocessing

class OCR(object):
    _instance = None
    _lock = multiprocessing.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OCR, cls).__new__(cls)
        return cls._instance

    def __init__(self, configs):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.configs = configs
            self.ocr_input_q = multiprocessing.Queue()
            self.ocr_res_q = multiprocessing.Queue()
            self.reader = easyocr.Reader(lang_list=self.configs.lang_list) 
            print("OCR Initialized")
            self.start_processes()
                
    def start_processes(self):
        ocr_process = multiprocessing.Process(target=self.ocr) 
        ocr_process.start()
    
        print("ocr function started in process")
        
    def __preprocess(self, frame, tracked_bboxes) -> np.ndarray:

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
            
    def ocr(self) -> list:
        while True:
            if self.ocr_input_q.qsize() > 0:
                message_dict = self.ocr_input_q.get()
                message_dict_copy = copy.deepcopy(message_dict)
                ocr_input = self.__preprocess(message_dict_copy["frame"], message_dict_copy["tracked_bboxes"])

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
                    if ocr_results:
                        licence_plate_text = self.__postprocess(ocr_results)
                        message_dict_copy["licence_plate_text"] = licence_plate_text
                        
                        self.ocr_res_q.put(message_dict_copy)

            else:
                continue

    def __postprocess(self, ocr_results) -> str:

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
                
                bbox_height = ymax - ymin
                if (bbox_height >= self.configs.min_bbox_height) and (len(text) <= self.configs.max_detected_text_length): 
                    licence_plate_texts.append(text)
            
            licence_plate_text = "".join(licence_plate_texts)[:self.configs.max_lp_text_length]
            #print("LP: {}".format(licence_plate_text))  
            if (len(licence_plate_text) >= self.configs.min_lp_text_length) and (len(licence_plate_text) <= self.configs.max_lp_text_length):
                #print("LP1: {}".format(licence_plate_text))
                if licence_plate_text[:2] in self.configs.provincial_lp_codes:
                    #print("LP2: {}".format(licence_plate_text))
                    if re.fullmatch(self.configs.lp_pattern_regex, licence_plate_text): #Turkish LP template check (99 Y 9999, 99 Y 99999, 99 YY 999, 99 YY 9999, 9 YYY 99, 99 YYY 999)
                        #print("LP3: {}".format(licence_plate_text))
                        return licence_plate_text

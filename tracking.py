import copy
import multiprocessing
from tracker.byte_tracker import BYTETracker

class Tracking(object):
    def __init__(self, configs, ocr) -> None:
        self.configs = configs
        self.byte_track = BYTETracker(self.configs)
        self.tracking_input_q = multiprocessing.Queue()
        self.tracking_res_q = multiprocessing.Queue()
        print("Tracking module INIT")
        
        self.ocr_instance = ocr
        self.start_processes()
        
    def start_processes(self):
        tracking_process = multiprocessing.Process(target=self.tracking) 
        tracking_process.start()
        
        print("tracking function started in process")
        
    def tracking(self):
        while True:
            if self.tracking_input_q.qsize() > 0:
                message_dict = self.tracking_input_q.get()
                message_dict_copy = copy.deepcopy(message_dict)
                online_targets = self.byte_track.update(message_dict_copy["detection_scores"], message_dict_copy["detect_bboxes"], [self.configs.imgsz[0], self.configs.imgsz[1]], self.configs.imgsz)
                online_tlwhs = []
                online_ids = []

                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > self.configs.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > self.configs.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                
                if online_tlwhs:
                    message_dict_copy["tracked_bboxes"] = online_tlwhs
                    message_dict_copy["tracking_ids"] = online_ids
                    
                    self.ocr_instance.ocr_input_q.put(message_dict_copy)

                    
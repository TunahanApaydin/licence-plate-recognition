import yaml

class Configurations(object):
    def __init__(self, inference_conf_path: str) -> None:
        self.inference_conf_path = inference_conf_path
        self.inference_configs = None
        self.__read_config_file()
        
        self.weight = self.inference_configs["inference"]["weight"]
        self.source = self.inference_configs["inference"]["source"]
        self.image = self.inference_configs["inference"]["image"]
        self.video = self.inference_configs["inference"]["video"]
        self.result_save_path = self.inference_configs["inference"]["result_save_path"]
        self.data = self.inference_configs["inference"]["data"]
        self.imgsz = self.inference_configs["inference"]["imgsz"]
        self.conf_thres = self.inference_configs["inference"]["conf_thres"]
        self.iou_thres = self.inference_configs["inference"]["iou_thres"]
        self.max_det = self.inference_configs["inference"]["max_det"]
        self.device = self.inference_configs["inference"]["device"]
        self.view_img = self.inference_configs["inference"]["view_img"]
        self.class_names = self.inference_configs["inference"]["class_names"]
        
        self.track_thresh = self.inference_configs["tracking"]["track_thresh"]
        self.track_buffer = self.inference_configs["tracking"]["track_buffer"]
        self.match_thresh = self.inference_configs["tracking"]["match_thresh"]
        self.frame_rate = self.inference_configs["tracking"]["frame_rate"]
        self.aspect_ratio_thresh = self.inference_configs["tracking"]["aspect_ratio_thresh"]
        self.min_box_area = self.inference_configs["tracking"]["min_box_area"]
        self.mot20 = self.inference_configs["tracking"]["mot20"]
        
        self.text_threshold = self.inference_configs["ocr"]["text_threshold"]
        self.min_size = self.inference_configs["ocr"]["min_size"]
        self.ycenter_ths = self.inference_configs["ocr"]["ycenter_ths"]
        self.height_ths = self.inference_configs["ocr"]["height_ths"]
        self.width_ths = self.inference_configs["ocr"]["width_ths"]
        self.min_bbox_height = self.inference_configs["ocr"]["min_bbox_height"]
        self.max_detected_text_length = self.inference_configs["ocr"]["max_detected_text_length"]
        self.min_lp_text_length = self.inference_configs["ocr"]["min_lp_text_length"]
        self.max_lp_text_length = self.inference_configs["ocr"]["max_lp_text_length"]
        self.lang_list = self.inference_configs["ocr"]["lang_list"]
        self.allowlist = self.inference_configs["ocr"]["allowlist"]
        self.lp_pattern_regex = self.inference_configs["ocr"]["lp_pattern_regex"]
        self.provincial_lp_codes = self.inference_configs["ocr"]["provincial_lp_codes"]
        
        print("\nConfigurations:\n")
        for key, value in self.inference_configs.items():
            print(f"{key}: {value}")

    def __read_config_file(self) -> None:
        try:
            with open(self.inference_conf_path, 'r') as inference_config_file:
                self.inference_configs = yaml.load(inference_config_file, Loader=yaml.SafeLoader)
        except Exception as error:
            print(error)







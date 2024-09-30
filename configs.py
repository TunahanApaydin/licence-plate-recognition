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
        
        print("\nInference Configs:\n")
        for key, value in self.inference_configs.items():
            print(f"{key}: {value}")

    def __read_config_file(self) -> None:
        try:
            with open(self.inference_conf_path, 'r') as inference_config_file:
                self.inference_configs = yaml.load(inference_config_file, Loader=yaml.SafeLoader)
        except Exception as error:
            print(error)







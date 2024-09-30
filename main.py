import os
import cv2
import time
import numpy as np

from torch.utils.data import DataLoader

from detect import YOLOv9ONNXInference
from file_sorter import FileSorter
from configs import Configurations
from dataloaders import ImageLoader
from tracker.byte_tracker import BYTETracker


def check_model_file():
    if not os.path.exists(configs.weight):
        raise Exception("The specified file: {} was not found!".format(configs.weight))
    
    extension = os.path.splitext(configs.weight)[-1].lower()
    assert extension == ".onnx" , "{}, is an unknown file format. Use .onnx format".format(extension)
    
def run():
    check_model_file()

    inference.onnx_session()
    
    byte_track = BYTETracker(configs)
    results = []

    if configs.image:
        if not os.path.isdir(configs.source):
            raise FileNotFoundError("The specified file: {} was not found!".format(configs.source))
        
        # TODO: write dataloader instead of this
        file_sorter = FileSorter()
        _, image_file_paths = file_sorter.get_sorted_names(configs.source, "*.jpg")
        
        dataset = ImageLoader(configs=configs, extention="jpg")
        #image_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=12, shuffle=False, pin_memory=False)
        
        for frame_id, (image, image_path) in enumerate(dataset):
        
        #for frame_id, image_path in enumerate(image_file_paths):
            total_start = time.time()
            
            image_name = os.path.basename(image_path)
            # image = cv2.imread(image_path)
            inference.image_input_height, inference.image_input_width, _ = image.shape
            
            preprocess_start = time.time()
            input_tensor = inference.preprocess(image)
            preprocess_end = time.time()
            
            inference_start = time.time()
            outputs = inference.inference_session.run(inference.output_names, {inference.input_names[0]: input_tensor})[0]
            inference_end = time.time()
            
            postprocess_start = time.time()
            detections = inference.postprocess(outputs=outputs)
            postprocess_end = time.time()
            
            if detections is not None:
                scores = np.array([detection["confidence"] for detection in detections])
                bboxes = np.array([detection["box"] for detection in detections])

                online_targets = byte_track.update(scores, bboxes, [configs.imgsz[0], configs.imgsz[1]], configs.imgsz)

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > configs.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > configs.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n")
                        print(results)            
            draw_start = time.time()
            osd_image = inference.draw_bbox(image, detections)
            draw_end = time.time()
            
            #cv2.imwrite(inference.inference_configs["result_save_path"] + image_name, osd_image.astype(np.uint8))
            
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

if __name__ == "__main__":
    inference_conf_path = "inference.yaml"
    configs = Configurations(inference_conf_path)
    
    inference = YOLOv9ONNXInference(configs)
    run()
    
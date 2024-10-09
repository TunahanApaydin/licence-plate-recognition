import os
import cv2
import time
import numpy as np

from torch.utils.data import DataLoader

from detect import YOLOv9ONNXInference
from file_sorter import FileSorter
from configs import Configurations
from dataloaders import ImageLoader, VideoLoader
from tracker.byte_tracker import BYTETracker
from visualize import plot_tracking
from ocr import OCR

def check_model_file():
    if not os.path.exists(configs.weight):
        raise Exception("The specified file: {} was not found!".format(configs.weight))
    
    extension = os.path.splitext(configs.weight)[-1].lower()
    assert extension == ".onnx" , "{}, is an unknown file format. Use .onnx format".format(extension)

def check_source_file():
    if configs.source == None:
        raise TypeError("Specify image or video path in {} configuration file".format(inference_conf_path))
    elif not os.path.isdir(configs.source):
        raise FileNotFoundError("The specified file: {} was not found!".format(configs.source))
    
def run():
    check_source_file()
    check_model_file()

    inference.onnx_session()
    
    byte_track = BYTETracker(configs)
    ocr = OCR(configs)
    results = []

    # if configs.image:
    #     if not os.path.isdir(configs.source):
    #         raise FileNotFoundError("The specified file: {} was not found!".format(configs.source))
        
    #     
    #     # file_sorter = FileSorter()
    #     # _, image_file_paths = file_sorter.get_sorted_names(configs.source, "*.jpg")
        
    if configs.video:
        dataset = VideoLoader(configs=configs, extention="mp4") # TODO: add other video formats
    elif configs.image:
        dataset = ImageLoader(configs=configs, extention="jpg")
    else:
        raise Exception("Specify the input type(exp: video: true, image: false) you want to use in the {} file".format(inference_conf_path))
    
    # vid_writer = cv2.VideoWriter(configs.result_save_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (int(1440), int(1440)))
    # counter = 0
    cv2.namedWindow("lp_number", cv2.WINDOW_NORMAL)
    for frame_id, (frame, frame_shape, path) in enumerate(dataset):
        total_start = time.time()
        
        image_name = os.path.basename(path)
        # image = cv2.imread(path)
        inference.image_input_height, inference.image_input_width = frame_shape
        
        preprocess_start = time.time()
        input_tensor = inference.preprocess(frame)
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
                    #results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n")

            ocr_input = ocr.preprocess(frame, zip(online_tlwhs, online_ids, online_scores))
            ocr_results = ocr.ocr(ocr_input)
            licence_plate_number = ocr.postprocess(ocr_results, ocr_input)
            online_im = plot_tracking(frame, online_tlwhs, online_ids, licence_plate_number, frame_id=frame_id + 1)
        cv2.imshow("lp_number", online_im)
        cv2.waitKey(10)
        # print(results) 
        # cv2.imwrite(configs.result_save_path + str(counter) + ".jpg", online_im.astype(np.uint8))
        # counter += 1
        #vid_writer.write(online_im)                       
        # draw_start = time.time()
        # osd_image = inference.draw_bbox(image, detections)
        # draw_end = time.time()
        
        #cv2.imwrite(inference.inference_configs["result_save_path"] + image_name, osd_image.astype(np.uint8))
        
        total_end = time.time()
        
        #print(image_name)
        preprocess_time = preprocess_end - preprocess_start
        inference_time = inference_end - inference_start
        postprocess_time = postprocess_end - postprocess_start
        #draw_time = draw_end - draw_start
        total_time = total_end - total_start
        
        # print("-*-"*50)
        # print("preprocess_time: ", preprocess_time)
        # print("inference_time: ", inference_time)
        # print("postprocess_time: ", postprocess_time)
        # print("draw_time: ", draw_time)
        # print("total_time: ", total_time)
        # print("FPS: ", 1.0/total_time)

if __name__ == "__main__":
    inference_conf_path = "configs.yaml"
    configs = Configurations(inference_conf_path)
    
    inference = YOLOv9ONNXInference(configs)
    run()
    
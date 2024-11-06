import os
import time
import copy
import multiprocessing

from ocr import OCR
from visualize import Visualize
from configs import Configurations
from database import DB
from detect import YOLOv9ONNXInference
from dataloaders import ImageLoader, VideoLoader


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

    ocr = OCR(configs)
    time.sleep(1)
    
    inference = YOLOv9ONNXInference(configs, ocr)
    inference.start_processes()
    time.sleep(2)
    
    db = DB(database_name=configs.database_name)
    db.connect_database()
    
    osd = Visualize(configs)
    
    is_registered_status = {}
    
    message_dict = {"frame_id": None,
                    "frame": None,
                    "frame_shape": None,
                    "detect_bboxes": None,
                    "detection_scores": None,
                    "tracked_bboxes": None,
                    "tracking_ids": None,
                    "licence_plate_text": None}
   
    if configs.video:
        dataset = VideoLoader(configs=configs)
    elif configs.image:
        dataset = ImageLoader(configs=configs)
    else:
        raise Exception("Specify the input type(exp: video: true, image: false) you want to use in the {} file".format(inference_conf_path))
    
    for frame_id, (frame, frame_shape, path) in enumerate(dataset):

        start_time = time.time() 
        
        message_dict["frame_id"] = frame_id
        message_dict["frame"] = frame
        message_dict["frame_shape"] = frame_shape
        
        inference.inf_input_q.put(message_dict)
        
        if not ocr.ocr_res_q.empty():
            ocr_res_msg_dict = ocr.ocr_res_q.get()
            ocr_res_msg_dict_copy = copy.deepcopy(ocr_res_msg_dict)
            
            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time if elapsed_time > 0 else 0
            #print("FPS: ", fps)
            
            is_registered, user = db.check_plate_in_database(table_name="users", plate=ocr_res_msg_dict_copy["licence_plate_text"])
            
            tracking_id = ocr_res_msg_dict_copy["tracking_ids"][0]
            
            if tracking_id not in is_registered_status:

                is_registered_status[tracking_id] = {
                    "is_registered": is_registered,
                    "plate_text": ocr_res_msg_dict_copy["licence_plate_text"] if is_registered else None
                }
            elif is_registered_status[tracking_id]["is_registered"] is False and is_registered:
                is_registered_status[tracking_id]["is_registered"] = True
                is_registered_status[tracking_id]["plate_text"] = ocr_res_msg_dict_copy["licence_plate_text"]

            is_registered_for_visual = is_registered_status[tracking_id]["is_registered"]
            licence_plate_text_for_visual = (is_registered_status[tracking_id]["plate_text"]
                                            if is_registered_for_visual
                                            else ocr_res_msg_dict_copy["licence_plate_text"])

            if configs.save_result or configs.show_result:
                osd.osd(ocr_res_msg_dict_copy["frame"],
                        ocr_res_msg_dict_copy["tracked_bboxes"],
                        ocr_res_msg_dict_copy["tracking_ids"],
                        licence_plate_text_for_visual,
                        is_registered_for_visual,
                        ocr_res_msg_dict_copy["frame_id"],
                        fps=fps,
                        text_scale=configs.text_scale,
                        text_thickness=configs.text_thickness,
                        line_thickness=configs.line_thickness)
                
        message_dict = {"frame_id": frame_id,
                        "frame": None,
                        "frame_shape": None,
                        "detect_bboxes": None,
                        "detection_scores": None,
                        "tracked_bboxes": None,
                        "tracking_ids": None,
                        "licence_plate_text": None}
                
        if ocr.ocr_res_q.empty():                
            time.sleep(0.05)        
    
    db.close_connection()
           
if __name__ == "__main__":
    inference_conf_path = "configs.yaml"
    configs = Configurations(inference_conf_path)
    
    multiprocessing.set_start_method('spawn', force=True)
    
    run()
    
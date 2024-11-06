# License Plate Recognition Project
This project is a comprehensive system for detecting, tracking, recognizing, and validating vehicle license plates in real time. Leveraging deep learning and computer vision techniques, it accurately processes license plate information with efficient handling of multiple steps through multiprocessing. Key technologies include YOLOv9 for object detection, BYTETrack for tracking, easyOCR for text recognition, and SQLite for data management.

![Sample Result](https://raw.githubusercontent.com/TunahanApaydin/licence-plate-recognition/master/demo_video/demo_video_thumbnail.png)


## Demo Video

[![Watch the video - 0.4x Slow Motion](https://raw.githubusercontent.com/TunahanApaydin/licence-plate-recognition/master/demo_video/demo_video_thumbnail.png)](https://github.com/TunahanApaydin/licence-plate-recognition/raw/master/demo_video/demo_video.mp4)




## Project Overview
The main objectives of this project include:

1- **License Plate Detection:** Using YOLOv9 trained models, the system identifies and localizes license plates within video frames, performing inference efficiently with ONNX technology.  
2- **Object Tracking:** BYTETrack efficiently tracks each detected license plate, allowing accurate frame-by-frame tracking.  
3- **Optical Character Recognition (OCR):** With easyOCR, detected license plates are converted into text format for further processing.  
4- **Database Validation:** The recognized text is compared with records in an SQLite database to verify the license plate's registration status.  
5- **Efficient Multiprocessing:** All components utilize multiprocessing, ensuring that each module operates in parallel for real-time performance.  

## Project Structure
```
license-plate-recognition/      # Main project folder
│
├── main.py                     # Main script for running the project
├── detect.py                   # YOLOv9 model inference code
├── tracking.py                 # BYTETrack tracking code
├── ocr.py                      # OCR processing code using EasyOCR
├── database.py                 # Database-related functions for plate registration
├── dataloader.py               # Dataloader for load images or video files
├── file_sorter.py              # Helper functions for dataloader module
├── configs.py                  # Configuration settings in Python format
├── configs.yaml                # Configuration file for project parameters (paths, model, settings)
├── visualize.py                # Visualization code for displaying results
├── registered_lp.db            # Database for registered licence plates
├── README.md                   # Project documentation file
│
├── models/                     # Directory for model files
│   └── your_model.onnx         # The trained YOLOv9 model file in ONNX format
│
├── tracker/                    # Directory for tracking-related code (BYTETrack)
│   ├── __init__.py
│   └── byte_tracker.py         # BYTETrack code for tracking
│
├── result_videos/              # Directory for result videos
│   └── output_video.mp4        # Resulting videos after processing
│
├── source/                     # Directory for source images and videos
│   ├── videos/                 # Videos for processing
│   │   └── sample_video.mp4    # Sample video file
│   └── images/                 # Images for processing
│       └── sample_image.jpg    # Sample image file
```

`main.py`: Manages the workflow, handling input frames and coordinating processes.  
`detect.py`: Runs YOLOv9 detection on each frame using ONNX for high-speed inference.  
`tracking.py`: Implements BYTETrack, tracking the plates between frames and ensuring stability in the OCR process.  
`ocr.py`: Runs the OCR process to convert detected plate images into text.  
`database.py`: Handles interactions with an SQLite database to validate recognized plates.  
`visualize.py`: Visualizes the results, overlaying detection boxes, tracking IDs, and recognition text on video frames.  

## Usage
You can configure settings for inference, tracking, OCR, database, and OSD using the `configs.yaml` file.  For example, you can specify the path to your model file using the `weights` parameter, and the path to your input video/image with the `source` parameter. 

Then, run `main.py`. The results will be saved to the path specified in the configuration file.

## Results and Performance
The License Plate Recognition system demonstrates significant improvements in both accuracy and performance after the integration of multiprocessing. Below are key details:

| Metric                        | Without Multiprocessing | With Multiprocessing |
|-------------------------------|-------------------------|----------------------|
| **FPS (Frames per second)**   | 30 FPS                  | 150 FPS              |
| **License Plate Detection**   | Slower processing       | Faster processing    |
| **Tracking**                  | Lower performance       | High efficiency      |
| **OCR Processing**            | Time-consuming          | Real-time OCR        |

By distributing the workload across multiple processes, the system scales efficiently, allowing for better handling of high-volume video streams with reduced latency.

### Without Multiprocessing:
Before implementing multiprocessing, the system was capable of processing video frames at an average of 30 FPS (frames per second). This was sufficient for basic functionality but became a bottleneck for larger-scale or real-time applications, especially when processing high-resolution videos or when many vehicles were in the frame.

### With Multiprocessing:
By utilizing Python's multiprocessing library, we have offloaded several tasks to multiple processes, significantly improving the throughput. This approach resulted in an average processing speed of 150 FPS or higher, providing a substantial boost in efficiency. Each step of the pipeline ,license plate detection, tracking and OCR, processing concurrently, reducing the overall time required to process each frame.

## Future Improvements

Future improvements for the License Plate Recognition system include:

- **Integration of custom OCR detection and recognition models** to improve accuracy and performance, particularly for specific license plate styles or languages.
- **Optimization of the tracking algorithm** to handle more complex scenarios and improve real-time processing speed.
- **User interface development** for more intuitive visualization of the recognition process.
- **Extended dataset for training models** to improve the system's generalization across various environments and lighting conditions.

These enhancements aim to make the system more robust, adaptable, and scalable in diverse use cases.

## Acknowledgements

<details>
  <summary>Expand</summary>

  - [YOLOv9](https://github.com/WongKinYiu/yolov9/tree/main)
  - [ONNX Runtime Inference Examples](https://github.com/microsoft/onnxruntime-inference-examples)
  - [YOLOv9 ONNX](https://github.com/danielsyahputra/yolov9-onnx/tree/master)
  - [ByteTrack](https://github.com/ifzhang/ByteTrack/tree/main/yolox/tracker)
  - [EasyOCR](https://github.com/JaidedAI/EasyOCR)

</details>





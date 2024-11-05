import cv2
import numpy as np

class Visualize(object):
    def __init__(self, configs) -> None:
        self.configs = configs
        if self.configs.save_result:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(self.configs.result_save_path + 'result.mp4', fourcc, 30.0, (1920,  1080), isColor=True)

    def __get_color(self, idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

        return color

    def osd(self, frame, tlwhs, obj_ids, licence_plate_number, is_registered, frame_id=0, fps=0., text_scale=2, text_thickness=3, line_thickness=3):
        if self.configs.show_result:
            cv2.namedWindow("Licence Plate Recognition", cv2.WINDOW_NORMAL)        
        
        img = np.ascontiguousarray(np.copy(frame))
        
        if is_registered:
            cv2.putText(img, 'FPS: %.2f Vehicle: %s' % (fps, "Registered"),
                            (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=2)
        else:
            cv2.putText(img, 'FPS: %.2f Vehicle: %s' % (fps, "Not Registered"),
                                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        
        if tlwhs != None:
            for i, tlwh in enumerate(tlwhs):
                x1, y1, w, h = tlwh
                intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                obj_id = int(obj_ids[i])
                id_text = '{}'.format(int(obj_id))
                
                if is_registered:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                
                if licence_plate_number:
                    text = id_text + " - " + licence_plate_number
                else:
                    text = id_text
                    
                cv2.rectangle(img, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
                cv2.putText(img, text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, color, thickness=text_thickness)
            
            if self.configs.save_result:
                self.video_writer.write(img)
                
            if self.configs.show_result:
                cv2.imshow("Licence Plate Recognition", img)
                cv2.waitKey(1)
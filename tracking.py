# %%
from ultralytics import YOLO
import os
import cv2

# %%
class ObjectTracking:
    def __init__(self):
        #path for video, 0 for webcam 
        #ex ./Videos/Test_1.mp4
        #ex for webcam 0,1,2 if you have multiple
        self.video = 0
        self.tracker_config = 'bytetrack.yaml' 
        self.model = YOLO('weights (2).pt')

    def track_object(self):
        cap = cv2.VideoCapture(self.video)
        
        # Verify video opened
        if not cap.isOpened():
            print(f"Error opening video file: {self.video}")
            return

        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            results = self.model.track(source=frame, persist=True, tracker=self.tracker_config, conf=0.55, iou=0.4)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
                    cv2.putText(frame, f"Id {track_id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            cv2.imshow("frame", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# %%
def run_track_object():
    ot = ObjectTracking()
    ot.track_object()

# %%
run_track_object()



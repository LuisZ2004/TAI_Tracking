import depthai as dai
from ultralytics import YOLO
import cv2
import numpy as np

class ObjectTracking:
    def __init__(self):
        self.tracker_config = 'bytetrack.yaml' 
        self.model = YOLO('weights2_21_2026.pt')

        self.pipeline = dai.Pipeline()
        
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.xoutVideo = self.pipeline.create(dai.node.XLinkOut)
        self.xoutVideo.setStreamName("video")
        
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P) 
        
        # 1. Set a preview size (YOLO typically expects 640x640 or 640x480)
        self.camRgb.setPreviewSize(640, 480) 
        # 2. Tell it to interleave the color channels (HWC format for OpenCV)
        self.camRgb.setInterleaved(True)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        # 3. Link the 'preview' output instead of 'video'
        self.camRgb.preview.link(self.xoutVideo.input)

    def track_object(self):
        # Connect to OAK-D and start pipeline
        with dai.Device(self.pipeline) as device:
            # Output queue will be used to get the video frames from the camera
            video_queue = device.getOutputQueue(name="video", maxSize=1, blocking=False)
            
            print("Successfully connected to OAK-D. Starting video stream...")

            while True:
                # Fetch the frame from the OAK-D
                video_in = video_queue.get()
                frame = video_in.getCvFrame()

                # Run YOLO tracking on the frame
                results = self.model.track(source=frame, persist=True, tracker=self.tracker_config, conf=0.55, iou=0.4)
                
                # Check if objects are detected and tracked
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                    for box, track_id in zip(boxes, track_ids):
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
                        cv2.putText(frame, f"Id {track_id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # Display the frame
                cv2.imshow("OAK-D YOLO Tracking", frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
        cv2.destroyAllWindows()

def run_track_object():
    ot = ObjectTracking()
    ot.track_object()

if __name__ == "__main__":
    run_track_object()
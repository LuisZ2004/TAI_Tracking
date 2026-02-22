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

            print("Successfully connected to OAK-D. Grabbing a test frame...")

            # Fetch a single frame
            video_in = video_queue.get()
            frame = video_in.getCvFrame()

            # Save the frame as an image file on the Pi
            cv2.imwrite("test_frame.jpg", frame)
            print("Saved 'test_frame.jpg'. Check your folder to see if the image is valid!")
                    
        cv2.destroyAllWindows()

def run_track_object():
    ot = ObjectTracking()
    ot.track_object()

if __name__ == "__main__":
    run_track_object()
import depthai as dai
from ultralytics import YOLO
import cv2

class ObjectTracking:
    def __init__(self):
        self.tracker_config = 'bytetrack.yaml'
        self.model = YOLO('weights2_21_2026.pt')

    def track_object(self):
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera).build()
            queue = cam.requestOutput(
                (640, 480),
                type=dai.ImgFrame.Type.BGR888p,
                fps=30
            ).createOutputQueue()

            pipeline.start()
            print("Successfully connected to OAK-D. Starting video stream...")

            while pipeline.isRunning():
                video_in = queue.get()
                if video_in is None:
                    continue

                frame = video_in.getCvFrame()

                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_config,
                    conf=0.55,
                    iou=0.4,
                    verbose=False
                )

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    for box, track_id in zip(boxes, track_ids):
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
                        cv2.putText(frame, f"Id {track_id}", (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                cv2.imshow("OAK-D YOLO Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cv2.destroyAllWindows()

def run_track_object():
    ot = ObjectTracking()
    ot.track_object()

if __name__ == "__main__":
    run_track_object()

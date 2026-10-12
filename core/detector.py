
import threading
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model_lock = threading.Lock()


def detect_people(frame, confidence_threshold=0.15):
    with model_lock:
        result = model.predict(
            source=frame,
            imgsz=416,
            classes=[0],
            conf=confidence_threshold,
            verbose=False
        )[0]

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0].item())

        detections.append([
            int(x1),
            int(y1),
            int(x2 - x1),
            int(y2 - y1),
            confidence
        ])

    return detections

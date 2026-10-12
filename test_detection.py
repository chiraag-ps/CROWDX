
import cv2
from ultralytics import YOLO

PHONE_URL = "http://192.168.1.11:8080/video"

model = YOLO("yolov8n.pt")

for name, source in [("Laptop", 0), ("Phone", PHONE_URL)]:
    print(f"\nTesting {name} camera...")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"{name}: Could not open camera")
        cap.release()
        continue

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print(f"{name}: Could not read a frame")
        continue

    results = model.predict(
        frame,
        imgsz=640,
        conf=0.10,
        classes=[0],
        verbose=False
    )

    result = results[0]
    count = len(result.boxes) if result.boxes is not None else 0

    print(f"{name}: Person detections = {count}")

    for box in result.boxes:
        print(f"  Confidence: {float(box.conf[0]):.3f}")

    annotated = result.plot()
    cv2.imshow(f"{name} - YOLO Test", annotated)
    cv2.waitKey(0)

cv2.destroyAllWindows()


import cv2
from core.detector import detect_people

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open laptop camera")
    raise SystemExit(1)

ok, frame = cap.read()
cap.release()

if not ok:
    print("Could not read laptop frame")
    raise SystemExit(1)

detections = detect_people(frame, confidence_threshold=0.15)

print("CrowdX detector output:", detections)
print("Detection count:", len(detections))

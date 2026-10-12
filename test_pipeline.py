
import cv2
from core.detector import detect_people
from core.tracker import track_people
from core.density_engine import compute_density
from core.risk_engine import compute_risk

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read laptop camera")

detections = detect_people(frame, confidence_threshold=0.15)
print("1. Detections:", detections)

objects = track_people(frame, detections, camera_id="test")
print("2. Tracked objects:", objects)

density = compute_density(objects)
print("3. Density:", density)

risk_map, count, overcrowd, zones = compute_risk(density)
print("4. Final count:", count)
print("5. Overcrowded:", overcrowd)
print("6. Risk zones:", zones)

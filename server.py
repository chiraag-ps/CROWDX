"""CrowdX Flask server with separate capture and AI-processing workers.

Run from the CrowdX project root:
    python server.py
Then open http://localhost:5000
"""

import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit

try:
    import config.settings as settings
    FRAME_WIDTH = int(getattr(settings, "FRAME_WIDTH", 640))
    FRAME_HEIGHT = int(getattr(settings, "FRAME_HEIGHT", 480))
except Exception:
    FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# Optional MongoDB support: CrowdX still runs when MongoDB is not installed/running.
MONGO_OK = False
DB_NAME = "crowdx"
alerts_col = snapshots_col = None
try:
    from pymongo import MongoClient, DESCENDING
    MONGO_URI = "mongodb://localhost:27017/"
    _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
    _mongo_client.admin.command("ping")
    _db = _mongo_client[DB_NAME]
    alerts_col = _db["alerts"]
    snapshots_col = _db["snapshots"]
    alerts_col.create_index([("timestamp", DESCENDING)])
    alerts_col.create_index([("camera", 1)])
    snapshots_col.create_index([("timestamp", DESCENDING)])
    MONGO_OK = True
    print("[MongoDB] Connected to local MongoDB")
except Exception as exc:
    print(f"[MongoDB] Not connected ({exc}). Running without database.")

from core.detector import detect_people
from core.tracker import track_people
from core.density_engine import compute_density
from core.risk_engine import compute_risk
from core.trend_engine import TrendEngine
from utils.visualizer import draw
from utils.sound_alert import play_alert
from utils.location import get_location
from alerts.alert_manager import send_alert

app = Flask(__name__)
app.config["SECRET_KEY"] = "crowdx-local-dev-key"
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

CAMERA_SOURCES = {
    "cam1": 0,
    "cam2": "http://192.168.1.11:8080/video",  # Update if your phone's IP changes.
}
CONF_TH = 0.30
SIM_MODE = False
SIM_TH = 3
INFERENCE_INTERVAL = 0.10  # target at most 10 inference cycles/second/camera
SNAPSHOT_INTERVAL = 10

# frames: JPEGs consumed by the browser. Updated on every captured frame,
# not only when YOLO finishes, so the visible video stays smoother.
frames = {"cam1": None, "cam2": None}
frame_locks = {k: threading.Lock() for k in frames}
raw_frames = {"cam1": None, "cam2": None}
raw_locks = {k: threading.Lock() for k in frames}

# Latest AI overlay for each camera. Capture workers read this between frames.
overlays = {
    k: {"objects": [], "risk_map": {}, "count": 0, "overcrowd": False,
        "overcrowd_zones": [], "trend": {"trend_label": "STABLE", "growth_rate": 0.0,
        "predicted_count": 0, "predicted_crash": False}}
    for k in frames
}
overlay_locks = {k: threading.Lock() for k in frames}

state = {
    "cam1": {"total": 0, "overcrowd": False, "zones": {}, "overcrowd_zones": [],
             "trend": {"trend_label": "STABLE", "growth_rate": 0.0, "predicted_count": 0, "predicted_crash": False},
             "best_zone": None, "safe_zones": []},
    "cam2": {"total": 0, "overcrowd": False, "zones": {}, "overcrowd_zones": [],
             "trend": {"trend_label": "STABLE", "growth_rate": 0.0, "predicted_count": 0, "predicted_crash": False},
             "best_zone": None, "safe_zones": []},
    "global_status": "SAFE",
    "message_for_crowd": {"status": "SAFE", "headline": "Zone Safe", "body": "Monitoring...", "action": "All clear", "color": "green"},
}
state_lock = threading.Lock()
alert_active = {"cam1": False, "cam2": False}
trend_engines = {"cam1": TrendEngine(), "cam2": TrendEngine()}
last_snapshot = {"cam1": 0.0, "cam2": 0.0}


def blank_jpg(message="No Signal"):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(image, message[:48], (35, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    cv2.putText(image, "Check camera connection", (35, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buffer.tobytes() if ok else b""


def to_jpg(frame, quality=75):
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes() if ok else blank_jpg("Frame encoding failed")


def maps_link():
    try:
        location = get_location()
        if location and len(location) >= 2 and location[0] is not None:
            return f"https://www.google.com/maps?q={location[0]},{location[1]}"
    except Exception:
        pass
    return "Location unavailable"


def db_save_alert(alert_type, camera, count, zones, growth_rate, location_url):
    if not MONGO_OK:
        return
    try:
        alerts_col.insert_one({
            "timestamp": datetime.now(timezone.utc), "type": alert_type, "camera": camera,
            "people_count": int(count), "unsafe_zones": list(zones),
            "growth_rate": round(float(growth_rate), 3), "location_url": location_url,
            "resolved": False,
        })
    except Exception as exc:
        print(f"[MongoDB] Alert save error: {exc}")


def db_resolve_alerts(camera):
    if not MONGO_OK:
        return
    try:
        alerts_col.update_many({"camera": camera, "resolved": False},
                               {"$set": {"resolved": True, "resolved_at": datetime.now(timezone.utc)}})
    except Exception as exc:
        print(f"[MongoDB] Resolve error: {exc}")


def db_save_snapshot(camera, count, trend_label, growth_rate, overcrowd, unsafe_zones):
    if not MONGO_OK:
        return
    try:
        snapshots_col.insert_one({
            "timestamp": datetime.now(timezone.utc), "camera": camera,
            "people_count": int(count), "trend": trend_label,
            "growth_rate": round(float(growth_rate), 3), "overcrowd": bool(overcrowd),
            "unsafe_zones": list(unsafe_zones),
        })
    except Exception as exc:
        print(f"[MongoDB] Snapshot save error: {exc}")


def crowd_msg(cam):
    trend = cam["trend"].get("trend_label", "STABLE")
    count = cam["total"]
    safe_zones = cam.get("safe_zones", [])
    if cam["overcrowd"] or trend == "SURGING":
        return {"status": "DANGER", "headline": "OVERCROWDING DETECTED",
                "body": f"{count} people detected. Check high-risk zones.",
                "action": f"Move toward Zone {safe_zones[0]}" if safe_zones else "Move toward the nearest safe exit",
                "color": "red"}
    if trend == "GROWING":
        return {"status": "WARNING", "headline": "Crowd Growing", "body": f"{count} people detected. Monitor crowd movement.",
                "action": "Move toward open space", "color": "orange"}
    return {"status": "SAFE", "headline": "Zone Safe", "body": f"{count} people detected. No current warning.",
            "action": "Continue monitoring", "color": "green"}


def capture_thread(source, key):
    """Continuously capture and publish frames independently of AI inference."""
    while True:
        print(f"[{key}] Opening camera source: {source}")
        cap = cv2.VideoCapture(source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if key == "cam1":
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, 30)
            if not cap.isOpened():
                print(f"[{key}] Could not open camera; retrying in 2 seconds")
                cap.release()
                time.sleep(2)
                continue

            failures = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 12:
                        print(f"[{key}] Camera read failed; reconnecting")
                        break
                    time.sleep(0.03)
                    continue
                failures = 0
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
                with raw_locks[key]:
                    raw_frames[key] = frame.copy()

                # Draw only the latest known AI result on every fresh camera frame.
                with overlay_locks[key]:
                    overlay = dict(overlays[key])
                try:
                    shown = draw(frame.copy(), overlay["objects"], overlay["risk_map"],
                                 overlay["count"], overlay["overcrowd"],
                                 overlay["overcrowd_zones"], overlay["trend"])
                except Exception:
                    shown = frame
                encoded = to_jpg(shown)
                with frame_locks[key]:
                    frames[key] = encoded
        except Exception as exc:
            print(f"[{key}] Capture error: {exc}")
        finally:
            cap.release()
        time.sleep(0.5)


def processing_thread(key):
    """Run detection/tracking on the newest frame; never build a frame backlog."""
    trend_engine = trend_engines[key]
    last_inference = 0.0
    last_snapshot_time = 0.0
    last_frame_stamp = 0
    location_url = maps_link()  # Avoid a location lookup on every inference cycle.
    while True:
        with raw_locks[key]:
            frame = None if raw_frames[key] is None else raw_frames[key].copy()
        if frame is None:
            time.sleep(0.05)
            continue
        now = time.time()
        if now - last_inference < INFERENCE_INTERVAL:
            time.sleep(0.005)
            continue
        last_inference = now
        try:

            detections = detect_people(frame, confidence_threshold=0.15)
            print(f"[{key}] YOLO detections: {len(detections)}")

            objects = track_people(frame, detections, camera_id=key)
            print(f"[{key}] Tracked objects: {len(objects)}")

            density = compute_density(objects)
            risk_map, count, overcrowd, overcrowd_zones = compute_risk(density)

            trend_input = (
                {zone: density.get(zone, 0) for zone in overcrowd_zones}
                if overcrowd_zones else density
            )
            trend_engine.update(count, trend_input)
            trend_data = trend_engine.get_summary()


            if SIM_MODE and count >= SIM_TH:
                overcrowd = True
                overcrowd_zones = list(density.keys())
                trend_data["trend_label"] = "SURGING"
                trend_data["predicted_crash"] = True

            zones_info, safe_zones = {}, []
            for zone, value in risk_map.items():
                zone_count, risk = value
                zones_info[str(zone)] = {"count": zone_count, "risk": risk,
                                         "status": "danger" if zone in overcrowd_zones else "safe"}
                if zone not in overcrowd_zones:
                    safe_zones.append(zone)
            best_zone = min(safe_zones, key=lambda z: risk_map[z][0]) if safe_zones else None

            with overlay_locks[key]:
                overlays[key] = {"objects": objects, "risk_map": risk_map, "count": count,
                                 "overcrowd": overcrowd, "overcrowd_zones": overcrowd_zones,
                                 "trend": dict(trend_data)}

            with state_lock:
                state[key].update({
                    "total": int(count), "overcrowd": bool(overcrowd), "zones": zones_info,
                    "overcrowd_zones": [str(z) for z in overcrowd_zones], "trend": dict(trend_data),
                    "best_zone": str(best_zone) if best_zone is not None else None,
                    "safe_zones": [str(z) for z in safe_zones],
                })
                cams = (state["cam1"], state["cam2"])
                if any(c["overcrowd"] or c["trend"].get("predicted_crash") for c in cams):
                    state["global_status"] = "DANGER"
                elif any(c["trend"].get("trend_label") == "GROWING" for c in cams):
                    state["global_status"] = "WARNING"
                else:
                    state["global_status"] = "SAFE"
                state["message_for_crowd"] = crowd_msg(state[key])
                payload = {k: v.copy() if isinstance(v, dict) else v for k, v in state.items()}

            try:
                socketio.emit("crowd_update", payload)
            except Exception:
                pass

            if now - last_snapshot_time >= SNAPSHOT_INTERVAL:
                last_snapshot_time = now
                threading.Thread(target=db_save_snapshot,
                                 args=(key, count, trend_data.get("trend_label", "STABLE"),
                                       trend_data.get("growth_rate", 0), overcrowd,
                                       [str(z) for z in overcrowd_zones]), daemon=True).start()

            danger = bool(overcrowd or trend_data.get("predicted_crash"))
            if danger and not alert_active[key]:
                alert_type = "SURGE" if trend_data.get("predicted_crash") else "OVERCROWD"
                message = (f"{alert_type} alert | camera={key} | people={count} | "
                           f"zones={overcrowd_zones} | {location_url}")
                threading.Thread(target=send_alert, args=(message,), daemon=True).start()
                threading.Thread(target=play_alert, daemon=True).start()
                alert_active[key] = True
                threading.Thread(target=db_save_alert,
                                 args=(alert_type, key, count, [str(z) for z in overcrowd_zones],
                                       trend_data.get("growth_rate", 0), location_url), daemon=True).start()
            elif not danger and alert_active[key]:
                alert_active[key] = False
                threading.Thread(target=db_resolve_alerts, args=(key,), daemon=True).start()

        except Exception as exc:
            print(f"[{key}] AI processing error: {exc}")
            time.sleep(0.02)


def mjpeg(key):
    fallback = blank_jpg(f"{key.upper()} Starting...")
    while True:
        with frame_locks[key]:
            data = frames[key] or fallback
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        time.sleep(0.033)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/crowd")
def crowd():
    return render_template("crowd.html")


@app.route("/stream/cam1")
def stream1():
    return Response(mjpeg("cam1"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stream/cam2")
def stream2():
    return Response(mjpeg("cam2"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    with state_lock:
        return jsonify(state)


@app.route("/api/crowd-status")
def api_crowd():
    with state_lock:
        return jsonify({
            "status": state["global_status"], "message": state["message_for_crowd"],
            "cam1": {"total": state["cam1"]["total"], "trend": state["cam1"]["trend"]["trend_label"], "best_zone": state["cam1"]["best_zone"]},
            "cam2": {"total": state["cam2"]["total"], "trend": state["cam2"]["trend"]["trend_label"], "best_zone": state["cam2"]["best_zone"]},
        })


def _mongo_docs(collection, query, limit):
    docs = list(collection.find(query, {"_id": 0}).sort("timestamp", DESCENDING).limit(limit))
    for doc in docs:
        if hasattr(doc.get("timestamp"), "isoformat"):
            doc["timestamp"] = doc["timestamp"].isoformat()
        if hasattr(doc.get("resolved_at"), "isoformat"):
            doc["resolved_at"] = doc["resolved_at"].isoformat()
    return docs


@app.route("/api/alerts")
def api_alerts():
    if not MONGO_OK:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        return jsonify(_mongo_docs(alerts_col, {}, 50))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/alerts/unresolved")
def api_alerts_unresolved():
    if not MONGO_OK:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        return jsonify(_mongo_docs(alerts_col, {"resolved": False}, 100))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/snapshots")
def api_snapshots():
    if not MONGO_OK:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        return jsonify(_mongo_docs(snapshots_col, {}, 100))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/db-status")
def api_db_status():
    return jsonify({"mongodb_connected": MONGO_OK, "db": DB_NAME if MONGO_OK else None})


@socketio.on("connect")
def on_connect():
    with state_lock:
        emit("crowd_update", state)


if __name__ == "__main__":
    frames["cam1"] = blank_jpg("CAM1 Starting...")
    frames["cam2"] = blank_jpg("CAM2 Starting...")
    for key, source in CAMERA_SOURCES.items():
        threading.Thread(target=capture_thread, args=(source, key), daemon=True, name=f"capture-{key}").start()
        threading.Thread(target=processing_thread, args=(key,), daemon=True, name=f"process-{key}").start()

    print("\n" + "=" * 55)
    print("CrowdX is running")
    print("Dashboard:   http://localhost:5000")
    print("Crowd page:  http://localhost:5000/crowd")
    print("Cam1 stream: http://localhost:5000/stream/cam1")
    print("Cam2 stream: http://localhost:5000/stream/cam2")
    print("MongoDB:     " + ("connected" if MONGO_OK else "not connected (optional)"))
    print("=" * 55 + "\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)

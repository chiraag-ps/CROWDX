"""Persistent per-camera person tracking for CrowdX.

Uses lightweight IoU/center-distance association, so it does not depend on
DeepSORT's optional PyTorch embedder or pkg_resources. Input detections must be
[x, y, width, height, confidence] (confidence is optional). Output objects are
(track_id, x, y, width, height), matching CrowdX's visualizer/density modules.
"""
from dataclasses import dataclass
import threading


@dataclass
class _Track:
    box: tuple
    missed: int = 0


_lock = threading.Lock()
_tracks_by_camera = {}
_next_id_by_camera = {}

# Tune these only if tracking frequently changes IDs.
_MAX_MISSED_FRAMES = 12
_MIN_IOU = 0.05
_MAX_CENTER_DISTANCE = 0.75  # fraction of the larger box diagonal


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center_distance(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    dx, dy = acx - bcx, acy - bcy
    distance = (dx * dx + dy * dy) ** 0.5
    diagonal = max((aw * aw + ah * ah) ** 0.5,
                   (bw * bw + bh * bh) ** 0.5, 1.0)
    return distance / diagonal


def track_people(frame, detections, camera_id="default"):
    """Associate current detections with persistent IDs independently per camera."""
    boxes = []
    for det in detections or []:
        if len(det) < 4:
            continue
        x, y, w, h = (int(round(float(v))) for v in det[:4])
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))

    with _lock:
        tracks = _tracks_by_camera.setdefault(camera_id, {})
        next_id = _next_id_by_camera.setdefault(camera_id, 1)

        # Candidate pairs sorted by best overlap, then nearest normalized center.
        candidates = []
        for track_id, track in tracks.items():
            for det_index, box in enumerate(boxes):
                overlap = _iou(track.box, box)
                distance = _center_distance(track.box, box)
                if overlap >= _MIN_IOU or distance <= _MAX_CENTER_DISTANCE:
                    # Overlap dominates; distance breaks ties and helps mild motion.
                    score = overlap - 0.15 * distance
                    candidates.append((score, track_id, det_index, box))
        candidates.sort(reverse=True, key=lambda item: item[0])

        used_tracks = set()
        used_detections = set()
        assignments = {}
        for score, track_id, det_index, box in candidates:
            if track_id in used_tracks or det_index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(det_index)
            assignments[det_index] = track_id

        # Age unmatched tracks so brief missed detections do not immediately change IDs.
        for track_id in list(tracks):
            if track_id not in used_tracks:
                tracks[track_id].missed += 1
                if tracks[track_id].missed > _MAX_MISSED_FRAMES:
                    del tracks[track_id]

        objects = []
        for det_index, box in enumerate(boxes):
            track_id = assignments.get(det_index)
            if track_id is None:
                track_id = next_id
                next_id += 1
            tracks[track_id] = _Track(box=box, missed=0)
            x, y, w, h = box
            objects.append((int(track_id), x, y, w, h))

        _next_id_by_camera[camera_id] = next_id
        return objects


def reset_tracker(camera_id=None):
    """Reset one camera's tracks, or all cameras (useful for tests)."""
    with _lock:
        if camera_id is None:
            _tracks_by_camera.clear()
            _next_id_by_camera.clear()
        else:
            _tracks_by_camera.pop(camera_id, None)
            _next_id_by_camera.pop(camera_id, None)

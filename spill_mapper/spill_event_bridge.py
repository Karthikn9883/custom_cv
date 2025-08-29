#!/usr/bin/env python3
# spill_event_bridge.py
# Fuse (segmentation-based) spill detection with pixel→world mapping and emit JSON per detection.
# - Tries GStreamer pipeline first (HW decode on Jetson); falls back to FFmpeg with raw RTSP URL.
# - Uses detection_adapter.Detector (DeepLabV3 .pth or other) and room_transform.yaml (homography).
# - Overlays optional visualization.

import argparse, json, uuid, time, yaml, re, inspect
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2

# --------------------------------------------------------------------
# GStreamer/FFmpeg capture helpers
# --------------------------------------------------------------------

def has_gstreamer() -> bool:
    """Return True if current OpenCV has GStreamer support."""
    try:
        return re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()) is not None
    except Exception:
        return False

def looks_like_pipeline(src: str) -> bool:
    s = src.strip()
    return ("!" in s) or s.startswith("rtspsrc") or s.startswith("uridecodebin")

def extract_rtsp_url(pipeline_or_url: str) -> str:
    """If given a pipeline, extract the location= RTSP URL; otherwise return input."""
    if "location=" in pipeline_or_url:
        s = pipeline_or_url.split("location=", 1)[1]
        url = s.split()[0].strip().strip('"').strip("'")
        return url
    return pipeline_or_url

def open_capture(src: str) -> cv2.VideoCapture:
    """
    Try to open with GStreamer if it looks like a pipeline and GStreamer is available.
    On failure, fall back to FFmpeg using the raw RTSP URL.
    """
    cap = None
    if looks_like_pipeline(src) and has_gstreamer():
        cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("[cap] Using GStreamer pipeline")
            return cap
        print("[warn] GStreamer open failed; falling back to FFmpeg with RTSP URL")
        src = extract_rtsp_url(src)

    # FFmpeg fallback (CPU decode)
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if cap.isOpened():
        print("[cap] Using FFmpeg (CPU decode)")
        return cap

    return cap  # unopened

# --------------------------------------------------------------------
# Homography utilities
# --------------------------------------------------------------------

def load_transform(path_yaml: str):
    """Load a 3x3 pixel→world homography from room_transform.yaml."""
    with open(path_yaml, "r") as f:
        d = yaml.safe_load(f)

    H = None
    if "transformation_matrix" in d:
        H = np.array(d["transformation_matrix"], dtype=np.float64)
    elif "H" in d:
        H = np.array(d["H"], dtype=np.float64)

    if H is None or H.shape != (3,3):
        raise SystemExit(f"Bad transform file (need 3x3 homography): {path_yaml}")

    units = d.get("units", "meters")
    return H, units, d

def pixel_to_world(u: float, v: float, H: np.ndarray):
    """Apply homography to (u,v) pixel to get (X,Y) world."""
    p = np.array([float(u), float(v), 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-12:
        raise ZeroDivisionError("Homography produced w≈0")
    q /= q[2]
    return float(q[0]), float(q[1])

# --------------------------------------------------------------------
# JSON event builder
# --------------------------------------------------------------------

def make_event(camera_id, src, frame_idx, frame_shape, det, world_xy, units, transform_file):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "event_type": "spill_detected",
        "event_id": str(uuid.uuid4()),
        "timestamp_utc": now,
        "camera": {
            "id": camera_id,
            "rtsp": src
        },
        "image": {
            "frame_id": int(frame_idx),
            "width": int(frame_shape[1]),
            "height": int(frame_shape[0]),
        },
        "pixel": {
            "bbox": [round(x, 2) for x in det["bbox"]],
            "center": [round(det["center"][0], 2), round(det["center"][1], 2)],
            "confidence": round(float(det.get("conf", 0.0)), 3),
            "label": det.get("label", "spill")
        },
        "world": {
            "x_m": round(world_xy[0], 4),
            "y_m": round(world_xy[1], 4),
            "units": units
        },
        "transform": {
            "type": "homography",
            "file": str(transform_file)
        }
    }

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Spill detection → world coords → JSON events (Jetson-friendly).")
    ap.add_argument("--camera-id", default="cctv-1")
    ap.add_argument("--rtsp", required=True, help="GStreamer pipeline OR raw RTSP URL")
    ap.add_argument("--transform", required=True, help="room_transform.yaml (pixel→world homography)")
    ap.add_argument("--weights", required=True, help="DeepLabV3 .pth or other supported weights")
    ap.add_argument("--spill-class", type=int, default=1, help="Segmentation class index for 'spill'")
    ap.add_argument("--json-out", default=None, help="Append JSONL to this path")
    ap.add_argument("--draw", action="store_true", help="Show overlay window")
    ap.add_argument("--max-fps", type=float, default=10.0, help="Processing rate cap")
    args = ap.parse_args()

    # Load transform
    H, units, _meta = load_transform(args.transform)

    # Import detector and be tolerant to signature differences (infer_size optional)
    from detection_adapter import Detector
    init_params = inspect.signature(Detector.__init__).parameters
    if "infer_size" in init_params:
        det = Detector(weights=args.weights, spill_class=args.spill_class, infer_size=(640, 640))
    else:
        print("[warn] detection_adapter.Detector has no 'infer_size'; using adapter default.")
        det = Detector(weights=args.weights, spill_class=args.spill_class)

    # Open capture with robust fallback
    cap = open_capture(args.rtsp)
    if not cap or not cap.isOpened():
        raise SystemExit("Failed to open video source (GStreamer and FFmpeg both failed).")

    # JSON sink
    jf = open(args.json_out, "a", encoding="utf-8") if args.json_out else None

    last_t = 0.0
    frame_idx = 0

    if args.draw:
        cv2.namedWindow("spill_event_bridge", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("spill_event_bridge", 1280, 720)

    print("[i] Running. JSON lines on stdout. Press 'q' to quit (if window open).")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # FPS throttle
        if args.max_fps > 0:
            t = time.time()
            if (t - last_t) < (1.0 / args.max_fps):
                if args.draw:
                    cv2.imshow("spill_event_bridge", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                continue
            last_t = t

        # Inference
        detections = det.infer(frame)  # expected centers/bboxes in ORIGINAL frame space

        # Emit events + overlay
        for d in detections:
            # Ensure center exists
            if "center" not in d or d["center"] is None:
                x1, y1, x2, y2 = d["bbox"]
                u = (x1 + x2) * 0.5
                v = (y1 + y2) * 0.5
                d["center"] = [u, v]
            else:
                u, v = d["center"]
            X, Y = pixel_to_world(d["center"][0], d["center"][1], H)

            evt = make_event(
                camera_id=args.camera_id,
                src=args.rtsp,
                frame_idx=frame_idx,
                frame_shape=frame.shape,
                det=d,
                world_xy=(X, Y),
                units=units,
                transform_file=args.transform
            )
            line = json.dumps(evt, separators=(",", ":"))
            print(line)
            if jf:
                jf.write(line + "\n")
                jf.flush()

            if args.draw:
                x1, y1, x2, y2 = map(int, d["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{d.get('label','spill')} {d.get('conf',0.0):.2f}",
                            (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"({X:.2f},{Y:.2f}) {units}", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        if args.draw:
            cv2.imshow("spill_event_bridge", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    if jf:
        jf.close()
    cap.release()
    if args.draw:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

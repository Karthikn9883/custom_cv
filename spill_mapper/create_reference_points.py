#!/usr/bin/env python3
import argparse, yaml, cv2, os, time
from pathlib import Path

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Collect pixel↔world reference pairs from RTSP/webcam.")
    ap.add_argument("--rtsp", help="RTSP URL")
    ap.add_argument("--camera", type=int, help="Webcam index")
    ap.add_argument("--room", required=True)
    ap.add_argument("--camera-id", required=True)
    args = ap.parse_args()

    # open source
    if args.rtsp:
        cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(args.camera if args.camera is not None else 0)
    if not cap.isOpened(): raise SystemExit("Failed to open video source")

    win = "create_reference_points (p=pause, click to add, s=save, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 1280, 720)
    paused, frame = False, None
    px, wx = [], []   # pixel_points, world_points

    def on_mouse(event, x, y, flags, param):
        nonlocal px, wx, paused
        if paused and event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked pixel: ({x},{y})")
            try:
                X = float(input("Enter world X (meters): ").strip())
                Y = float(input("Enter world Y (meters): ").strip())
                px.append([float(x), float(y)])
                wx.append([X, Y])
                print(f"Added pair: pixel=({x},{y}) -> world=({X},{Y})")
            except Exception:
                print("Invalid input; try again.")

    cv2.setMouseCallback(win, on_mouse)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok: break
        vis = frame.copy()
        for (u,v),(X,Y) in zip(px, wx):
            cv2.drawMarker(vis, (int(u),int(v)), (0,255,0), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(vis, f"({X:.2f},{Y:.2f})m", (int(u)+8,int(v)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,0), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'): paused = not paused
        elif k == ord('q'): break
        elif k == ord('s'):
            if len(px) < 4:
                print("Need at least 4 points."); continue
            out_dir = Path(__file__).resolve().parent / "rooms" / args.room / args.camera_id
            ensure_dir(out_dir)
            out_path = out_dir / "reference_points.yaml"
            with open(out_path, "w") as f:
                yaml.safe_dump({
                    "room": args.room, "camera_id": args.camera_id,
                    "units": "meters",
                    "pixel_points": px,
                    "world_points": wx
                }, f)
            print(f"Saved {len(px)} pairs → {out_path}")

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

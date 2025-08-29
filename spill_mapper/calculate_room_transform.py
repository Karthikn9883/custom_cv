#!/usr/bin/env python3
import argparse, yaml, numpy as np
from pathlib import Path
import cv2, math

def load_pairs(p: Path):
    with open(p, "r") as f: d = yaml.safe_load(f)
    P = np.array(d["pixel_points"], dtype=np.float64)
    W = np.array(d["world_points"], dtype=np.float64)
    return d, P, W

def reproj_err(H, P, W):
    ones = np.ones((P.shape[0],1))
    ph = np.hstack([P, ones])
    q = (H @ ph.T).T
    q = q[:,:2] / q[:,2:3]
    dif = q - W
    err = np.linalg.norm(dif, axis=1)
    return err, q

def main():
    ap = argparse.ArgumentParser(description="Compute homography from saved reference pairs.")
    ap.add_argument("--room", required=True)
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--pairs", default=None, help="Path to reference_points.yaml")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent / "rooms" / args.room / args.camera_id
    pairs_path = Path(args.pairs) if args.pairs else (base / "reference_points.yaml")
    meta, P, W = load_pairs(pairs_path)
    if P.shape[0] < 4: raise SystemExit("Need at least 4 pairs")

    H, mask = cv2.findHomography(P, W, cv2.RANSAC, ransacReprojThreshold=0.02)  # ~2cm tol in meters
    if H is None: raise SystemExit("findHomography failed")

    err, Wp = reproj_err(H, P, W)
    mean_e, rms_e, max_e = float(err.mean()), math.sqrt(float((err**2).mean())), float(err.max())
    print(f"Mean error: {mean_e:.4f} m | RMS: {rms_e:.4f} m | Max: {max_e:.4f} m")

    out = {
        "coordinate_system": "room_map",
        "transformation_type": "homography",
        "transformation_matrix": H.tolist(),
        "units": meta.get("units","meters"),
        "num_reference_points": int(P.shape[0]),
        "mean_error_meters": mean_e,
        "rms_error_meters": rms_e,
        "max_error_meters": max_e,
        "room": args.room,
        "camera_id": args.camera_id,
        "pixel_points": P.tolist(),
        "world_points": W.tolist()
    }
    (base).mkdir(parents=True, exist_ok=True)
    out_path = base / "room_transform.yaml"
    with open(out_path, "w") as f: yaml.safe_dump(out, f, sort_keys=False)
    print(f"Saved transform → {out_path}")

if __name__ == "__main__":
    main()

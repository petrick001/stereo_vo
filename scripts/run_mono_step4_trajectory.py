import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
import yaml

def read_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r or r[0].startswith("#"):
                continue
            rows.append((float(r[0]), r[1].strip()))
    return rows

def load_K_from_kalibr_camchain(camchain_yaml: Path):
    with camchain_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    fx, fy, cx, cy = [float(x) for x in data["cam0"]["intrinsics"]]
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K

def R_to_quat_wxyz(R):
    # Returns quaternion (qx,qy,qz,qw) for TUM format.
    # Stable conversion from rotation matrix.
    qw = np.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) / 2.0
    qx = np.sqrt(max(0.0, 1.0 + R[0,0] - R[1,1] - R[2,2])) / 2.0
    qy = np.sqrt(max(0.0, 1.0 - R[0,0] + R[1,1] - R[2,2])) / 2.0
    qz = np.sqrt(max(0.0, 1.0 - R[0,0] - R[1,1] + R[2,2])) / 2.0
    qx = np.copysign(qx, R[2,1] - R[1,2])
    qy = np.copysign(qy, R[0,2] - R[2,0])
    qz = np.copysign(qz, R[1,0] - R[0,1])
    return qx, qy, qz, qw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="Path to mav0 folder")
    ap.add_argument("--calib", required=True, help="Path to dso/camchain.yaml")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num", type=int, default=300, help="How many frames to process")
    ap.add_argument("--out", default="mono_room2.tum")
    ap.add_argument("--step", type=int, default=1, help="Use every Nth frame")
    args = ap.parse_args()

    seq = Path(args.seq)
    cam0_csv = seq / "cam0" / "data.csv"
    cam0_data = seq / "cam0" / "data"
    K = load_K_from_kalibr_camchain(Path(args.calib))

    frames = read_csv(cam0_csv)
    start = args.start
    end = min(len(frames), start + args.num * args.step)

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    ratio = 0.75

    # World-to-camera pose (T_wc). Start at identity.
    R_wc = np.eye(3, dtype=np.float64)
    t_wc = np.zeros((3, 1), dtype=np.float64)

    traj = []  # list of (timestamp, R_wc, t_wc)

    # Load first image
    t0, rel0 = frames[start]
    img_prev = cv2.imread(str(cam0_data / rel0), cv2.IMREAD_GRAYSCALE)
    kp_prev, des_prev = orb.detectAndCompute(img_prev, None)

    traj.append((t0, R_wc.copy(), t_wc.copy()))

    for idx in range(start, end - args.step, args.step):
        tA, relA = frames[idx]
        tB, relB = frames[idx + args.step]

        img_curr = cv2.imread(str(cam0_data / relB), cv2.IMREAD_GRAYSCALE)
        kp_curr, des_curr = orb.detectAndCompute(img_curr, None)

        if des_prev is None or des_curr is None or len(kp_prev) < 20 or len(kp_curr) < 20:
            # keep pose, skip
            traj.append((tB, R_wc.copy(), t_wc.copy()))
            img_prev, kp_prev, des_prev = img_curr, kp_curr, des_curr
            continue

        knn = bf.knnMatch(des_prev, des_curr, k=2)
        good = []
        for m, n in knn:
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) < 30:
            traj.append((tB, R_wc.copy(), t_wc.copy()))
            img_prev, kp_prev, des_prev = img_curr, kp_curr, des_curr
            continue

        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good])
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(pts_prev, pts_curr, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or mask is None:
            traj.append((tB, R_wc.copy(), t_wc.copy()))
            img_prev, kp_prev, des_prev = img_curr, kp_curr, des_curr
            continue

        inlier_prev = pts_prev[mask.ravel() == 1]
        inlier_curr = pts_curr[mask.ravel() == 1]
        if len(inlier_prev) < 20:
            traj.append((tB, R_wc.copy(), t_wc.copy()))
            img_prev, kp_prev, des_prev = img_curr, kp_curr, des_curr
            continue

        _, R_rel, t_rel, _ = cv2.recoverPose(E, inlier_prev, inlier_curr, K)

        # Monocular scale is unknown: set ||t_rel|| = 1 (up-to-scale trajectory) :contentReference[oaicite:1]{index=1}
        t_rel = t_rel / (np.linalg.norm(t_rel) + 1e-12)

        # Compose: T_wc_new = T_rel * T_wc  (because R_rel,t_rel maps prev->curr in camera coords)
        # We maintain world->camera; update with relative motion:
        R_wc = R_rel @ R_wc
        t_wc = R_rel @ t_wc + t_rel

        traj.append((tB, R_wc.copy(), t_wc.copy()))

        # advance
        img_prev, kp_prev, des_prev = img_curr, kp_curr, des_curr

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export in TUM format: timestamp tx ty tz qx qy qz qw :contentReference[oaicite:2]{index=2}
    with out_path.open("w", newline="") as f:
        for ts, R, t in traj:
            qx, qy, qz, qw = R_to_quat_wxyz(R)
            f.write(f"{ts:.6f} {t[0,0]:.6f} {t[1,0]:.6f} {t[2,0]:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    print(f"Wrote {len(traj)} poses to {out_path}")

if __name__ == "__main__":
    main()

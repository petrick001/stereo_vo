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


def load_camchain(camchain_yaml: Path):
    with camchain_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    fx, fy, cx, cy = [float(x) for x in data["cam0"]["intrinsics"]]
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # cam1 pose wrt cam0
    T10 = np.array(data["cam1"]["T_cn_cnm1"], dtype=np.float64)
    B = float(abs(T10[0, 3]))
    return K, B


def R_to_quat_xyzw(R):
    qw = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    qy = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    qz = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0

    qx = np.copysign(qx, R[2, 1] - R[1, 2])
    qy = np.copysign(qy, R[0, 2] - R[2, 0])
    qz = np.copysign(qz, R[1, 0] - R[0, 1])
    return qx, qy, qz, qw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="Path to mav0 folder")
    ap.add_argument("--calib", required=True, help="Path to dso/camchain.yaml")
    ap.add_argument("--start", type=int, default=2600)
    ap.add_argument("--num", type=int, default=80)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", default="results/stereo_room2_debug.tum")
    args = ap.parse_args()

    seq = Path(args.seq)
    cam0_csv = seq / "cam0" / "data.csv"
    cam1_csv = seq / "cam1" / "data.csv"
    cam0_data = seq / "cam0" / "data"
    cam1_data = seq / "cam1" / "data"

    L = read_csv(cam0_csv)
    R = read_csv(cam1_csv)

    K, B = load_camchain(Path(args.calib))
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    print(f"fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}, baseline={B:.6f}", flush=True)

    # StereoSGBM disparity
    num_disp = 16 * 10
    block_size = 7
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # Track with optical flow for stability
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    start = args.start
    end = min(len(L), start + args.num * args.step)

    # Camera pose in world directly
    R_cw = np.eye(3, dtype=np.float64)
    t_cw = np.zeros((3, 1), dtype=np.float64)

    traj = []

    ts0, _ = L[start]
    traj.append((ts0, R_cw.copy(), t_cw.copy()))

    for idx in range(start, end - args.step, args.step):
        _, relLA = L[idx]
        tsB, relLB = L[idx + args.step]
        _, relRA = R[idx]

        imgL0 = cv2.imread(str(cam0_data / relLA), cv2.IMREAD_GRAYSCALE)
        imgR0 = cv2.imread(str(cam1_data / relRA), cv2.IMREAD_GRAYSCALE)
        imgL1 = cv2.imread(str(cam0_data / relLB), cv2.IMREAD_GRAYSCALE)

        if imgL0 is None or imgR0 is None or imgL1 is None:
            print(f"idx={idx} skipped: image load failed", flush=True)
            traj.append((tsB, R_cw.copy(), t_cw.copy()))
            continue

        disp0 = sgbm.compute(imgL0, imgR0).astype(np.float32) / 16.0

        # detect good corners on left_t
        p0 = cv2.goodFeaturesToTrack(
            imgL0,
            maxCorners=3000,
            qualityLevel=0.001,
            minDistance=5
        )

        if p0 is None or len(p0) < 60:
            print(f"idx={idx} skipped: too few corners", flush=True)
            traj.append((tsB, R_cw.copy(), t_cw.copy()))
            continue

        p1, st, err = cv2.calcOpticalFlowPyrLK(imgL0, imgL1, p0, None, **lk_params)

        if p1 is None or st is None:
            print(f"idx={idx} skipped: optical flow failed", flush=True)
            traj.append((tsB, R_cw.copy(), t_cw.copy()))
            continue

        st = st.reshape(-1)
        p0 = p0.reshape(-1, 2)[st == 1]
        p1 = p1.reshape(-1, 2)[st == 1]

        if len(p0) < 60:
            print(f"idx={idx} skipped: too few tracked points ({len(p0)})", flush=True)
            traj.append((tsB, R_cw.copy(), t_cw.copy()))
            continue

        obj_pts = []
        img_pts = []

        h, w = disp0.shape

        for (u0, v0), (u1, v1) in zip(p0, p1):
            uu = int(round(u0))
            vv = int(round(v0))

            if uu < 0 or vv < 0 or uu >= w or vv >= h:
                continue

            d = disp0[vv, uu]
            if d <= 0.5:
                continue

            Z = (fx * B) / d

            # reject bad depth, but be less strict
            if Z <= 0.0 or Z > 15.0:
                continue

            X = (u0 - cx) * Z / fx
            Y = (v0 - cy) * Z / fy

            obj_pts.append([X, Y, Z])
            img_pts.append([u1, v1])

        obj_pts = np.asarray(obj_pts, dtype=np.float64)
        img_pts = np.asarray(img_pts, dtype=np.float64)

        if len(obj_pts) < 40:
            print(f"idx={idx} skipped: too few 3D-2D points ({len(obj_pts)})", flush=True)
            traj.append((tsB, R_cw.copy(), t_cw.copy()))
            continue

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=K,
            distCoeffs=None,
            iterationsCount=2000,
            reprojectionError=2.0,
            confidence=0.999
        )

        if not ok or inliers is None or len(inliers) < 25:
            inlier_count = 0 if inliers is None else len(inliers)
            print(f"idx={idx} skipped: weak PnP (inliers={inlier_count})", flush=True)
            traj.append((tsB, R_cw.copy(), t_cw.copy()))
            continue

        R_10, _ = cv2.Rodrigues(rvec)   # pose of cam1 wrt cam0
        t_10 = tvec

        # invert relative transform to get motion from cam0 -> cam1 in world accumulation form
        R_01 = R_10.T
        t_01 = -R_10.T @ t_10

        step_norm = float(np.linalg.norm(t_01))
        print(f"idx={idx} 3D2D={len(obj_pts)} inliers={len(inliers)} step={step_norm:.4f} m", flush=True)

        # reject obviously bad steps
        if step_norm > 0.20:
            print(f"idx={idx} skipped: step too large ({step_norm:.4f} m)", flush=True)
            traj.append((tsB, R_cw.copy(), t_cw.copy()))
            continue

        # compose camera pose in world
        t_cw = t_cw + R_cw @ t_01
        R_cw = R_cw @ R_01

        traj.append((tsB, R_cw.copy(), t_cw.copy()))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # TUM format: timestamp tx ty tz qx qy qz qw
    with out_path.open("w", newline="") as f:
        for ts, Rmat, tvec in traj:
            qx, qy, qz, qw = R_to_quat_xyzw(Rmat)
            f.write(
                f"{ts:.6f} "
                f"{tvec[0,0]:.6f} {tvec[1,0]:.6f} {tvec[2,0]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )

    print(f"Wrote {len(traj)} poses to {out_path}", flush=True)


if __name__ == "__main__":
    main()

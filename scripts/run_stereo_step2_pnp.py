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


def load_stereo_from_camchain(camchain_yaml: Path):
    with camchain_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    fx, fy, cx, cy = [float(x) for x in data["cam0"]["intrinsics"]]
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    # baseline from cam1 wrt cam0
    T10 = np.array(data["cam1"]["T_cn_cnm1"], dtype=np.float64)
    B = float(abs(T10[0, 3]))
    return K, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--i", type=int, default=2600)
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()

    seq = Path(args.seq)
    cam0_csv = seq / "cam0" / "data.csv"
    cam1_csv = seq / "cam1" / "data.csv"
    cam0_data = seq / "cam0" / "data"
    cam1_data = seq / "cam1" / "data"

    L = read_csv(cam0_csv)
    R = read_csv(cam1_csv)
    i = args.i
    j = i + args.step
    if j >= min(len(L), len(R)):
        raise ValueError("Index out of range")

    _, relL0 = L[i]
    _, relL1 = L[j]
    _, relR0 = R[i]

    imgL0 = cv2.imread(str(cam0_data / relL0), cv2.IMREAD_GRAYSCALE)
    imgL1 = cv2.imread(str(cam0_data / relL1), cv2.IMREAD_GRAYSCALE)
    imgR0 = cv2.imread(str(cam1_data / relR0), cv2.IMREAD_GRAYSCALE)
    if imgL0 is None or imgL1 is None or imgR0 is None:
        raise FileNotFoundError("Could not read images")

    K, B = load_stereo_from_camchain(Path(args.calib))
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # 1) disparity at time i (left0 vs right0)
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
    disp0 = sgbm.compute(imgL0, imgR0).astype(np.float32) / 16.0

    # 2) track features from L0 -> L1
    orb = cv2.ORB_create(nfeatures=2000)
    kp0, des0 = orb.detectAndCompute(imgL0, None)
    kp1, des1 = orb.detectAndCompute(imgL1, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des0, des1, k=2)

    good = []
    ratio = 0.75
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])

    # 3) build 3D points from pts0 using disparity -> depth (metric)
    obj_pts = []
    img_pts = []

    for (u0, v0), (u1, v1) in zip(pts0, pts1):
        d = disp0[int(round(v0)), int(round(u0))]
        if d <= 0.5:
            continue
        Z = (fx * B) / d
        X = (u0 - cx) * Z / fx
        Y = (v0 - cy) * Z / fx  # fx ~ fy; ok for this dataset baseline step
        obj_pts.append([X, Y, Z])
        img_pts.append([u1, v1])

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)

    print("Good matches:", len(good))
    print("3D-2D correspondences:", len(obj_pts))

    if len(obj_pts) < 30:
        raise RuntimeError("Too few 3D-2D points for PnP. Try a different frame index.")

    # 4) PnP RANSAC to estimate motion (metric) :contentReference[oaicite:2]{index=2}
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=K,
        distCoeffs=None,
        iterationsCount=2000,
        reprojectionError=2.0,
        confidence=0.999
    )
    if not ok:
        raise RuntimeError("solvePnPRansac failed")

    inl = 0 if inliers is None else int(len(inliers))
    print("PnP inliers:", inl)
    print("t (meters) =", tvec.ravel())

    R, _ = cv2.Rodrigues(rvec)
    print("R=\n", R)


if __name__ == "__main__":
    main()

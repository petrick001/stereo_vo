import argparse
import csv
from pathlib import Path
import cv2
import numpy as np

def read_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r or r[0].startswith("#"):
                continue
            rows.append((float(r[0]), r[1].strip()))
    return rows

def load_K_from_euroc_cam_yaml(yaml_path: Path):
    """
    EuRoC-style camera yaml often has:
      intrinsics: [fx, fy, cx, cy]
    We'll parse that using OpenCV FileStorage (works for YAML).
    """
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open {yaml_path}")

    intr = fs.getNode("intrinsics")
    if intr.empty():
        raise KeyError(f"'intrinsics' not found in {yaml_path}")

    fx, fy, cx, cy = [float(x) for x in intr.mat().flatten()] if intr.isSeq() is False else \
                     [float(intr.at(i).real()) for i in range(4)]

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="Path to mav0 folder")
    ap.add_argument("--i", type=int, default=2600, help="Start frame index")
    args = ap.parse_args()

    seq = Path(args.seq)
    cam0_csv = seq / "cam0" / "data.csv"
    cam0_data = seq / "cam0" / "data"

    # intrinsics file is usually here in exported EuRoC format:
    cam0_yaml = seq / "cam0" / "sensor.yaml"
    if not cam0_yaml.exists():
        raise FileNotFoundError(f"Missing {cam0_yaml}. (We need K for Essential matrix.)")

    K = load_K_from_euroc_cam_yaml(cam0_yaml)
    print("K=\n", K)

    left_list = read_csv(cam0_csv)
    i = args.i
    if i < 0 or i + 1 >= len(left_list):
        raise ValueError(f"--i must be in [0, {len(left_list)-2}]")

    _, rel0 = left_list[i]
    _, rel1 = left_list[i + 1]
    img0 = cv2.imread(str(cam0_data / rel0), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(cam0_data / rel1), cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create(nfeatures=2000)
    k0, d0 = orb.detectAndCompute(img0, None)
    k1, d1 = orb.detectAndCompute(img1, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d0, d1, k=2)

    good = []
    ratio = 0.75
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    pts0 = np.float32([k0[m.queryIdx].pt for m in good])
    pts1 = np.float32([k1[m.trainIdx].pt for m in good])

    # Essential matrix with RANSAC (required verification step) :contentReference[oaicite:2]{index=2}
    E, mask = cv2.findEssentialMat(
        pts0, pts1, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None or mask is None:
        raise RuntimeError("findEssentialMat failed.")

    inliers = int(mask.sum())
    print(f"Good matches: {len(good)}")
    print(f"RANSAC inliers: {inliers}")

    # Recover pose (R, t up to scale) :contentReference[oaicite:3]{index=3}
    inlier_pts0 = pts0[mask.ravel() == 1]
    inlier_pts1 = pts1[mask.ravel() == 1]

    _, R, t, _ = cv2.recoverPose(E, inlier_pts0, inlier_pts1, K)
    print("R=\n", R)
    print("t (up to scale)=\n", t.ravel())

if __name__ == "__main__":
    main()

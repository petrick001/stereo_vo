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
    """
    Kalibr camchain.yaml typically contains:
      cam0:
        intrinsics: [fx, fy, cx, cy]
    """
    with camchain_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "cam0" not in data or "intrinsics" not in data["cam0"]:
        raise KeyError("Expected cam0.intrinsics in camchain.yaml")

    fx, fy, cx, cy = [float(x) for x in data["cam0"]["intrinsics"]]
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="Path to mav0 folder")
    ap.add_argument("--calib", required=True, help="Path to Kalibr camchain.yaml")
    ap.add_argument("--i", type=int, default=2600, help="Start frame index")
    args = ap.parse_args()

    seq = Path(args.seq)
    cam0_csv = seq / "cam0" / "data.csv"
    cam0_data = seq / "cam0" / "data"

    K = load_K_from_kalibr_camchain(Path(args.calib))
    print("K=\n", K)

    left_list = read_csv(cam0_csv)
    i = args.i
    if i < 0 or i + 1 >= len(left_list):
        raise ValueError(f"--i must be in [0, {len(left_list)-2}]")

    _, rel0 = left_list[i]
    _, rel1 = left_list[i + 1]
    img0 = cv2.imread(str(cam0_data / rel0), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(cam0_data / rel1), cv2.IMREAD_GRAYSCALE)
    if img0 is None or img1 is None:
        raise FileNotFoundError("Could not load one of the images.")

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

    # Essential matrix with RANSAC (required verification step) :contentReference[oaicite:1]{index=1}
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

    inlier_pts0 = pts0[mask.ravel() == 1]
    inlier_pts1 = pts1[mask.ravel() == 1]

    _, R, t, _ = cv2.recoverPose(E, inlier_pts0, inlier_pts1, K)
    print("t (up to scale) =", t.ravel())

if __name__ == "__main__":
    main()

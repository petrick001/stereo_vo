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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="Path to mav0 folder")
    ap.add_argument("--i", type=int, default=2600, help="Start frame index")
    args = ap.parse_args()

    seq = Path(args.seq)
    cam0_csv = seq / "cam0" / "data.csv"
    cam0_data = seq / "cam0" / "data"

    left_list = read_csv(cam0_csv)
    i = args.i
    if i < 0 or i + 1 >= len(left_list):
        raise ValueError(f"--i must be in [0, {len(left_list)-2}]")

    # Load two consecutive frames
    _, rel0 = left_list[i]
    _, rel1 = left_list[i + 1]
    img0 = cv2.imread(str(cam0_data / rel0), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(cam0_data / rel1), cv2.IMREAD_GRAYSCALE)
    if img0 is None or img1 is None:
        raise FileNotFoundError("Could not load one of the images.")

    # ORB features
    orb = cv2.ORB_create(nfeatures=2000)
    k0, d0 = orb.detectAndCompute(img0, None)
    k1, d1 = orb.detectAndCompute(img1, None)

    if d0 is None or d1 is None:
        raise RuntimeError("No descriptors found in one of the frames.")

    # KNN matching + ratio test (classical VO step) :contentReference[oaicite:1]{index=1}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d0, d1, k=2)

    good = []
    ratio = 0.75
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    print(f"Frame {i} -> {i+1}")
    print(f"Keypoints: {len(k0)} -> {len(k1)}")
    print(f"Raw knn matches: {len(knn)}")
    print(f"Good matches (ratio={ratio}): {len(good)}")

    # Draw matches (limit for visibility)
    vis = cv2.drawMatches(img0, k0, img1, k1, good[:200], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("ORB matches (cam0)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

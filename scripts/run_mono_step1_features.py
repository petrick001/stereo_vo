import argparse
import csv
from pathlib import Path
import cv2

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
    ap.add_argument("--step", type=int, default=1, help="Process every Nth frame")
    args = ap.parse_args()

    seq = Path(args.seq)
    cam0_csv = seq / "cam0" / "data.csv"
    cam0_data = seq / "cam0" / "data"

    left_list = read_csv(cam0_csv)

    orb = cv2.ORB_create(nfeatures=2000)

    for i in range(0, len(left_list), args.step):
        t, rel = left_list[i]
        img_path = cam0_data / rel
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read {img_path}")

        kps, des = orb.detectAndCompute(img, None)

        vis = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.putText(vis, f"i={i}  keypoints={len(kps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)

        cv2.imshow("mono: ORB keypoints (cam0)", vis)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

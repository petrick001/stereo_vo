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
            ts = float(r[0])
            rel = r[1].strip()
            rows.append((ts, rel))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="Path to mav0 folder")
    ap.add_argument("--step", type=int, default=1, help="Show every Nth frame")
    args = ap.parse_args()

    seq = Path(args.seq)

    cam0_csv = seq / "cam0" / "data.csv"
    cam1_csv = seq / "cam1" / "data.csv"

    if not cam0_csv.exists() or not cam1_csv.exists():
        raise FileNotFoundError(f"Missing {cam0_csv} or {cam1_csv}")

    left_list = read_csv(cam0_csv)
    right_list = read_csv(cam1_csv)
    n = min(len(left_list), len(right_list))
    print("Frames:", n)

    for i in range(0, n, args.step):
        t0, rel0 = left_list[i]
        t1, rel1 = right_list[i]

        p0 = seq / "cam0" / "data" / rel0
        p1 = seq / "cam1" / "data" / rel1

        imgL = cv2.imread(str(p0), cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)

        if imgL is None or imgR is None:
            raise FileNotFoundError(f"Could not read {p0} or {p1}")

        t = 0.5 * (t0 + t1)
        print(f"i={i:06d} t={t:.0f}  L={imgL.shape} R={imgR.shape}")

        cv2.imshow("left", imgL)
        cv2.imshow("right", imgR)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

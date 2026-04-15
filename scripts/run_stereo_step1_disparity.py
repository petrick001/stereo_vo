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


def load_stereo_params_from_camchain(camchain_yaml: Path):
    """
    From Kalibr camchain.yaml:
      cam0.intrinsics = [fx, fy, cx, cy]
      cam1.T_cn_cnm1 (or similar) provides extrinsics.
    For depth, we need:
      - fx (pixels)
      - baseline B (meters) = |translation_x between cams|
    """
    with camchain_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    fx, fy, cx, cy = [float(x) for x in data["cam0"]["intrinsics"]]
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    # Kalibr camchain commonly stores cam1 pose w.r.t cam0 in:
    # data["cam1"]["T_cn_cnm1"] (4x4 as nested lists).
    # If key differs, we’ll print keys and adjust.
    if "T_cn_cnm1" in data.get("cam1", {}):
        T = np.array(data["cam1"]["T_cn_cnm1"], dtype=np.float64)
    elif "T_cam_imu" in data.get("cam1", {}):
        # Not expected here; included as fallback placeholder
        raise KeyError("cam1.T_cn_cnm1 not found; camchain format differs.")
    else:
        raise KeyError("cam1.T_cn_cnm1 not found; camchain format differs.")

    # baseline is magnitude of x-translation between stereo cameras
    baseline = float(abs(T[0, 3]))
    return K, baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="Path to mav0 folder")
    ap.add_argument("--calib", required=True, help="Path to dso/camchain.yaml")
    ap.add_argument("--i", type=int, default=2600, help="Frame index")
    args = ap.parse_args()

    seq = Path(args.seq)
    cam0_csv = seq / "cam0" / "data.csv"
    cam1_csv = seq / "cam1" / "data.csv"
    cam0_data = seq / "cam0" / "data"
    cam1_data = seq / "cam1" / "data"

    left_list = read_csv(cam0_csv)
    right_list = read_csv(cam1_csv)

    i = args.i
    if i < 0 or i >= min(len(left_list), len(right_list)):
        raise ValueError("Bad frame index.")

    _, relL = left_list[i]
    _, relR = right_list[i]

    imgL = cv2.imread(str(cam0_data / relL), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(str(cam1_data / relR), cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        raise FileNotFoundError("Could not read left/right images.")

    K, B = load_stereo_params_from_camchain(Path(args.calib))
    fx = float(K[0, 0])

    print("fx =", fx)
    print("baseline B (m) =", B)

    # StereoSGBM disparity (pixels * 16 in OpenCV, then we divide by 16)
    # Settings are conservative and should work as a baseline. :contentReference[oaicite:2]{index=2}
    num_disp = 16 * 10  # must be multiple of 16
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

    disp = sgbm.compute(imgL, imgR).astype(np.float32) / 16.0

    # Mask invalid disparities (<= 0)
    valid = disp > 0.5

    # Depth from disparity: Z = fB/d  :contentReference[oaicite:3]{index=3}
    Z = np.zeros_like(disp, dtype=np.float32)
    Z[valid] = (fx * B) / disp[valid]

    # Visualize disparity nicely
    disp_vis = disp.copy()
    disp_vis[~valid] = 0
    disp_vis = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = disp_vis.astype(np.uint8)

    # Print a simple depth sanity check at image center
    h, w = disp.shape
    u, v = w // 2, h // 2
    print(f"Center disparity d={disp[v, u]:.2f} px, depth Z={Z[v, u]:.2f} m")

    cv2.imshow("Left", imgL)
    cv2.imshow("Right", imgR)
    cv2.imshow("Disparity (SGBM)", disp_vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

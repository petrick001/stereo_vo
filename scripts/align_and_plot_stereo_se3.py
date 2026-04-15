import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_tum(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            p = line.split()
            if len(p) != 8:
                continue
            rows.append(list(map(float, p)))
    return np.array(rows)


def associate_by_nearest(gt, est, max_diff=5e7):
    gt_ts = gt[:, 0]
    est_ts = est[:, 0]

    pairs = []
    j = 0
    for i in range(len(est_ts)):
        t = est_ts[i]
        while j + 1 < len(gt_ts) and abs(gt_ts[j + 1] - t) < abs(gt_ts[j] - t):
            j += 1
        if abs(gt_ts[j] - t) <= max_diff:
            pairs.append((j, i))
    return pairs


def rigid_align_3d(A, B):
    """
    Find R, t such that:
        B_aligned = R @ B + t
    using Umeyama / Kabsch without scale.
    A, B: Nx3
    """
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)

    AA = A - mu_A
    BB = B - mu_B

    H = BB.T @ AA
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_A - R @ mu_B
    return R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--stereo", required=True)
    args = ap.parse_args()

    gt = load_tum(args.gt)
    stereo = load_tum(args.stereo)

    pairs = associate_by_nearest(gt, stereo)
    if len(pairs) == 0:
        raise RuntimeError("No timestamp matches found.")

    gt_xyz = np.array([gt[i, 1:4] for i, _ in pairs])
    st_xyz = np.array([stereo[j, 1:4] for _, j in pairs])

    R, t = rigid_align_3d(gt_xyz, st_xyz)
    st_aligned = (R @ st_xyz.T).T + t

    err = np.linalg.norm(gt_xyz - st_aligned, axis=1)
    rmse = np.sqrt(np.mean(err ** 2))

    print(f"Matched poses: {len(pairs)}")
    print(f"Stereo SE(3)-aligned RMSE: {rmse:.6f} m")

    plt.figure(figsize=(8, 6))
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="Ground Truth", linewidth=3)
    plt.plot(st_aligned[:, 0], st_aligned[:, 2], label="Stereo VO (SE3 aligned)", linewidth=2)
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Stereo VO vs Ground Truth (SE3 aligned)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
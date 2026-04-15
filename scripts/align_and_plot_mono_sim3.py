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


def sim3_align(A, B):
    """
    Find s, R, t such that:
        A ≈ s * R @ B + t
    A, B are Nx3
    """
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)

    AA = A - mu_A
    BB = B - mu_B

    H = BB.T @ AA / len(A)
    U, D, Vt = np.linalg.svd(H)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = Vt.T @ S @ U.T

    var_B = np.mean(np.sum(BB ** 2, axis=1))
    scale = np.trace(np.diag(D) @ S) / var_B

    t = mu_A - scale * (R @ mu_B)
    return scale, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--mono", required=True)
    args = ap.parse_args()

    gt = load_tum(args.gt)
    mono = load_tum(args.mono)

    pairs = associate_by_nearest(gt, mono)
    if len(pairs) == 0:
        raise RuntimeError("No timestamp matches found.")

    gt_xyz = np.array([gt[i, 1:4] for i, _ in pairs])
    mono_xyz = np.array([mono[j, 1:4] for _, j in pairs])

    s, R, t = sim3_align(gt_xyz, mono_xyz)
    mono_aligned = (s * (R @ mono_xyz.T)).T + t

    err = np.linalg.norm(gt_xyz - mono_aligned, axis=1)
    rmse = np.sqrt(np.mean(err ** 2))

    print(f"Matched poses: {len(pairs)}")
    print(f"Mono Sim(3)-aligned RMSE: {rmse:.6f} m")
    print(f"Estimated scale: {s:.6f}")

    plt.figure(figsize=(8, 6))
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="Ground Truth", linewidth=3)
    plt.plot(mono_aligned[:, 0], mono_aligned[:, 2], label=f"Mono VO (Sim3 aligned, s={s:.4f})", linewidth=2)
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Monocular VO vs Ground Truth (Sim3 aligned)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
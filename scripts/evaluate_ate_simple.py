import argparse
import numpy as np


def load_tum(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            vals = list(map(float, parts))
            data.append(vals)
    return np.array(data)


def associate_by_nearest(gt, est, max_diff=5e7):
    """
    Associate by nearest timestamp.
    TUM VI timestamps are in nanoseconds, so max_diff=5e7 means 50 ms.
    """
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


def rmse(errors):
    return np.sqrt(np.mean(errors ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--est", required=True)
    args = ap.parse_args()

    gt = load_tum(args.gt)
    est = load_tum(args.est)

    pairs = associate_by_nearest(gt, est)
    if len(pairs) == 0:
        raise RuntimeError("No timestamp matches found.")

    gt_xyz = np.array([gt[i, 1:4] for i, _ in pairs])
    est_xyz = np.array([est[j, 1:4] for _, j in pairs])

    # simple origin alignment
    gt_xyz = gt_xyz - gt_xyz[0]
    est_xyz = est_xyz - est_xyz[0]

    # translation error only
    err = np.linalg.norm(gt_xyz - est_xyz, axis=1)

    print(f"Matched poses: {len(pairs)}")
    print(f"ATE RMSE (translation only): {rmse(err):.6f} m")
    print(f"ATE mean: {np.mean(err):.6f} m")
    print(f"ATE median: {np.median(err):.6f} m")
    print(f"ATE max: {np.max(err):.6f} m")


if __name__ == "__main__":
    main()
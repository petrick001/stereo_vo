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


def sim3_align(A, B):
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
    s = np.trace(np.diag(D) @ S) / var_B
    t = mu_A - s * (R @ mu_B)
    return s, R, t


parser = argparse.ArgumentParser()
parser.add_argument("--gt", required=True)
parser.add_argument("--mono", required=True)
parser.add_argument("--stereo", required=True)
args = parser.parse_args()

gt = load_tum(args.gt)
mono = load_tum(args.mono)
stereo = load_tum(args.stereo)

pairs_m = associate_by_nearest(gt, mono)
pairs_s = associate_by_nearest(gt, stereo)

gt_m = np.array([gt[i, 1:4] for i, _ in pairs_m])
mono_xyz = np.array([mono[j, 1:4] for _, j in pairs_m])

gt_s = np.array([gt[i, 1:4] for i, _ in pairs_s])
st_xyz = np.array([stereo[j, 1:4] for _, j in pairs_s])

s_m, R_m, t_m = sim3_align(gt_m, mono_xyz)
mono_aligned = (s_m * (R_m @ mono_xyz.T)).T + t_m

R_s, t_s = rigid_align_3d(gt_s, st_xyz)
st_aligned = (R_s @ st_xyz.T).T + t_s

rmse_m = np.sqrt(np.mean(np.sum((gt_m - mono_aligned) ** 2, axis=1)))
rmse_s = np.sqrt(np.mean(np.sum((gt_s - st_aligned) ** 2, axis=1)))

plt.figure(figsize=(9, 7))
plt.plot(gt_s[:, 0], gt_s[:, 2], label="Ground Truth", linewidth=3)
plt.plot(mono_aligned[:, 0], mono_aligned[:, 2],
         label=f"Mono VO (Sim3 aligned, s={s_m:.4f}, RMSE={rmse_m:.3f} m)", linewidth=2)
plt.plot(st_aligned[:, 0], st_aligned[:, 2],
         label=f"Stereo VO (SE3 aligned, RMSE={rmse_s:.3f} m)", linewidth=2)

plt.scatter(gt_s[0, 0], gt_s[0, 2], s=80, label="GT Start")
plt.scatter(gt_s[-1, 0], gt_s[-1, 2], s=80, label="GT End")

plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title("Trajectory Comparison — Room2")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show(block=True)
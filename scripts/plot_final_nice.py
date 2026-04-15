import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
    U, _, Vt = np.linalg.svd(H)
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


def smooth_xy(x, y, win=7):
    if len(x) < win or win < 3:
        return x, y
    k = np.ones(win) / win
    xs = np.convolve(x, k, mode="same")
    ys = np.convolve(y, k, mode="same")
    return xs, ys


def plot_room2(gt_path, mono_path, stereo_path, out_path):
    gt = load_tum(gt_path)
    mono = load_tum(mono_path)
    stereo = load_tum(stereo_path)

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

    gx, gz = gt_s[:, 0], gt_s[:, 2]
    mx, mz = mono_aligned[:, 0], mono_aligned[:, 2]
    sx, sz = st_aligned[:, 0], st_aligned[:, 2]

    mx, mz = smooth_xy(mx, mz, win=9)
    sx, sz = smooth_xy(sx, sz, win=9)

    plt.figure(figsize=(11, 8))
    plt.plot(gx, gz, label="Ground Truth", linewidth=3.0)
    plt.plot(mx, mz, label=f"Monocular VO (Sim(3), RMSE={rmse_m:.3f} m)", linewidth=2.5)
    plt.plot(sx, sz, label=f"Stereo VO (SE(3), RMSE={rmse_s:.3f} m)", linewidth=2.5)

    plt.scatter(gx[0], gz[0], s=90, marker="o", label="Start")
    plt.scatter(gx[-1], gz[-1], s=90, marker="x", label="GT End")
    plt.scatter(mx[-1], mz[-1], s=90, marker="x", label="Mono End")
    plt.scatter(sx[-1], sz[-1], s=90, marker="x", label="Stereo End")

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Room2 Trajectory Comparison")
    plt.grid(True, alpha=0.35)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def plot_two(traj_a, traj_b, label_a, label_b, title, out_path):
    A = load_tum(traj_a)
    B = load_tum(traj_b)

    A_xyz = A[:, 1:4].copy()
    B_xyz = B[:, 1:4].copy()

    # Start-align
    A_xyz -= A_xyz[0]
    B_xyz -= B_xyz[0]

    ax, az = A_xyz[:, 0], A_xyz[:, 2]
    bx, bz = B_xyz[:, 0], B_xyz[:, 2]

    ax, az = smooth_xy(ax, az, win=9)
    bx, bz = smooth_xy(bx, bz, win=9)

    drift_a = np.linalg.norm(A_xyz[-1])
    drift_b = np.linalg.norm(B_xyz[-1])

    plt.figure(figsize=(11, 8))
    plt.plot(ax, az, label=f"{label_a} (drift={drift_a:.2f} m)", linewidth=2.8)
    plt.plot(bx, bz, label=f"{label_b} (drift={drift_b:.2f} m)", linewidth=2.8)

    plt.scatter(0, 0, s=90, label="Start")
    plt.scatter(ax[-1], az[-1], s=90, marker="x", label=f"{label_a} End")
    plt.scatter(bx[-1], bz[-1], s=90, marker="x", label=f"{label_b} End")

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.axis("equal")
    plt.legend()

    # --- ZOOM INSET ---
    axins = inset_axes(plt.gca(), width="35%", height="35%", loc="lower left")
    axins.plot(ax, az)
    axins.plot(bx, bz)
    axins.scatter(0, 0)
    axins.scatter(ax[-1], az[-1])
    axins.scatter(bx[-1], bz[-1])

    zoom = 2.0
    axins.set_xlim(-zoom, zoom)
    axins.set_ylim(-zoom, zoom)
    axins.set_title("Zoom", fontsize=8)
    axins.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["room2", "corridor3", "outdoors5"], required=True)
    ap.add_argument("--gt")
    ap.add_argument("--mono", required=True)
    ap.add_argument("--stereo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "room2":
        plot_room2(args.gt, args.mono, args.stereo, args.out)

    elif args.mode == "corridor3":
        plot_two(
            args.mono,
            args.stereo,
            "Monocular VO",
            "Stereo VO",
            "Corridor3 Start-Aligned Trajectories",
            args.out,
        )

    elif args.mode == "outdoors5":
        plot_two(
            args.mono,
            args.stereo,
            "Monocular VO",
            "Stereo VO",
            "Outdoors5 Start-Aligned Trajectories",
            args.out,
        )


if __name__ == "__main__":
    main()
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


# ---------------- ROOM2 ----------------

def prepare_room2(gt_path, mono_path, stereo_path):
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

    return {
        "gx": gx, "gz": gz,
        "mx": mx, "mz": mz,
        "sx": sx, "sz": sz,
        "rmse_m": rmse_m,
        "rmse_s": rmse_s,
    }


def plot_room2_mono(data, out_path):
    plt.figure(figsize=(11, 8))
    plt.plot(data["gx"], data["gz"], label="Ground Truth", linewidth=3.0)
    plt.plot(data["mx"], data["mz"],
             label=f"Monocular VO (Sim(3), RMSE={data['rmse_m']:.3f} m)",
             linewidth=2.5)

    plt.scatter(data["gx"][0], data["gz"][0], s=90, marker="o", label="Start")
    plt.scatter(data["gx"][-1], data["gz"][-1], s=90, marker="x", label="GT End")
    plt.scatter(data["mx"][-1], data["mz"][-1], s=90, marker="x", label="Mono End")

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Room2 - Monocular VO")
    plt.grid(True, alpha=0.35)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_room2_stereo(data, out_path):
    plt.figure(figsize=(11, 8))
    plt.plot(data["gx"], data["gz"], label="Ground Truth", linewidth=3.0)
    plt.plot(data["sx"], data["sz"],
             label=f"Stereo VO (SE(3), RMSE={data['rmse_s']:.3f} m)",
             linewidth=2.5)

    plt.scatter(data["gx"][0], data["gz"][0], s=90, marker="o", label="Start")
    plt.scatter(data["gx"][-1], data["gz"][-1], s=90, marker="x", label="GT End")
    plt.scatter(data["sx"][-1], data["sz"][-1], s=90, marker="x", label="Stereo End")

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Room2 - Stereo VO")
    plt.grid(True, alpha=0.35)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_room2_combined(data, out_path):
    plt.figure(figsize=(11, 8))
    plt.plot(data["gx"], data["gz"], label="Ground Truth", linewidth=3.0)
    plt.plot(data["mx"], data["mz"],
             label=f"Monocular VO (Sim(3), RMSE={data['rmse_m']:.3f} m)",
             linewidth=2.5)
    plt.plot(data["sx"], data["sz"],
             label=f"Stereo VO (SE(3), RMSE={data['rmse_s']:.3f} m)",
             linewidth=2.5)

    plt.scatter(data["gx"][0], data["gz"][0], s=90, marker="o", label="Start")
    plt.scatter(data["gx"][-1], data["gz"][-1], s=90, marker="x", label="GT End")
    plt.scatter(data["mx"][-1], data["mz"][-1], s=90, marker="x", label="Mono End")
    plt.scatter(data["sx"][-1], data["sz"][-1], s=90, marker="x", label="Stereo End")

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Room2 - Combined")
    plt.grid(True, alpha=0.35)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------- CORRIDOR / OUTDOORS ----------------

def prepare_start_aligned(traj_a, traj_b):
    A = load_tum(traj_a)
    B = load_tum(traj_b)

    A_xyz = A[:, 1:4].copy()
    B_xyz = B[:, 1:4].copy()

    A_xyz -= A_xyz[0]
    B_xyz -= B_xyz[0]

    ax, az = A_xyz[:, 0], A_xyz[:, 2]
    bx, bz = B_xyz[:, 0], B_xyz[:, 2]

    ax, az = smooth_xy(ax, az, win=9)
    bx, bz = smooth_xy(bx, bz, win=9)

    drift_a = np.linalg.norm(A_xyz[-1])
    drift_b = np.linalg.norm(B_xyz[-1])

    return {
        "ax": ax, "az": az,
        "bx": bx, "bz": bz,
        "drift_a": drift_a,
        "drift_b": drift_b,
    }


def add_inset(ax_main, x1, z1, x2=None, z2=None):
    axins = inset_axes(ax_main, width="35%", height="35%", loc="lower left")
    axins.plot(x1, z1)
    if x2 is not None and z2 is not None:
        axins.plot(x2, z2)
    axins.scatter(0, 0)
    axins.scatter(x1[-1], z1[-1])
    if x2 is not None and z2 is not None:
        axins.scatter(x2[-1], z2[-1])
    zoom = 2.0
    axins.set_xlim(-zoom, zoom)
    axins.set_ylim(-zoom, zoom)
    axins.set_title("Zoom", fontsize=8)
    axins.grid(True, alpha=0.3)


def plot_single_start_aligned(x, z, label, drift, title, out_path):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(x, z, label=f"{label} (drift={drift:.2f} m)", linewidth=2.8)
    ax.scatter(0, 0, s=90, label="Start")
    ax.scatter(x[-1], z[-1], s=90, marker="x", label=f"{label} End")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.axis("equal")
    ax.legend()

    add_inset(ax, x, z)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_combined_start_aligned(data, label_a, label_b, title, out_path):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(data["ax"], data["az"], label=f"{label_a} (drift={data['drift_a']:.2f} m)", linewidth=2.8)
    ax.plot(data["bx"], data["bz"], label=f"{label_b} (drift={data['drift_b']:.2f} m)", linewidth=2.8)

    ax.scatter(0, 0, s=90, label="Start")
    ax.scatter(data["ax"][-1], data["az"][-1], s=90, marker="x", label=f"{label_a} End")
    ax.scatter(data["bx"][-1], data["bz"][-1], s=90, marker="x", label=f"{label_b} End")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.axis("equal")
    ax.legend()

    add_inset(ax, data["ax"], data["az"], data["bx"], data["bz"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room2_gt", required=True)
    ap.add_argument("--room2_mono", required=True)
    ap.add_argument("--room2_stereo", required=True)
    ap.add_argument("--corridor3_mono", required=True)
    ap.add_argument("--corridor3_stereo", required=True)
    ap.add_argument("--outdoors5_mono", required=True)
    ap.add_argument("--outdoors5_stereo", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Room2
    room2 = prepare_room2(args.room2_gt, args.room2_mono, args.room2_stereo)
    plot_room2_mono(room2, out_dir / "room2_mono.png")
    plot_room2_stereo(room2, out_dir / "room2_stereo.png")
    plot_room2_combined(room2, out_dir / "room2_combined.png")

    # Corridor3
    corr = prepare_start_aligned(args.corridor3_mono, args.corridor3_stereo)
    plot_single_start_aligned(
        corr["ax"], corr["az"], "Monocular VO", corr["drift_a"],
        "Corridor3 - Monocular VO", out_dir / "corridor3_mono.png"
    )
    plot_single_start_aligned(
        corr["bx"], corr["bz"], "Stereo VO", corr["drift_b"],
        "Corridor3 - Stereo VO", out_dir / "corridor3_stereo.png"
    )
    plot_combined_start_aligned(
        corr, "Monocular VO", "Stereo VO",
        "Corridor3 - Combined", out_dir / "corridor3_combined.png"
    )

    # Outdoors5
    out = prepare_start_aligned(args.outdoors5_mono, args.outdoors5_stereo)
    plot_single_start_aligned(
        out["ax"], out["az"], "Monocular VO", out["drift_a"],
        "Outdoors5 - Monocular VO", out_dir / "outdoors5_mono.png"
    )
    plot_single_start_aligned(
        out["bx"], out["bz"], "Stereo VO", out["drift_b"],
        "Outdoors5 - Stereo VO", out_dir / "outdoors5_stereo.png"
    )
    plot_combined_start_aligned(
        out, "Monocular VO", "Stereo VO",
        "Outdoors5 - Combined", out_dir / "outdoors5_combined.png"
    )


if __name__ == "__main__":
    main()
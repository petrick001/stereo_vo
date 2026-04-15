import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_tum(path):
    data = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 8:
                continue
            vals = list(map(float, p))
            data.append(vals)
    return np.array(data)


parser = argparse.ArgumentParser()
parser.add_argument("--gt", required=True)
parser.add_argument("--mono", required=True)
parser.add_argument("--stereo", required=True)
args = parser.parse_args()

gt = load_tum(args.gt)
mono = load_tum(args.mono)
stereo = load_tum(args.stereo)

gt_xyz = gt[:,1:4]
mono_xyz = mono[:,1:4]
stereo_xyz = stereo[:,1:4]

# align starting point
gt_xyz -= gt_xyz[0]
mono_xyz -= mono_xyz[0]
stereo_xyz -= stereo_xyz[0]

plt.figure(figsize=(8,6))

plt.plot(gt_xyz[:,0], gt_xyz[:,2], label="Ground Truth", linewidth=3)
plt.plot(stereo_xyz[:,0], stereo_xyz[:,2], label="Stereo VO", linewidth=2)
plt.plot(mono_xyz[:,0], mono_xyz[:,2], label="Monocular VO", linewidth=2)

plt.xlabel("x (meters)")
plt.ylabel("z (meters)")
plt.title("Trajectory Comparison (Room2)")
plt.legend()
plt.grid(True)
plt.axis("equal")

plt.show()
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--traj", required=True)
args = parser.parse_args()

data = np.loadtxt(args.traj)

x = data[:,1]
z = data[:,3]

plt.figure()
plt.plot(x, z, linewidth=2)
plt.xlabel("x (meters)")
plt.ylabel("z (meters)")
plt.title("Stereo VO Trajectory")
plt.axis("equal")
plt.grid(True)

plt.show(block=True)
input("Press ENTER to close plot...")
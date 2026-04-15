import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_tum(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            ts, tx, ty, tz = map(float, parts[:4])
            data.append([ts, tx, ty, tz])
    return np.array(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    args = ap.parse_args()

    T = load_tum(args.traj)
    if T.size == 0:
        raise RuntimeError("No poses loaded.")

    x = T[:,1]
    y = T[:,2]
    z = T[:,3]

    plt.figure()
    plt.plot(x, z)  # x-z top-down-ish
    plt.xlabel("tx (up-to-scale)")
    plt.ylabel("tz (up-to-scale)")
    plt.title("Monocular VO trajectory (x vs z)")
    plt.show()

if __name__ == "__main__":
    main()

import argparse
import numpy as np


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
parser.add_argument("--traj", required=True)
args = parser.parse_args()

traj = load_tum(args.traj)

start = traj[0,1:4]
end = traj[-1,1:4]

drift = np.linalg.norm(end - start)

print("Start position:", start)
print("End position:", end)
print(f"Start-End Drift: {drift:.3f} meters")
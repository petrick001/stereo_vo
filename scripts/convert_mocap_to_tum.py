import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

rows = []

with open(args.csv) as f:
    reader = csv.reader(f)
    for r in reader:
        if r[0].startswith("#"):
            continue

        ts = float(r[0])
        tx = float(r[1])
        ty = float(r[2])
        tz = float(r[3])

        qw = float(r[4])
        qx = float(r[5])
        qy = float(r[6])
        qz = float(r[7])

        rows.append((ts, tx, ty, tz, qx, qy, qz, qw))

with open(args.out, "w") as f:
    for r in rows:
        f.write(
            f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]} {r[6]} {r[7]}\n"
        )

print("Wrote", len(rows), "poses")
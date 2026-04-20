# From Monocular VO to Stereo Visual Odometry on TUM VI

This project implements a classical geometry-based **Monocular Visual Odometry (VO)** pipeline and extends it to **Stereo Visual Odometry (SVO)** with metric scale using the **TUM VI dataset**.

The project follows the assignment requirement of building a non-deep-learning, reproducible stereo VO system using the provided calibration and evaluation sequences.

The exported **EuRoC/DSO 512×512** version of the dataset was used for experiments.

---

# Features

## Monocular VO

* Feature detection and matching
* Essential matrix estimation with RANSAC
* Relative pose recovery
* Trajectory chaining (up-to-scale)

## Stereo VO

* Stereo disparity estimation
* Depth recovery from disparity
* 3D–2D pose estimation using PnP + RANSAC
* Metric-scale trajectory estimation

## Evaluation

* ATE (Absolute Trajectory Error)
* Start–End Drift
* Trajectory comparison plots
* Reproducible scripts

---

# Dataset

Sequences used:

* Room2
* Corridor3
* Outdoors5

Dataset source: TUM VI Benchmark

---

# Repository Structure

```text
stereo_vo/
├── configs/
├── results/
│   ├── archive/
│   ├── groundtruth/
│   ├── plots/
│   └── trajectories/
├── scripts/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

* `configs/` : sequence configuration files
* `results/trajectories/` : final TUM trajectories
* `results/plots/` : final figures
* `results/groundtruth/` : reference trajectory
* `results/archive/` : debug/intermediate outputs
* `scripts/` : runnable pipeline and evaluation scripts
* `src/` : reserved for reusable modules (current implementation is script-based)

---

# Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version: **3.10+**

---

# How to Run

## Monocular VO (Room2)

```bash
python scripts/run_mono_step4_trajectory.py --seq data\dataset-room2_512_16\dataset-room2_512_16\mav0 --calib data\dataset-room2_512_16\dataset-room2_512_16\dso\camchain.yaml --out results\trajectories\mono_room2.tum
```

## Stereo VO (Room2)

```bash
python scripts/run_stereo_clean_trajectory.py --seq data\dataset-room2_512_16\dataset-room2_512_16\mav0 --calib data\dataset-room2_512_16\dataset-room2_512_16\dso\camchain.yaml --out results\trajectories\stereo_room2_final.tum
```

---

# Evaluation

## ATE Example

```bash
python scripts/evaluate_ate_simple.py --gt results\groundtruth\room2_groundtruth.tum --est results\trajectories\stereo_room2_final.tum
```

## Drift Example

```bash
python scripts/compute_start_end_drift.py --traj results\trajectories\stereo_room2_final.tum
```

---

# Plotting

```bash
python scripts/plot_final_comparison_room2.py
python scripts/plot_final_nice.py
```

---

# Final Outputs

## Trajectories

Located in `results/trajectories/`

* mono_room2.tum
* mono_corridor3.tum
* mono_outdoors5.tum
* stereo_room2_final.tum
* stereo_corridor3.tum
* stereo_outdoors5.tum

## Plots

Located in `results/plots/`

* room2_final_plot.png
* corridor3_final_plot.png
* outdoors5_final_plot.png

## Ground Truth

Located in `results/groundtruth/`

* room2_groundtruth.tum

---

# Reproducibility

* OS: Windows
* Language: Python
* Libraries: NumPy, OpenCV, SciPy, Matplotlib
* Classical geometry-based pipeline
* No deep learning methods used
* Uses provided calibration files
* Final outputs saved in TUM format
--- 
# Random seed: deterministic OpenCV / NumPy pipeline where applicable.
---

# Author

Theodore Petrick Reimmer
MSc VIBOT

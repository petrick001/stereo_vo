# Stereo Visual Odometry on TUM VI

This project implements a classical geometry-based monocular visual odometry (VO) pipeline and extends it to stereo visual odometry (SVO) with metric scale on the TUM VI dataset.

## Dataset
Sequences used:
- Room2
- Corridor3
- Outdoors5

## Repository Structure
- `scripts/` : runnable scripts for monocular VO, stereo VO, plotting, and evaluation
- `src/` : source modules
- `results/trajectories/` : final TUM trajectory files
- `results/plots/` : final trajectory comparison plots
- `results/groundtruth/` : Room2 ground truth
- `results/archive/` : older/debug trajectory outputs

## Setup
Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
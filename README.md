<div align="center">
  <h1>DrivingScene-3DGS 🚗💨✨</h1>
  <p><strong>LiDAR-Prior Composite 3D Gaussian Splatting & Photorealistic Colorization for Autonomous Driving Scenes</strong></p>
  
  [![Status](https://img.shields.io/badge/Status-Active_Development-orange.svg)]()
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()
  [![ROS](https://img.shields.io/badge/ROS-Noetic-brightgreen.svg)]()
</div>

---

> [!WARNING]
> **Status: 🚧 Work in Progress / Active Development**  
> This project is currently in active research and development on the `ondevelopment` branch. Features, calibration routines, and data formatters are subject to change.

## 📌 Overview

**DrivingScene-3DGS** is a complete pipeline designed to achieve clean, photorealistic 3D scene reconstruction and dense LiDAR point cloud colorization from autonomous vehicle datasets.

Traditional real-time colorization methods (such as direct VIO / ESIKF back-projection in R3LIVE) suffer from:
* **Double walls and ghosting** during sharp turns or velocity changes.
* **Severe visual blur & drift** caused by time-sync or extrinsic calibration discrepancies.
* **Camera field-of-view limits** leaving non-overlapping regions uncolored.

This project implements a **LiDAR-Prior Composite Gaussian Splatting architecture** (inspired by *DrivingGaussian* [CVPR 2024] and *3DGS* [SIGGRAPH 2023]):
1. **Geometric Ground Truth**: 100% frozen, drift-free LiDAR trajectory and geometric point cloud from **FAST-LIO**.
2. **LiDAR Seed Prior**: Downsampled point clouds (~1M points) serve as explicit 3D Gaussian seeds, eliminating the need for brittle Structure-from-Motion (SfM) initialization.
3. **Surround Multi-Camera Alignment**: Rigorous time synchronization and pinhole projection across 5-6 surround vehicle cameras.
4. **View-Consistent Appearance Modeling**: Optimization of spherical harmonics, scale, and opacity to eliminate floaters and double-surface artifacts.

---

## 🛠 Repository Structure

```text
.
├── IMPLEMENTATION_PLAN.md      # Detailed 4-phase technical roadmap & training guide
├── THEORY_AND_MATH.md          # Comprehensive mathematical formulation (Covariances, SH, ESIKF)
├── colorize_fastlio_map.py     # Multi-camera bag parser, pose sync, and COLMAP export engine
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start

### 1. Requirements

Ensure you have Python 3.8+ with PyTorch and CUDA support, as well as a ROS environment (e.g. ROS Noetic):

```bash
# Install Python dependencies
pip install -r requirements.txt

# Source your ROS workspace (required for processing .bag files)
source /opt/ros/noetic/setup.bash
```

### 2. Export COLMAP Format for 3DGS Training

Extract synchronized surround images, calibrated camera poses, and LiDAR 3D seed points from your ROS bag. This exports the data into a structure suitable for Gaussian Splatting training.

```bash
python3 colorize_fastlio_map.py \
    --pcd  /path/to/lio_map_down_0.1m.pcd \
    --traj /path/to/lic_lio.log \
    --bag  /path/to/your_bag.bag \
    --export_colmap \
    --colmap_out ./colmap_dataset/ \
    --out  /tmp/dummy.pcd \
    --every_n 5
```

### 3. Direct Distance-Weighted PCD Colorization

To directly generate a colored `.pcd` pointcloud map without neural training (using simple projection and distance weighting):

```bash
python3 colorize_fastlio_map.py \
    --pcd  /path/to/lio_map_xyzi.pcd \
    --traj /path/to/lic_lio.log \
    --bag  /path/to/your_bag.bag \
    --out  ./lio_map_colored.pcd \
    --every_n 3
```

---

## 🌿 Branches

* `main`: Stable documentation, plans, and verified release scripts.
* `ondevelopment`: Active development branch with experimental LiDAR priors, dynamic object handling, and continuous improvements.

---

## 📚 References & Acknowledgments

* **3D Gaussian Splatting for Real-Time Radiance Field Rendering** (SIGGRAPH 2023) — Kerbl et al.
* **DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes** (CVPR 2024) — Zhou et al.
* **FAST-LIO2 / R3LIVE** — HKU MaRS Lab.

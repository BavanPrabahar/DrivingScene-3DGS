# 3D Gaussian Splatting — Detailed Implementation Plan
## Goal: Colorize Uncolored FAST-LIO PCD Using 3DGS

---

## What We Are Building

```
INPUT:
  lio_map_xyzi.pcd      ← FAST-LIO output (XYZ + intensity, NO colour)
  your_drive.bag        ← ROS bag with 6 camera streams
  lic_lio.log           ← FAST-LIO trajectory (vehicle pose at every timestamp)
  6camera_fastlivo2.txt ← Camera calibration (Rcl, Pcl, fx, fy, cx, cy)

OUTPUT:
  lio_map_colored.pcd   ← Same geometry, every point now has [R, G, B]
```

Pipeline overview:
```
Phase 1: Data Preparation
  Extract images from ROS bag
  Pair each image with its camera pose (from lic_lio.log + calibration)
  Export in COLMAP format (standard input for 3DGS)

Phase 2: 3DGS Training
  Feed COLMAP data to gaussian-splatting or nerfstudio
  Train for 30,000 iterations (~1-3 hours on GPU)
  Output: trained Gaussian model (.ply with position/shape/opacity/SH coefficients)

Phase 3: Colour Extraction
  Load trained Gaussian model
  For each point in your FAST-LIO PCD:
    - Find the nearest trained Gaussian(s)
    - Query its colour from the "toward vehicle" direction
    - Assign that colour to the PCD point

Phase 4: Export
  Save coloured XYZRGB .pcd
  Visualise in CloudCompare / Open3D
```

---

## Phase 1 — Data Preparation

### 1A. Extract images from ROS bag

You need images paired with timestamps.
The `colorize_fastlio_map.py --export_colmap` flag already does this.

```bash
source /opt/ros/noetic/setup.bash

python3 ~/Videos/r3live_ws_hi/r3live_ws/colorize_fastlio_map.py \
    --pcd   ~/r3live_output/lio_map_xyzi.pcd \
    --bag   /path/to/your_drive.bag \
    --traj  ~/r3live_output/lic_lio.log \
    --export_colmap \
    --colmap_out  ./colmap_input/ \
    --out   /tmp/dummy.pcd \
    --every_n 5   # extract every 5th frame to reduce data size
```

This produces:
```
colmap_input/
  images/
    front/       kf000001.jpg  kf000002.jpg  ...
    front_left/  kf000001.jpg  ...
    back/        kf000001.jpg  ...
    back_left/   kf000001.jpg  ...
    back_right/  kf000001.jpg  ...
  sparse/0/
    cameras.txt  ← camera intrinsics per camera
    images.txt   ← camera pose per image (world→camera transform)
    points3D.txt ← LiDAR map points as sparse 3D seed
```

### 1B. Verify the COLMAP export

Run this to visually inspect camera poses:

```bash
pip install open3d
python3 - << 'EOF'
import numpy as np, open3d as o3d

# Read images.txt and visualize camera frustums
cams = []
with open("./colmap_input/sparse/0/images.txt") as f:
    for line in f:
        if line.startswith("#") or len(line.strip()) == 0: continue
        parts = line.split()
        if len(parts) == 10:  # header line
            qw,qx,qy,qz = map(float, parts[1:5])
            tx,ty,tz    = map(float, parts[5:8])
            from scipy.spatial.transform import Rotation as R
            R_cw = R.from_quat([qx,qy,qz,qw]).as_matrix()
            # Camera position in world = -R_cw^T @ t_cw
            cam_pos = -R_cw.T @ np.array([tx,ty,tz])
            cams.append(cam_pos)

pts = o3d.geometry.PointCloud()
pts.points = o3d.utility.Vector3dVector(np.array(cams))
pts.paint_uniform_color([1,0,0])  # red dots = camera positions
o3d.visualization.draw_geometries([pts])
EOF
```

Camera positions should trace the shape of your test track. If they look scrambled, the timestamp matching between bag and lic_lio.log needs tuning.

### 1C. Understand what cameras.txt contains

```
# cameras.txt format:
# CAMERA_ID  MODEL  W  H  fx  fy  cx  cy  k1  k2
1  OPENCV  1280  720  967.8213  873.9037  629.7818  358.5338  -0.1540  0.0982
2  OPENCV  1280  720  970.4983  876.2753  635.2812  351.8158  -0.1639  0.1161
...
```

These come directly from your `6camera_fastlivo2.txt`. Nothing to change.

### 1D. Understand what images.txt contains

```
# images.txt format:
# IMAGE_ID  qw  qx  qy  qz  tx  ty  tz  CAMERA_ID  NAME
# (one blank line between images)
1  0.99  0.01  0.02  0.00  -12.3  4.5  0.2  1  front/kf000001.jpg
```

The pose here is **camera-in-world** (T_cam_world), computed as:
```
T_cam_world = T_cam_lidar × T_lidar_world
            = [Rcl | tcl] × T_wl^(-1)
```
This is computed correctly by the export_colmap() function using your R_cl and t_cl values.

---

## Phase 2 — 3DGS Training

### 2A. Install

**Option 1: Nerfstudio (recommended for beginners — clean API)**
```bash
# Create isolated environment
conda create -n nerfstudio python=3.10
conda activate nerfstudio

# PyTorch with CUDA
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Nerfstudio + splatfacto (3DGS backend)
pip install nerfstudio
ns-install-cli
```

**Option 2: Original gaussian-splatting repo (more control)**
```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
conda env create --file environment.yml
conda activate gaussian_splatting
```

### 2B. Train with Nerfstudio

```bash
conda activate nerfstudio

ns-train splatfacto \
    --data ./colmap_input/ \
    --output-dir ./gs_output/ \
    --viewer.quit-on-train-completion True \
    --pipeline.model.sh-degree 3 \
    --pipeline.model.num-downscales 2 \
    --max-num-iterations 30000
```

Key parameters:
- `--pipeline.model.sh-degree 3` — spherical harmonics degree (3 = full view-dependent colour)
- `--max-num-iterations 30000` — training steps (more = better quality, more time)
- `--pipeline.model.num-downscales 2` — start training at half resolution, upscale during training

Monitor training at: `http://localhost:7007` (Nerfstudio viewer opens automatically)

**What to watch during training:**
- PSNR (Peak Signal-to-Noise Ratio) should increase: 15 dB (bad) → 25 dB (good) → 30+ dB (excellent)
- Number of Gaussians grows from your PCD size (~1M) to 3-6M as densification runs
- After ~5,000 iterations, rendered images should look like your camera photos

### 2C. Train with original gaussian-splatting repo

```bash
cd gaussian-splatting

python train.py \
    -s ./colmap_input/ \
    -m ./output/ \
    --iterations 30000 \
    --sh_degree 3 \
    --images images \
    --eval
```

Output after training:
```
output/
  point_cloud/
    iteration_7000/   point_cloud.ply   ← checkpoint at 7k
    iteration_30000/  point_cloud.ply   ← final model ← USE THIS
  cameras.json
  cfg_args
```

The `point_cloud.ply` contains ALL Gaussian parameters:
- `x, y, z` — Gaussian centre
- `f_dc_0, f_dc_1, f_dc_2` — DC (constant) SH coefficients (base RGB)
- `f_rest_0 ... f_rest_44` — higher-order SH coefficients (view-dependent)
- `opacity` — raw opacity (logit scale, apply sigmoid to get 0→1)
- `scale_0, scale_1, scale_2` — log-scale of Gaussian axes
- `rot_0, rot_1, rot_2, rot_3` — quaternion defining Gaussian orientation

---

## Phase 3 — Colour Extraction from Trained 3DGS

This is the custom step — no existing tool does this out of the box.

### 3A. Load the trained Gaussian model

```python
import numpy as np
from plyfile import PlyData   # pip install plyfile

def load_gaussians(ply_path):
    """
    Load trained 3DGS .ply and return all parameters as numpy arrays.
    """
    ply = PlyData.read(ply_path)
    v = ply['vertex']

    # Position
    xyz     = np.stack([v['x'], v['y'], v['z']], axis=1)     # (N, 3)

    # Opacity: stored as raw logit, convert to [0,1] via sigmoid
    opacity = 1.0 / (1.0 + np.exp(-v['opacity'].astype(np.float32)))  # (N,)

    # DC colour coefficients (SH degree 0):  stored as (f_dc_0, f_dc_1, f_dc_2)
    # These give the base (view-independent) colour
    f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1)  # (N, 3)

    # Higher SH coefficients: f_rest_0 ... f_rest_44 for degree 3
    # 15 coefficients per channel × 3 channels = 45 total
    try:
        f_rest = np.stack([v[f'f_rest_{i}'] for i in range(45)], axis=1)  # (N, 45)
        # Reshape to (N, 15, 3): 15 SH coefficients per RGB channel
        f_rest = f_rest.reshape(-1, 15, 3)
    except Exception:
        f_rest = np.zeros((len(xyz), 15, 3), dtype=np.float32)

    print(f"[3DGS] Loaded {len(xyz):,} Gaussians from {ply_path}")
    print(f"[3DGS] Opacity range: {opacity.min():.3f} → {opacity.max():.3f}")
    print(f"[3DGS] Solid Gaussians (opacity>0.3): {(opacity>0.3).sum():,}")

    return {
        'xyz':     xyz.astype(np.float32),
        'opacity': opacity.astype(np.float32),
        'f_dc':    f_dc.astype(np.float32),
        'f_rest':  f_rest.astype(np.float32),
    }
```

### 3B. Evaluate Spherical Harmonics

SH converts view-direction to colour.
For each Gaussian, colour = `SH_basis(direction) · SH_coefficients`

```python
def eval_sh_deg3(dir_vec, f_dc, f_rest):
    """
    Evaluate SH colour for a batch of Gaussians given a viewing direction.

    Args:
        dir_vec: (3,) unit vector pointing FROM scene point TOWARD camera
        f_dc:    (N, 3)  DC SH coefficients
        f_rest:  (N, 15, 3)  higher-order SH coefficients

    Returns:
        colours: (N, 3) RGB in [-1, 1] range (apply sigmoid to get [0,1])
    """
    x, y, z = dir_vec[0], dir_vec[1], dir_vec[2]

    # SH basis functions (degree 0 and 1 are most important)
    # Degree 0: constant
    C0 = 0.28209479177387814

    # Degree 1:
    C1 = 0.4886025119029199

    # Degree 2:
    C2 = [1.0925484305920792,
          -1.0925484305920792,
          0.31539156525252005,
          -1.0925484305920792,
          0.5462742152960396]

    # Degree 3:
    C3 = [-0.5900435899266435,
           2.890611442640554,
          -0.4570457994644658,
           0.3731763325901154,
          -0.4570457994644658,
           1.445305721320277,
          -0.5900435899266435]

    # Compute colour = sum of (SH_basis × coefficient) for each degree
    # Degree 0 (1 coefficient, view-independent base colour)
    colour = C0 * f_dc   # (N, 3)

    # Degree 1 (3 coefficients, linear view-dependence)
    colour += C1 * (-y) * f_rest[:, 0, :]   # Y_1^-1
    colour += C1 * ( z) * f_rest[:, 1, :]   # Y_1^0
    colour += C1 * (-x) * f_rest[:, 2, :]   # Y_1^1

    # Degree 2 (5 coefficients)
    xy = x*y;  xz = x*z;  yz = y*z
    x2 = x*x;  y2 = y*y;  z2 = z*z
    colour += C2[0] * xy              * f_rest[:, 3, :]
    colour += C2[1] * yz              * f_rest[:, 4, :]
    colour += C2[2] * (2*z2-x2-y2)   * f_rest[:, 5, :]
    colour += C2[3] * xz              * f_rest[:, 6, :]
    colour += C2[4] * (x2-y2)        * f_rest[:, 7, :]

    # Degree 3 (7 coefficients)
    colour += C3[0] * y*(3*x2-y2)    * f_rest[:, 8,  :]
    colour += C3[1] * xy*z            * f_rest[:, 9,  :]
    colour += C3[2] * y*(4*z2-x2-y2) * f_rest[:, 10, :]
    colour += C3[3] * z*(2*z2-3*x2-3*y2)*f_rest[:,11,:]
    colour += C3[4] * x*(4*z2-x2-y2) * f_rest[:, 12, :]
    colour += C3[5] * z*(x2-y2)      * f_rest[:, 13, :]
    colour += C3[6] * x*(x2-3*y2)    * f_rest[:, 14, :]

    # Activate: sigmoid maps unbounded SH output to [0, 1]
    colour = 1.0 / (1.0 + np.exp(-(colour + 0.5)))   # note: 3DGS adds 0.5 bias
    return np.clip(colour, 0.0, 1.0)    # (N, 3) RGB in [0,1]
```

### 3C. Assign colour to each FAST-LIO PCD point

```python
import open3d as o3d
from scipy.spatial import KDTree
from tqdm import tqdm

def colorize_pcd_from_3dgs(pcd_path, gaussians, traj_poses, out_path,
                             opacity_thresh=0.1, k_neighbours=5):
    """
    Assign RGB to every FAST-LIO map point using the trained 3DGS model.

    Args:
        pcd_path      : uncolored FAST-LIO .pcd
        gaussians     : dict from load_gaussians()
        traj_poses    : list of (timestamp, T_world_lidar 4×4) from lic_lio.log
        out_path      : output colored .pcd
        opacity_thresh: only use Gaussians with opacity > this value
        k_neighbours  : how many nearby Gaussians to blend
    """
    print(f"[LOAD] {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)   # (N, 3)
    N   = len(pts)

    # ── Filter to solid Gaussians only ──────────────────────────────────────
    solid_mask = gaussians['opacity'] > opacity_thresh
    g_xyz     = gaussians['xyz'][solid_mask]          # (M, 3)
    g_opacity = gaussians['opacity'][solid_mask]      # (M,)
    g_f_dc    = gaussians['f_dc'][solid_mask]         # (M, 3)
    g_f_rest  = gaussians['f_rest'][solid_mask]       # (M, 15, 3)
    print(f"[3DGS] Using {len(g_xyz):,} solid Gaussians (opacity>{opacity_thresh})")

    # ── Build KDTree on Gaussian positions ──────────────────────────────────
    print("[3DGS] Building KDTree on Gaussian centres...")
    g_tree = KDTree(g_xyz)

    # ── For each map point, find nearest Gaussians + query colour ───────────
    # We need a viewing direction: use the trajectory centroid as "vehicle"
    # (simplified — for full accuracy, find nearest pose per point)
    all_positions = np.array([T[:3,3] for _, T in traj_poses])
    traj_centroid = all_positions.mean(axis=0)   # rough vehicle centre
    traj_tree     = KDTree(all_positions)

    final_colours = np.full((N, 3), 0.45, dtype=np.float32)  # grey default

    for i in tqdm(range(N), desc="Querying 3DGS", unit="pt"):
        pt = pts[i]

        # Find k nearest Gaussians
        dists, idxs = g_tree.query(pt, k=min(k_neighbours, len(g_xyz)))
        if np.isscalar(idxs):
            idxs  = [idxs]
            dists = [dists]

        # Find nearest vehicle position for this map point → viewing direction
        _, nearest_pose_idx = traj_tree.query(pt)
        T_wl   = traj_poses[nearest_pose_idx][1]
        veh_pos = T_wl[:3, 3]
        view_dir = pt - veh_pos                       # point → toward camera
        norm     = np.linalg.norm(view_dir)
        if norm < 1e-6:
            continue
        view_dir /= norm                              # unit vector

        # Query colour from each nearby Gaussian, blend by opacity × 1/dist
        rgb_accum = np.zeros(3, dtype=np.float64)
        w_total   = 0.0

        for d, gi in zip(dists, idxs):
            col = eval_sh_deg3(view_dir,
                               g_f_dc[gi:gi+1],      # (1, 3)
                               g_f_rest[gi:gi+1])    # (1, 15, 3)
            col = col[0]                              # (3,)
            w   = float(g_opacity[gi]) / (d + 0.01)  # weight: opacity / dist
            rgb_accum += w * col
            w_total   += w

        if w_total > 0:
            final_colours[i] = np.clip(rgb_accum / w_total, 0.0, 1.0)

    # ── Save ────────────────────────────────────────────────────────────────
    pcd.colors = o3d.utility.Vector3dVector(final_colours.astype(np.float64))
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"[DONE] Saved → {out_path}")
```

### 3D. Full pipeline script

```python
#!/usr/bin/env python3
"""
colorize_pcd_3dgs.py — Colorize FAST-LIO PCD using trained 3DGS model

Usage:
    python3 colorize_pcd_3dgs.py \
        --pcd    ~/r3live_output/lio_map_xyzi.pcd \
        --gs_ply ./output/point_cloud/iteration_30000/point_cloud.ply \
        --traj   ~/r3live_output/lic_lio.log \
        --out    ~/r3live_output/lio_map_3dgs_colored.pcd

Install:
    pip install open3d numpy scipy plyfile tqdm
"""

import argparse, sys
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

# paste load_gaussians(), eval_sh_deg3(), colorize_pcd_from_3dgs() here

def load_trajectory(path):
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            v = list(map(float, line.split()))
            if len(v) < 8: continue
            T = np.eye(4)
            T[:3,:3] = Rot.from_quat(v[4:8]).as_matrix()
            T[:3, 3] = v[1:4]
            poses.append((v[0], T))
    print(f"[TRAJ] {len(poses)} poses loaded")
    return poses

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pcd',    required=True)
    p.add_argument('--gs_ply', required=True, help='3DGS point_cloud.ply')
    p.add_argument('--traj',   required=True, help='lic_lio.log TUM format')
    p.add_argument('--out',    required=True)
    p.add_argument('--opacity_thresh', type=float, default=0.1)
    p.add_argument('--k',              type=int,   default=5)
    a = p.parse_args()

    poses    = load_trajectory(a.traj)
    gaussians = load_gaussians(a.gs_ply)
    colorize_pcd_from_3dgs(a.pcd, gaussians, poses, a.out,
                            opacity_thresh=a.opacity_thresh,
                            k_neighbours=a.k)

if __name__ == '__main__':
    main()
```

---

## Phase 4 — Verify Results

```bash
# View in Open3D
python3 -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('lio_map_3dgs_colored.pcd')
print(f'Points: {len(pcd.points):,}')
print(f'Has colours: {pcd.has_colors()}')
o3d.visualization.draw_geometries([pcd])
"

# Or view in CloudCompare (drag and drop the .pcd file)
```

---

## What You Must Study — Topic by Topic

### ✅ Topic 1 — Linear Algebra (you mostly know this already)

**Why:** Gaussian covariance Σ, projection math, SH basis vectors
**What specifically:**
- Covariance matrix eigendecomposition: `Σ = R @ diag(s²) @ R^T`
  - R = rotation (Gaussian orientation)
  - s = scale (Gaussian radius per axis)
- Matrix exponential / logarithm (used internally by 3DGS optimizer)
- Jacobian of camera projection (used for splatting)

**Exercise:**
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D Gaussian blob
mu = np.array([0.0, 0.0, 0.0])
Sigma = np.array([[4.0, 0.0, 0.0],   # elongated 4m in X
                  [0.0, 0.1, 0.0],   # thin 0.1m in Y  (like a wall)
                  [0.0, 0.0, 0.1]])  # thin 0.1m in Z

# Visualise as ellipsoid
vals, vecs = np.linalg.eig(Sigma)
# vecs[:,i] = i-th principal axis
# vals[i]   = i-th radius squared
print("Radii:", np.sqrt(vals))     # should be [2.0, 0.316, 0.316]
print("Axes:",  vecs)              # should be identity matrix
```

---

### ✅ Topic 2 — Camera Projection (you already know this from your calibration)

**You already have:** `P_cam = R_cl @ P_lidar + t_cl`, then `u = fx*x/z + cx`

**New step for 3DGS:** project a 3D Gaussian's covariance to 2D
This uses the **Jacobian of projection**:

$$J = \begin{bmatrix} \frac{f_x}{z} & 0 & -\frac{f_x x}{z^2} \\ 0 & \frac{f_y}{z} & -\frac{f_y y}{z^2} \end{bmatrix}$$

```python
def project_gaussian_covariance(Sigma_world, R_cw, t_cw, fx, fy, mu_world):
    """Project 3D Gaussian covariance to 2D image ellipse."""
    # Camera-frame Gaussian centre
    mu_cam = R_cw @ mu_world + t_cw
    x, y, z = mu_cam

    # Jacobian of projection at mu_cam
    J = np.array([
        [fx/z,  0,    -fx*x/z**2],
        [0,     fy/z, -fy*y/z**2]
    ])

    # Camera-frame covariance
    Sigma_cam = R_cw @ Sigma_world @ R_cw.T

    # 2D image-space covariance
    Sigma_2d = J @ Sigma_cam @ J.T   # (2, 2) — defines an ellipse in image

    return Sigma_2d, mu_cam
```

**Exercise:** Call this on a wall Gaussian (thin in one direction) and plot the resulting 2D ellipse. It should be very thin horizontally (wall edge-on) or a vertical line (wall face-on).

---

### ✅ Topic 3 — Alpha Compositing

**Why:** How 3DGS renders a pixel by blending multiple Gaussians front-to-back

**The formula:**

$$C_{pixel} = \sum_{i=1}^{N} c_i \cdot \alpha_i \cdot \prod_{j=1}^{i-1}(1 - \alpha_j)$$

$c_i$ = colour of i-th Gaussian
$\alpha_i$ = opacity × Gaussian density at this pixel
$\prod(1-\alpha_j)$ = "how much has NOT been covered by Gaussians in front of this one"

**Exercise — implement it for 1D:**
```python
# Imagine 5 colour layers (front to back)
colours  = np.array([[1.0,0,0], [0,1.0,0], [0,0,1.0], [1,1,0], [1,1,1]])
opacities = np.array([0.3, 0.5, 0.8, 0.6, 1.0])

# Alpha compositing loop
final = np.zeros(3)
T = 1.0  # transmittance = how much light gets through

for c, a in zip(colours, opacities):
    final += T * a * c   # this Gaussian's contribution
    T *= (1.0 - a)       # reduce remaining transmittance
    if T < 0.01:         # early stop — scene is fully occluded
        break

print("Rendered colour:", final)  # should be reddish-green (first two layers dominate)
```

---

### ✅ Topic 4 — Spherical Harmonics

**Why:** The view-dependent colour function of each Gaussian

**Core idea:** SH are basis functions defined on the unit sphere. Like Fourier series but for directions instead of time.

```
Degree 0: Y_0^0 = 0.282  (constant — same colour from all directions)
Degree 1: Y_1^-1 = 0.489*y   Y_1^0 = 0.489*z   Y_1^1 = 0.489*x
          (linear — smooth directional variation)
...higher degrees = higher-frequency angular variation
```

**Exercise — visualise SH basis:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Sample unit sphere
theta = np.linspace(0, np.pi, 50)
phi   = np.linspace(0, 2*np.pi, 100)
T, P  = np.meshgrid(theta, phi)

x = np.sin(T) * np.cos(P)
y = np.sin(T) * np.sin(P)
z = np.cos(T)

# SH degree 1, m=0:  Y_1^0 = 0.489 * z
Y10 = 0.489 * z

# Visualise: colour the sphere by Y_1^0 value
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, facecolors=plt.cm.RdBu((Y10+0.5)/1.0))
plt.title("SH basis Y_1^0 — top hemisphere = positive (blue), bottom = negative (red)")
plt.show()

# This means: if a Gaussian has a large Y_1^0 coefficient,
# it appears bright blue when viewed from above and red when viewed from below
```

**Key insight:** When you query `eval_sh_deg3(view_dir, f_dc, f_rest)`, you're computing:
`colour = f_dc[0]*Y_0^0 + f_rest[0]*Y_1^-1 + f_rest[1]*Y_1^0 + ...`

Higher SH degree = more angular detail = more realistic specular highlights. For your outdoor scene, degree 1-2 is usually enough.

---

### ✅ Topic 5 — Gradient Descent and Adam Optimizer

**Why:** Training 3DGS is just gradient descent on all Gaussian parameters

You don't need to implement this — PyTorch handles it. But you need to understand what it does:

```python
# Conceptually, 3DGS training is:
for iteration in range(30000):
    # Pick a random training camera
    cam = random.choice(training_cameras)

    # Render scene from that camera (differentiable!)
    rendered_image = render_gaussians(gaussians, cam)

    # Compute loss: how different is the render from the real photo?
    real_image = cam.gt_image                          # ground truth
    loss = L1_loss(rendered, real) + SSIM_loss(rendered, real)

    # Backpropagate: compute gradient of loss w.r.t. ALL Gaussian parameters
    loss.backward()

    # Adam step: update all parameters to reduce loss
    optimizer.step()        # moves μ, changes Σ, adjusts α and SH coefficients
    optimizer.zero_grad()
```

**What to study:** PyTorch `tensor.backward()`, `optimizer.step()`. Basic neural network tutorial (e.g., PyTorch 60-minute blitz) covers this completely.

---

### ✅ Topic 6 — COLMAP Format

**Why:** 3DGS expects data in COLMAP format (industry standard)

You need to understand what each file contains:
- `cameras.txt` — intrinsic parameters per camera model
- `images.txt` — extrinsic pose per image
- `points3D.txt` — 3D point seed (your LiDAR map)

The export script already generates this. Study the [COLMAP documentation](https://colmap.github.io/format.html) for 30 minutes to understand the coordinate conventions.

---

## Recommended Study Path — Week by Week

### Week 1 — Foundations (3–4 hours/day)
```
Day 1: Alpha compositing
  → Do the Exercise for Topic 3 (the 5-layer blending code)
  → Understand why order matters (front-to-back vs back-to-front)

Day 2: Camera projection
  → Re-read your 6camera_fastlivo2.txt
  → Do the Jacobian exercise (Topic 2)
  → Understand what Σ_2D means geometrically

Day 3: Spherical Harmonics
  → Do the sphere visualisation exercise (Topic 4)
  → Try different SH basis functions
  → Understand: degree 0 = view-independent, degree 3 = full specular

Day 4: Read 3DGS paper sections:
  → Abstract (5 min)
  → Figure 1-3 (understand the pipeline visually)
  → Section 4 (Differentiable 3DGS Rasterization) — 45 min, with pencil
  → Section 5.2 (Adaptive Control of Gaussians)
```

### Week 2 — Data Pipeline (2–3 hours/day)
```
Day 1: Run colorize_fastlio_map.py --export_colmap
  → Check the output images visually (do they look like your track?)
  → Check the camera poses (do they trace the track shape?)

Day 2: Install nerfstudio
  → conda env setup
  → Run ns-train on a small test dataset first
    (use nerfstudio's "poster" example dataset to verify installation)

Day 3: Train on YOUR data
  → Run ns-train splatfacto --data ./colmap_input/
  → Monitor PSNR in the viewer
  → At 7,000 iterations, does the render look like your track?

Day 4: Explore the trained model
  → Load the point_cloud.ply with load_gaussians()
  → Print opacity histogram: how many solid vs transparent Gaussians?
  → Visualise solid Gaussians in Open3D (position + opacity as colour)
```

### Week 3 — Colour Extraction (2–3 hours/day)
```
Day 1: Implement eval_sh_deg3()
  → Test: for a fixed Gaussian, query colour from 360° of directions
  → Plot as a sphere — does the colour change smoothly with direction?

Day 2: Implement colorize_pcd_from_3dgs()
  → Run on a small subset of PCD points first (first 1000)
  → Check: are colours reasonable? (wall = white/grey, grass = green, road = dark)

Day 3: Run on full PCD
  → Might take 30–60 min for a million-point map
  → Load result in CloudCompare, compare with Method A (projection) result

Day 4: Tune
  → Try opacity_thresh = 0.05 vs 0.3 (more vs fewer Gaussians)
  → Try k = 3 vs 10 neighbours
  → Compare visual quality
```

### Week 4 — Integration and Optimisation
```
Day 1-2: Speed up with vectorisation
  → The inner loop over PCD points can be vectorised with KDTree batch query
  → Process 10,000 points at once instead of one at a time

Day 3: Compare Method A vs Method C visually
  → Side by side in CloudCompare
  → Which handles occluded wall corners better?
  → Which is more colour-consistent on the long straight sections?

Day 4: Document and integrate into your pipeline
  → Add 3DGS colorization as an optional step after PGO map cleanup
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 8 GB (RTX 3060) | 16+ GB (RTX 3080/4080) |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB free | 200 GB (for images + model) |
| Training time | ~3 h (8 GB GPU) | ~45 min (16 GB GPU) |
| Inference (query all PCD points) | 30-60 min | 10 min |

> If you don't have a local GPU, use **Google Colab Pro** (A100, 40 GB VRAM)
> or **Vast.ai** (rent RTX 3090 for ~\$0.30/hour).

---

## Key Resources

| Resource | What for | Time |
|---|---|---|
| [3DGS Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | Core method | 2 hours |
| [Nerfstudio Docs](https://docs.nerf.studio/) | Training setup | 1 hour |
| [gaussian-splatting repo README](https://github.com/graphdeco-inria/gaussian-splatting) | Original implementation | 30 min |
| [COLMAP format docs](https://colmap.github.io/format.html) | Data format | 30 min |
| [SH tutorial (Stupid SH Tricks)](https://www.ppsloan.org/publications/StupidSH36.pdf) | Spherical Harmonics math | Sections 1-3, 1 hour |
| PyTorch 60-minute blitz | Gradient descent basics | 1 hour |

---

## Common Mistakes to Avoid

| Mistake | What happens | Fix |
|---|---|---|
| Not undistorting images before 3DGS | Blurry/distorted render at corners | Use `k1, k2` coefficients in cameras.txt — nerfstudio handles this automatically |
| Wrong world→camera pose sign | Cameras appear at mirror positions | Double-check: `t_cw = -R_cw @ t_wl + t_cl` NOT `+R_cw @ t_wl` |
| SH colour without `+0.5` bias | Colours are too dark/saturated | 3DGS stores colours in logit space with 0.5 bias — always add it before sigmoid |
| Low training PSNR (< 20 dB) | Rendered images look blurry | Check camera poses are correct; try reducing `--num-downscales` |
| Querying colour with wrong view direction | Colour is wrong (view from wrong side) | Direction must point FROM scene point TOWARD camera (not camera→point) |
| Using all Gaussians including transparent ones | Colour contaminated by empty space | Always filter `opacity > 0.1` before querying |

# 3D Gaussian Splatting (3DGS) — Deep Explanation for Your Use Case

> You are coming from a **LiDAR SLAM / robotics** background. I'll explain 3DGS by
> connecting every concept to things you already know from FAST-LIO and R3LIVE.

---

## The Core Problem 3DGS Solves

You have:
- A set of **images** taken from known camera positions (your 6 cameras + FAST-LIO poses)
- You want to **reconstruct what the 3D scene looks like** so you can:
  - Render it from any viewpoint you choose
  - Extract colour for any 3D point `[X, Y, Z]`

This is called **Novel View Synthesis** — synthesize an image from a camera position you never actually recorded.

The old approaches: mesh reconstruction (too slow, needs perfect depth), NeRF (correct but GPU-hours to train and milliseconds per pixel to render).

**3DGS (SIGGRAPH 2023, Kerbl et al.)** does this in:
- 30 minutes training on a GPU
- Under 1 ms to render a full 1080p frame (real-time)

---

## Step 1 — What is a "Gaussian" in 3DGS?

In your FAST-LIO world, a **point cloud** is a set of XYZ positions — each point is infinitely small, like a mathematical dot.

In 3DGS, instead of dots, the scene is represented as a set of **3D blobs** called **Gaussians**. Each Gaussian has:

```
μ   = [x, y, z]          ← centre position in 3D world (same as your PCD points)
Σ   = 3×3 covariance      ← shape/size/orientation of the blob
                            (a sphere, a flat disk, a stretched ellipsoid)
α   = opacity (0.0→1.0)   ← how solid this blob is
                            0.0 = invisible glass
                            1.0 = completely opaque
c   = colour (RGB)         ← but NOT a single RGB — more on this below
```

Think of each Gaussian like a **soft cloud of paint** floating in 3D space. The covariance Σ controls whether that paint cloud is:
- A sphere (isotropic) → like a ball of fog
- A flat disk → like paint splattered on a wall surface
- A long thin rod → like a lamp post or fence pole

The optimizer **learns the shape** of each Gaussian so that it represents the actual scene geometry — walls become flat disk Gaussians, poles become rod Gaussians, grass becomes small noisy spheres.

### Mathematically

The density of a Gaussian at world point **x** near centre **μ** is:

$$G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})}$$

This gives a value from 0.0 (far from centre) to 1.0 (at centre). Multiplied by α, it gives the actual opacity contribution at that point.

---

## Step 2 — View-Dependent Colour (Spherical Harmonics)

This is where 3DGS is smarter than just storing a single RGB per Gaussian.

**The problem:** in the real world, colour depends on viewing direction. A wet road looks dark when you look straight down, but looks mirror-bright when you look at it at a grazing angle. A metal pole looks grey from one side, white from another.

**Solution: Spherical Harmonics (SH)**

Instead of `colour = [R, G, B]`, each Gaussian stores **SH coefficients** — a set of numbers that encode "what colour does this blob appear when viewed from direction `[dx, dy, dz]`?"

```
c(direction) = SH_basis(direction) · SH_coefficients

SH degree 0: 1 coefficient per channel  = flat colour (same from all angles)
SH degree 1: 4 coefficients per channel = simple directional variation
SH degree 3: 16 coefficients per channel = full specular highlights
```

For your outdoor road scene, **degree 1 or 2 is sufficient** — the environment is lit by the sun from one direction, walls are diffuse, no strong specular.

You don't need to derive this yourself. The 3DGS code handles it. What you need to understand:
- Each Gaussian has ~48 numbers to represent colour (16 SH coefficients × 3 channels)
- When you render, you pass the viewing direction and get back `[R, G, B]`
- When you want colour for a map point, you pass the direction "toward the vehicle" and get the colour as the vehicle would see it

---

## Step 3 — Splatting: How 3DGS Renders an Image

"Splatting" means: **project each 3D Gaussian onto the 2D image plane**.

For each camera pose `(R, t)` and intrinsics `(fx, fy, cx, cy)`:

```
Step 1: Transform Gaussian centre to camera frame
         μ_cam = R @ μ_world + t

Step 2: Project covariance Σ from 3D to 2D image plane
         Σ_2D = J @ R @ Σ @ R^T @ J^T
         where J is the Jacobian of the projection function

Step 3: In image space, this Gaussian becomes a 2D ellipse
         (like a soft brush stroke)

Step 4: Sort all Gaussians by depth (back to front)

Step 5: Alpha-composite them front to back:
         C_pixel = Σ_i  c_i × α_i × Π_{j<i}(1 - α_j)
         (each Gaussian contributes colour × its opacity × how much hasn't been covered yet)
```

The last step is **volumetric alpha compositing** — the same math used in NeRF, and also in computer graphics for transparency effects. If you know how R3LIVE's photometric alignment works, this is the 3D generalization of it.

**The key insight:** this entire pipeline is **differentiable**. You can compute gradients of the rendered pixel colour with respect to all Gaussian parameters (position, shape, opacity, colour-SH). This means you can train it with backpropagation.

---

## Step 4 — Training Loop

```
Initialize:
  Start from your FAST-LIO PCD points as Gaussian centres
  Shape = small spheres (Σ = σ²I, σ ≈ 0.01 m)
  Opacity = 0.1 (semi-transparent)
  Colour = average of all nearby pixel colours (rough init)

For each training iteration:
  1. Pick a random keyframe (camera pose + real image)
  2. Render the scene from that camera pose → synthetic image
  3. Compute loss:
       L = L1_loss(rendered, real) + λ × SSIM_loss(rendered, real)
       (pixel-wise colour error + structural similarity error)
  4. Backpropagate loss through the differentiable renderer
  5. Update all Gaussian parameters with Adam optimizer
       μ  → Gaussians drift to better positions
       Σ  → Gaussians reshape to match geometry
       α  → solid objects become opaque, empty space becomes transparent
       SH → colours match the real images

After ~30,000 iterations:
  - Real walls → flat disk Gaussians, opacity ≈ 1.0, correct colour
  - Ghost walls (drift artifact) → these are inconsistent across views
                                 → optimizer sets opacity → 0.0
  - Empty air → opacity ≈ 0.0
```

This is exactly **gradient descent** — the same thing as training a neural network. The difference from a standard neural network: the "parameters" are the Gaussian properties, not neuron weights.

### Adaptive Densification

During training, 3DGS also:
- **Splits** Gaussians that are too big (one large Gaussian → two smaller ones)
- **Clones** Gaussians in underrepresented regions
- **Prunes** Gaussians with opacity < 0.005 (remove transparent dust)

This is how it grows from your initial LiDAR point count to millions of Gaussians while staying accurate.

---

## Step 5 — How You Use It for Map Colorization

After training, you have a trained 3DGS model saved as a `.ply` file containing all Gaussians.

**To get RGB for a FAST-LIO map point `P = [X, Y, Z]`:**

```python
# Method 1 — Direct SH query (fast, approximate)
# Find the nearest Gaussian to P:
nearest_gaussian = find_nearest(gaussians, P)
# Query its colour from the direction the vehicle approaches from:
direction = normalize(vehicle_pose - P)   # direction from point toward vehicle
rgb = evaluate_SH(nearest_gaussian.sh_coeffs, direction)

# Method 2 — Render a virtual camera (accurate, slower)
# Place a virtual camera at the nearest keyframe position:
virtual_cam = nearest_keyframe_pose
# Render the scene from that camera:
rendered_image = render_3dgs(model, virtual_cam)
# Project P onto the rendered image and sample the pixel:
uv = project(P, virtual_cam)
rgb = rendered_image[uv[1], uv[0]]
```

Method 2 is more accurate because it uses alpha compositing — it handles transparency and occlusion correctly.

---

## Step 6 — Why Ghost Walls Disappear

This is the most powerful feature for your double-wall problem.

Your ghost wall exists because of LiDAR drift — the vehicle registered two slightly different positions for the same physical wall. Both sets of points are in your map.

During 3DGS training:
- The **real wall** is seen consistently from every keyframe that drives past it. From all angles, it renders the same colour, same position. Loss → 0. Gaussians → opaque, correct position.
- The **ghost wall** (drift copy) is **inconsistent** across views:
  - From some keyframes, it appears in front of the real wall → wrong depth order → high loss
  - From other keyframes, it doesn't appear at all → but a Gaussian placed there doesn't help reduce loss
  - The optimizer's best solution: **set its opacity → 0.0** (make it invisible)

After training you can filter your FAST-LIO PCD:
```python
# Load trained Gaussians
opacity = torch.sigmoid(gaussian_model._opacity)  # per-Gaussian opacity

# Map each FAST-LIO PCD point to its nearest Gaussian
# Points whose nearest Gaussian has low opacity → likely ghost / dynamic object
real_mask  = opacity > 0.3   # solid geometry
ghost_mask = opacity < 0.05  # ghost or dynamic object
```

This gives you an **automatic ghost wall detector** that requires no hand-tuning.

---

## The Math You Need to Understand

Here is the exact learning path — in order:

### Level 1 — Prerequisites (you likely know most of this)
| Topic | Why needed | Resource |
|---|---|---|
| **Homogeneous coordinates** | Projection math (μ_cam = R @ μ + t) | Any computer vision textbook |
| **Camera model** (pinhole, intrinsics K) | Projecting Gaussians to image plane | You already use this in your calibration |
| **Rotation matrices / quaternions** | Gaussian orientation (covariance Σ) | Already in your FAST-LIO code |
| **Gradient descent / backprop** | Training loop | Any ML course intro |

### Level 2 — Core 3DGS concepts
| Topic | Why needed | Where to learn |
|---|---|---|
| **3D Gaussian function** | What a Gaussian blob is | Study the formula in Step 1 above |
| **Covariance matrix** (eigenvectors = shape axes) | How Σ encodes blob shape | Linear algebra: `numpy.linalg.eig` on a 3×3 |
| **Projection of a Gaussian** (Σ_2D = J Σ J^T) | Splatting step | Section 4 of the original 3DGS paper |
| **Spherical Harmonics** (degree 0–3) | View-dependent colour | Wikipedia SH + graphics tutorials |
| **Alpha compositing** (Porter-Duff) | How colours blend front-to-back | Any computer graphics course |
| **Adam optimizer** | Training Gaussian parameters | PyTorch tutorial |

### Level 3 — Implementation details
| Topic | Why needed | Where |
|---|---|---|
| **CUDA tile-based rasterizer** | Why 3DGS is fast (30 FPS rendering) | `diff-gaussian-rasterization` repo README |
| **Adaptive densification** | Why Gaussians split/merge during training | Section 5.2 of 3DGS paper |
| **COLMAP format** | How to feed your data to 3DGS | COLMAP documentation |

---

## What to Study — In Order

### Week 1 — Understand the idea
```
1. Read the 3DGS paper abstract + intro (5 pages):
   https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
   Focus on: what are Gaussians, what is splatting, what is the loss

2. Watch this 10-minute visual explanation:
   "3D Gaussian Splatting explained" — search on YouTube
   You'll see the Gaussians growing and shaping in real time

3. Read Section 4 (Differentiable 3D Gaussian Rasterization) of the paper
   This is the core math. Read it with pencil and paper.
```

### Week 2 — Understand the math
```
4. Study alpha compositing:
   C = Σ c_i α_i Π_{j<i}(1 - α_j)
   Code it in 5 lines of Python for a 1D case first.

5. Study Spherical Harmonics degree 0 and 1:
   https://www.ppsloan.org/publications/StupidSH36.pdf  (sections 1-3 only)
   Implement: given [dx,dy,dz] direction, compute SH basis → multiply by coeffs → get RGB

6. Study Gaussian covariance → 3D shape:
   import numpy as np
   Sigma = np.diag([2.0, 0.1, 0.1])  # elongated along X axis
   vals, vecs = np.linalg.eig(Sigma)
   # vecs are the principal axes, vals are the radii — draw this in matplotlib
```

### Week 3 — Run it on your data
```
7. Install nerfstudio (easier interface to 3DGS):
   pip install nerfstudio
   ns-install-cli

8. Export your data using the colorize script's --export_colmap flag

9. Train:
   ns-train splatfacto --data ./colmap_input/

10. Render a novel view — verify it looks like your test track

11. Extract Gaussian positions + opacity → apply to your PCD
```

### Week 4 — Apply to your map colorization problem
```
12. For each FAST-LIO map point:
    - Find nearest Gaussian
    - Evaluate SH coefficients for the "toward vehicle" direction
    - Assign colour

13. Filter ghost walls by opacity threshold

14. Save colored PCD → load in CloudCompare or Autoware
```

---

## Connecting 3DGS to What You Already Know

| Concept from your SLAM work | Equivalent in 3DGS |
|---|---|
| LiDAR point cloud (discrete XYZ) | Gaussian centres (μ) |
| Voxel map (each voxel has occupancy) | Each Gaussian has opacity α |
| ICP fitness score | 3DGS loss: rendered vs real image |
| GTSAM optimizer (minimise graph error) | Adam optimizer (minimise photometric error) |
| Keyframe (pose + scan at one moment) | Training view (camera pose + real image) |
| Loop closure (correct accumulated drift) | Opacity pruning (remove inconsistent geometry) |
| Intensity channel in LiDAR | Spherical harmonic coefficients (view-dep. colour) |
| NDT map (geometry for localization) | 3DGS model (geometry + appearance) |

---

## For Your Specific Scene (Test Track)

Your track has:
- **Long flat walls** → will become very thin flat disk Gaussians (near-zero thickness in the normal direction)
- **Grass median** → many small noisy sphere Gaussians
- **Traffic signs** → small but distinctive Gaussians with sharp colours
- **Sky** → very few Gaussians, mostly transparent (no solid surface)
- **Ghost wall** → Gaussians placed there by initialization (from your FAST-LIO PCD), but with opacity → 0 after training because they're inconsistent with the images

The result: a **photorealistic render** of your track from any viewpoint, with ghost walls automatically removed.

---

## Key Paper to Read

> Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023).
> **3D Gaussian Splatting for Real-Time Radiance Field Rendering.**
> ACM Transactions on Graphics (SIGGRAPH), 42(4).
> https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

Read in this order: Abstract → Figures 1-3 → Section 4 → Section 5.2 → Results.

---

## Summary of What 3DGS IS and IS NOT

| 3DGS IS | 3DGS IS NOT |
|---|---|
| A scene representation (like a point cloud, but with shape + colour + opacity) | A neural network with layers and weights |
| Trained by minimizing photometric loss between rendered and real images | Trained by supervised labelling |
| A renderer: given camera pose → produces a photo-realistic image | A segmentation or detection algorithm |
| A ghost-detector: inconsistent geometry → opacity → 0 | A traditional SLAM or localization system |
| Your tool for: assign RGB to FAST-LIO XYZ | A replacement for FAST-LIO geometry |

3DGS handles **appearance**. FAST-LIO handles **geometry**. They complement each other perfectly — FAST-LIO gives you accurate XYZ, 3DGS gives you the RGB and opacity for each XYZ.

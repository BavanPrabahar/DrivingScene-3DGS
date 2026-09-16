#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
colorize_fastlio_map.py
================================================================================
Adds RGB color to an uncolored FAST-LIO point cloud map (.pcd).

Pipeline:
  1. Load FAST-LIO map  (geometry-only XYZ or XYZI .pcd)
  2. Load vehicle poses (from FAST-LIO lic_lio.log  — TUM format)
  3. Load camera images from ROS bag (all 6 cameras) or an image folder
  4. For each map point: project onto all visible cameras, pick best color
     using distance-weighted accumulation across all keyframes
  5. Save colored XYZRGB .pcd

Usage:
  # With a rosbag (recommended):
  python3 colorize_fastlio_map.py \
      --pcd  ~/r3live_output/lio_map_xyzi.pcd \
      --bag  /path/to/your_drive.bag \
      --traj ~/r3live_output/lic_lio.log \
      --out  ~/r3live_output/lio_map_colored.pcd

  # COLMAP export for 3D Gaussian Splatting (Method C):
  python3 colorize_fastlio_map.py \
      --pcd  ~/r3live_output/lio_map_xyzi.pcd \
      --bag  /path/to/your_drive.bag \
      --traj ~/r3live_output/lic_lio.log \
      --export_colmap --colmap_out ./colmap_input/ \
      --out  /tmp/dummy.pcd

Dependencies:
  pip install open3d numpy scipy opencv-python tqdm
  source /opt/ros/noetic/setup.bash    # for --bag mode

After running, copy output to workspace:
  cp ~/r3live_output/lio_map_colored.pcd ~/Videos/r3live_ws_hi/r3live_ws/

================================================================================
"""

import os, sys, time, argparse
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import KDTree
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 6-CAMERA CALIBRATION
# Taken directly from:
#   ~/Videos/r3live_ws_hi/r3live_ws/6txtcamera        (ext R and t)
#   ~/Videos/r3live_ws_hi/r3live_ws/6camera_fastlivo2.txt  (intrinsics)
#
# Convention:  P_cam = R_cl @ P_lidar + t_cl
#   where R_cl = camera_ext_R  (rows = camera axes in lidar frame, transposed)
#   and   t_cl = camera_ext_t
# ─────────────────────────────────────────────────────────────────────────────

CAMERAS = {
    # ── FRONT  /camera_24264461 ─────────────────────────────────────────────
    "front": {
        "topic": "/camera_24264461/image_raw",
        "W": 1920, "H": 1200,
        "fx": 1451.7319, "fy": 1456.5062,
        "cx": 944.6727,  "cy": 597.5563,
        "d0": -0.1540,   "d1":  0.0982,
        "R_cl": np.array([
            [ 0.0081, -0.0138,  0.9999],
            [-0.9998, -0.0178,  0.0078],
            [ 0.0177, -0.9997, -0.0139]], dtype=np.float64),
        "t_cl": np.array([-0.2087,  0.0514, -0.3421], dtype=np.float64),
    },
    # ── FRONT-LEFT  /camera_24135744 ────────────────────────────────────────
    "front_left": {
        "topic": "/camera_24135744/image_raw",
        "W": 1920, "H": 1200,
        "fx": 1455.7474, "fy": 1460.4588,
        "cx": 952.9218,  "cy": 586.3597,
        "d0": -0.1639,   "d1":  0.1161,
        "R_cl": np.array([
            [ 0.8467,  0.0143,  0.5319],
            [-0.5320,  0.0312,  0.8461],
            [-0.0045, -0.9994,  0.0341]], dtype=np.float64),
        "t_cl": np.array([-0.1485, -0.0818, -0.2948], dtype=np.float64),
    },
    # ── BACK  /camera_24264465 ──────────────────────────────────────────────
    "back": {
        "topic": "/camera_24264465/image_raw",
        "W": 1920, "H": 1200,
        "fx": 1439.3085, "fy": 1440.9965,
        "cx": 937.6015,  "cy": 593.4765,
        "d0": -0.1577,   "d1":  0.1055,
        "R_cl": np.array([
            [-0.0138,  0.0115, -0.9998],
            [-0.9998, -0.0152,  0.0136],
            [-0.0150,  0.9998,  0.0117]], dtype=np.float64),
        "t_cl": np.array([ 0.2098, -0.0604,  0.3382], dtype=np.float64),
    },
    # ── BACK-LEFT  /camera_24222613 ─────────────────────────────────────────
    "back_left": {
        "topic": "/camera_24222613/image_raw",
        "W": 1920, "H": 1200,
        "fx": 1339.8000, "fy": 1342.5000,
        "cx": 945.8166,  "cy": 612.2553,
        "d0": -0.1672,   "d1":  0.0758,
        "R_cl": np.array([
            [ 0.8340, -0.0876, -0.5448],
            [ 0.5482,  0.0180,  0.8362],
            [-0.0635, -0.9960,  0.0631]], dtype=np.float64),
        "t_cl": np.array([-0.5050,  0.7186, -0.4939], dtype=np.float64),
    },
    # ── BACK-RIGHT  /camera_24264477 ────────────────────────────────────────
    "back_right": {
        "topic": "/camera_24264477/image_raw",
        "W": 1920, "H": 1200,
        "fx": 1432.9617, "fy": 1433.7948,
        "cx": 956.1567,  "cy": 581.9657,
        "d0": -0.1642,   "d1":  0.1050,
        "R_cl": np.array([
            [-0.8556, -0.0361, -0.5164],
            [ 0.5152,  0.0382, -0.8562],
            [ 0.0506, -0.9986, -0.0141]], dtype=np.float64),
        "t_cl": np.array([-0.1138, -0.1270,  0.1284], dtype=np.float64),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CORE MATH — Pinhole + Radial Distortion Projection
# ─────────────────────────────────────────────────────────────────────────────

def project_points(P_lidar, cam):
    """
    Project a batch (N,3) of LiDAR-frame points into a camera.

    Math:
      P_cam = R_cl @ P_lidar + t_cl       # rigid transform
      x_n   = P_cam.x / P_cam.z           # normalised x
      y_n   = P_cam.y / P_cam.z           # normalised y
      r2    = x_n^2 + y_n^2
      k     = 1 + d0*r2 + d1*r2^2         # radial distortion factor
      u     = fx * x_n * k + cx
      v     = fy * y_n * k + cy

    Returns:
      uvs   (N,2) float — pixel coords (may be out-of-bounds)
      valid (N,)  bool  — in front of camera AND inside image
      depth (N,)  float — Z in camera frame (metres)
    """
    P_cam = (cam["R_cl"] @ P_lidar.T).T + cam["t_cl"]   # (N,3)
    depth = P_cam[:, 2]
    safe  = np.where(depth > 0.3, depth, 1.0)

    x_n = P_cam[:, 0] / safe
    y_n = P_cam[:, 1] / safe
    r2  = x_n**2 + y_n**2
    k   = 1.0 + cam["d0"]*r2 + cam["d1"]*r2**2
    u   = cam["fx"] * x_n * k + cam["cx"]
    v   = cam["fy"] * y_n * k + cam["cy"]

    valid = ((depth > 0.3) &
             (u >= 0) & (u < cam["W"]-1) &
             (v >= 0) & (v < cam["H"]-1))
    return np.stack([u, v], axis=1), valid, depth


def bilinear_rgb(img_bgr, u, v):
    """Bilinear sample at sub-pixel (u,v); returns RGB in [0,1]."""
    x0, y0 = int(u), int(v)
    x1 = min(x0+1, img_bgr.shape[1]-1)
    y1 = min(y0+1, img_bgr.shape[0]-1)
    dx, dy = u - x0, v - y0
    bgr = (img_bgr[y0, x0]*(1-dx)*(1-dy) + img_bgr[y0, x1]*dx*(1-dy) +
           img_bgr[y1, x0]*(1-dx)*dy      + img_bgr[y1, x1]*dx*dy)
    return np.array([bgr[2], bgr[1], bgr[0]], dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY LOADER (Supports both TUM format and R3LIVE lic_lio.log)
# ─────────────────────────────────────────────────────────────────────────────

def load_trajectory(path, first_imu_time=0.0):
    from scipy.spatial.transform import Rotation as Rot
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = list(map(float, line.split()))
            if len(v) < 7:
                continue
            # Check format:
            # If line is TUM (8 elements: timestamp, tx, ty, tz, qx, qy, qz, qw):
            if len(v) == 8:
                t = v[0]
                tx, ty, tz = v[1:4]
                R = Rot.from_quat(v[4:8]).as_matrix()
            elif len(v) >= 25:
                # lic_lio.log format:
                # [0]: relative time from first_imu_time
                # [1-3]: rot_angle (so3 vector)
                # [4-6]: pos_end (x, y, z)
                t_rel = v[0]
                t = t_rel + first_imu_time
                so3 = np.array(v[1:4])
                tx, ty, tz = v[4:7]
                theta = np.linalg.norm(so3)
                if theta < 1e-12:
                    R = np.eye(3)
                else:
                    R = Rot.from_rotvec(so3).as_matrix()
            else:
                continue

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3,  3] = [tx, ty, tz]
            poses.append((t, T))
    print(f"[TRAJ] {len(poses)} poses loaded from {path}")
    return poses


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_from_bag(bag_path, every_n=3):
    """Extract images from ROS bag for all 5 camera topics, plus first_imu_time."""
    try:
        import rosbag
        from cv_bridge import CvBridge
    except ImportError:
        print("[ERROR] rosbag unavailable.  Run:  source /opt/ros/noetic/setup.bash")
        sys.exit(1)

    bridge  = CvBridge()
    topic2n = {c["topic"]: n for n, c in CAMERAS.items()}
    frames  = {n: [] for n in CAMERAS}
    counts  = {n: 0  for n in CAMERAS}
    first_imu_time = 0.0

    print(f"[BAG] Reading {bag_path} (every_n={every_n})")
    with rosbag.Bag(bag_path) as bag:
        # Check first IMU timestamp for timestamp alignment
        for _, msg, _ in bag.read_messages(topics=["/imu/data_raw", "/livox/imu"]):
            first_imu_time = msg.header.stamp.to_sec()
            print(f"[BAG] First IMU timestamp: {first_imu_time:.6f}")
            break

        for topic, msg, _ in tqdm(bag.read_messages(topics=list(topic2n)),
                                   desc="Extracting images", unit="msg"):
            name = topic2n.get(topic)
            if not name:
                continue
            counts[name] += 1
            if counts[name] % every_n:
                continue
            try:
                img = bridge.imgmsg_to_cv2(msg, "bgr8")
                frames[name].append((msg.header.stamp.to_sec(), img))
            except Exception:
                pass

    for n, fl in frames.items():
        print(f"  {n}: {len(fl)} frames")
    return frames, first_imu_time


def load_from_folder(folder, cam_name="front"):
    """Fallback: images in a folder named  <timestamp>.jpg / .png"""
    import glob
    frames = {cam_name: []}
    for p in sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                    glob.glob(os.path.join(folder, "*.png"))):
        try:    ts = float(os.path.splitext(os.path.basename(p))[0])
        except: ts = 0.0
        img = cv2.imread(p)
        if img is not None:
            frames[cam_name].append((ts, img))
    print(f"[DIR] {cam_name}: {len(frames[cam_name])} images")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COLORIZER  — Method A  (distance-weighted projection)
# ─────────────────────────────────────────────────────────────────────────────

def colorize(pcd_path, poses, frames, out_path, kf_step=5, max_r=40.0):
    """
    Core colorization loop.

    For each keyframe pose:
      1. Find all map points within max_r metres (KD-tree ball query)
      2. Transform them into the LiDAR frame at that keyframe
      3. Project onto each of the 6 cameras
      4. Sample bilinear RGB from the closest image in time
      5. Accumulate weighted color:  w = 1 / (distance^2 + eps)
         → closer camera views have much higher influence than far views

    Points never seen by any camera → default grey (0.45, 0.45, 0.45)
    """
    print(f"\n[LOAD] {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float64)   # (N,3)
    N   = len(pts)
    print(f"[INFO] Map: {N:,} points")

    C_rgb = np.zeros((N, 3), dtype=np.float64)   # weighted RGB accumulator
    C_w   = np.zeros(N,      dtype=np.float64)   # weight accumulator

    print("[INFO] Building KD-tree…")
    tree = KDTree(pts)

    # Cache timestamp arrays per camera for fast nearest-frame lookup
    ts_lut = {n: np.array([f[0] for f in fl])
              for n, fl in frames.items() if fl}

    keyframes = poses[::kf_step]
    print(f"[INFO] {len(keyframes)} keyframes  (step={kf_step})")

    for kf_ts, T_wl in tqdm(keyframes, desc="Colorizing", unit="kf"):
        t_wl = T_wl[:3, 3]
        R_wl = T_wl[:3, :3]

        # Map points within radius of this vehicle pose
        idxs = np.array(tree.query_ball_point(t_wl, r=max_r), dtype=np.int64)
        if len(idxs) == 0:
            continue

        # World → LiDAR frame:  P_l = R_wl^T (P_w - t_wl)
        P_l  = (R_wl.T @ (pts[idxs] - t_wl).T).T   # (M,3)
        dist = np.linalg.norm(P_l, axis=1)           # (M,)

        for cam_name, cam in CAMERAS.items():
            ts_arr = ts_lut.get(cam_name)
            if ts_arr is None or len(ts_arr) == 0:
                continue

            # Nearest image in time
            i_img = int(np.argmin(np.abs(ts_arr - kf_ts)))
            img   = frames[cam_name][i_img][1]       # BGR numpy

            uvs, valid, _ = project_points(P_l, cam)
            if not valid.any():
                continue

            vi = np.where(valid)[0]          # local indices into P_l
            gi = idxs[vi]                   # global indices into pts
            w  = 1.0 / (dist[vi]**2 + 0.01) # distance^2 weight

            for j, (li, gidx) in enumerate(zip(vi, gi)):
                u  = float(uvs[li, 0])
                v  = float(uvs[li, 1])
                rgb = bilinear_rgb(img, u, v)
                wt  = float(w[j])
                C_rgb[gidx] += wt * rgb
                C_w[gidx]   += wt

    # ── Normalise ────────────────────────────────────────────────────────────
    final = np.full((N, 3), 0.45, dtype=np.float32)   # grey for unseen points
    seen  = C_w > 0
    final[seen] = (C_rgb[seen] / C_w[seen, None]).astype(np.float32)

    pct = 100.0 * seen.sum() / N
    print(f"\n[DONE] Colored {seen.sum():,} / {N:,} points  ({pct:.1f}%)")
    if pct < 50:
        print("[TIP]  Low coverage → try:  --kf_step 2   --max_dist 60")

    pcd.colors = o3d.utility.Vector3dVector(final.astype(np.float64))
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"[SAVE] {out_path}")
    return pcd


# ─────────────────────────────────────────────────────────────────────────────
# COLMAP EXPORT  — needed for Method C (3DGS training)
# ─────────────────────────────────────────────────────────────────────────────

def export_colmap(poses, frames, map_pts, out_dir):
    """
    Exports a COLMAP-format sparse reconstruction from:
      - FAST-LIO vehicle poses
      - 6 camera images + calibration
      - LiDAR map points (used as sparse 3D point seed)

    After running this:
      git clone https://github.com/graphdeco-inria/gaussian-splatting
      cd gaussian-splatting
      pip install submodules/diff-gaussian-rasterization submodules/simple-knn
      python train.py -s <out_dir>/ --model_path ./gs_out/ --iterations 30000
    """
    from scipy.spatial.transform import Rotation as Rot

    sp  = os.path.join(out_dir, "sparse", "0")
    imd = os.path.join(out_dir, "images")
    os.makedirs(sp, exist_ok=True)
    os.makedirs(imd, exist_ok=True)

    # cameras.txt
    with open(os.path.join(sp, "cameras.txt"), "w") as f:
        f.write("# CAMERA_ID MODEL W H fx fy cx cy k1 k2\n")
        for cid, (cname, c) in enumerate(CAMERAS.items(), 1):
            f.write(f"{cid} OPENCV {c['W']} {c['H']} "
                    f"{c['fx']} {c['fy']} {c['cx']} {c['cy']} "
                    f"{c['d0']} {c['d1']} 0.0 0.0\n")

    # Pre-build timestamp lookups for each camera
    ts_lut = {n: np.array([f[0] for f in fl]) for n, fl in frames.items() if fl}

    # images.txt — one entry per (keyframe × camera)
    with open(os.path.join(sp, "images.txt"), "w") as f:
        img_id = 1
        for ki, (kf_ts, T_wl) in enumerate(poses[::5]):
            R_wl = T_wl[:3, :3];  t_wl = T_wl[:3, 3]
            for cid, (cname, c) in enumerate(CAMERAS.items(), 1):
                ts_arr = ts_lut.get(cname)
                if ts_arr is None or len(ts_arr) == 0:
                    continue

                # Find exact closest image in time
                i_img = int(np.argmin(np.abs(ts_arr - kf_ts)))
                if abs(ts_arr[i_img] - kf_ts) > 0.5:  # skip if image is too far from pose
                    continue
                img = frames[cname][i_img][1]

                # Save the image
                cd = os.path.join(imd, cname)
                os.makedirs(cd, exist_ok=True)
                img_rel_name = f"{cname}/kf{ki:06d}.jpg"
                cv2.imwrite(os.path.join(imd, img_rel_name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                R_cw = c["R_cl"] @ R_wl.T
                t_cw = -R_cw @ t_wl + c["t_cl"]
                q    = Rot.from_matrix(R_cw).as_quat()   # qx qy qz qw
                f.write(f"{img_id} {q[3]:.8f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} "
                        f"{t_cw[0]:.6f} {t_cw[1]:.6f} {t_cw[2]:.6f} {cid} {img_rel_name}\n\n")
                img_id += 1

    # points3D.txt — LiDAR points as sparse init (10:1 subsample for speed)
    with open(os.path.join(sp, "points3D.txt"), "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for i, pt in enumerate(map_pts[::10]):
            f.write(f"{i} {pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} "
                    f"128 128 128 0.5\n")

    print(f"[COLMAP] Done → {out_dir}/")
    print(f"[COLMAP] Next: python train.py -s {out_dir}/ --model_path ./gs_out/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Colorize FAST-LIO PCD map using 6 calibrated cameras")
    p.add_argument("--pcd",    required=True,  help="Input FAST-LIO .pcd")
    p.add_argument("--traj",   required=True,  help="lic_lio.log (TUM format)")
    p.add_argument("--out",    required=True,  help="Output colored .pcd")
    p.add_argument("--bag",    default=None,   help="ROS bag with camera topics")
    p.add_argument("--imgdir", default=None,   help="Image folder (single cam)")
    p.add_argument("--every_n",  type=int,   default=3,    help="Bag: every N-th frame")
    p.add_argument("--kf_step",  type=int,   default=5,    help="Every N-th pose")
    p.add_argument("--max_dist", type=float, default=40.0, help="Search radius (m)")
    p.add_argument("--export_colmap", action="store_true",
                   help="Export COLMAP format for 3DGS (Method C)")
    p.add_argument("--colmap_out", default="./colmap_input/")
    a = p.parse_args()

    t0 = time.time()

    if a.bag:
        frames, first_imu_time = load_from_bag(a.bag, every_n=a.every_n)
    elif a.imgdir:
        frames = load_from_folder(a.imgdir)
        first_imu_time = 0.0
    else:
        print("[ERROR] Provide --bag or --imgdir"); sys.exit(1)

    poses = load_trajectory(a.traj, first_imu_time=first_imu_time)
    if not poses:
        print("[ERROR] No poses found in trajectory file"); sys.exit(1)

    total_imgs = sum(len(v) for v in frames.values())
    if total_imgs == 0:
        print("[ERROR] No images loaded — check bag topics or folder path"); sys.exit(1)

    pcd_o   = o3d.io.read_point_cloud(a.pcd)
    map_pts = np.asarray(pcd_o.points, dtype=np.float64)

    if a.export_colmap:
        export_colmap(poses, frames, map_pts, a.colmap_out)
    else:
        colorize(a.pcd, poses, frames, a.out,
                 kf_step=a.kf_step, max_r=a.max_dist)

    print(f"\n[TOTAL] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()


# =============================================================================
#  METHOD B  —  SuperPoint Confidence Weighting  (explanation + drop-in code)
# =============================================================================
#
# WHAT SUPERPOINT GIVES YOU:
#   SuperPoint is a neural net that outputs a HEATMAP — every pixel gets a
#   score 0.0 (flat / texture-less) → 1.0 (strong corner or edge).
#
#   When you project a ghost-wall point (from the drift copy) onto a camera
#   frame, it usually lands on a region that looks wrong — misaligned texture,
#   wrong depth. The SuperPoint heatmap score at that pixel is LOW.
#   The real wall, seen head-on, lands on a crisp edge → HIGH score.
#   Multiplying weight by SP score → ghost gets near-zero weight, real wins.
#
# HOW TO ADD IT (drop-in replacement for the weight line in colorize()):
# ─────────────────────────────────────────────────────────────────────────────
#   # 1. Install:
#   #    pip install torch torchvision
#   #    git clone https://github.com/magicleap/SuperGluePretrainedNetwork
#   #    export PYTHONPATH=$PYTHONPATH:./SuperGluePretrainedNetwork
#   #
#   # 2. Import at top of this file:
#   import torch
#   from models.superpoint import SuperPoint as SP
#   sp_model = SP({'nms_radius':4, 'keypoint_threshold':0.005,
#                  'max_keypoints':-1}).eval()
#   if torch.cuda.is_available():
#       sp_model = sp_model.cuda()
#
#   # 3. Cache heatmap per (keyframe, camera) — call this once per image:
#   def sp_heatmap(img_bgr, model):
#       gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
#       t = torch.tensor(gray)[None, None]
#       if next(model.parameters()).is_cuda: t = t.cuda()
#       with torch.no_grad():
#           out = model({'image': t})
#       return out['scores'][0].cpu().numpy()   # shape (H, W)
#
#   # 4. In the colorize() inner loop, replace:
#   #      w = 1.0 / (dist[vi]**2 + 0.01)
#   #    with:
#   heat = sp_heatmap(img, sp_model)           # (H,W) float
#   for j, (li, gidx) in enumerate(zip(vi, gi)):
#       u_, v_ = float(uvs[li,0]), float(uvs[li,1])
#       sp_conf = float(heat[min(int(v_), heat.shape[0]-1),
#                            min(int(u_), heat.shape[1]-1)])
#       wt = sp_conf / (float(dist[li]) + 0.1)    # ML confidence / distance
#       ...
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
#  METHOD C  —  3D Gaussian Splatting  (explanation + full setup)
# =============================================================================
#
# CONCEPT:
#   3DGS represents the scene as N×(position, orientation, scale, opacity,
#   colour-SH-coefficients) 3D Gaussian blobs.
#   During training it renders the scene from your keyframe camera poses and
#   minimises photometric loss vs the real images — learning colour AND opacity.
#
# WHY IT SOLVES DOUBLE WALLS:
#   The optimizer sees both the "real" wall and the "ghost" wall from many
#   camera angles. The ghost wall contradicts the images in at least some views
#   (it appears in front of things that should be behind it).
#   3DGS sets its opacity → 0 to minimise loss.
#   After training:
#     real wall Gaussians:  opacity ≈ 1.0  ← keep
#     ghost wall Gaussians: opacity ≈ 0.0  ← flag → remove from map
#
#   This is the ONLY method that automatically detects ghost geometry.
#
# FULL SETUP:
# ─────────────────────────────────────────────────────────────────────────────
#   # Step 1 — export COLMAP format (uses this script's --export_colmap flag):
#   python3 colorize_fastlio_map.py \
#       --pcd  ~/r3live_output/lio_map_xyzi.pcd \
#       --bag  /path/to/drive.bag \
#       --traj ~/r3live_output/lic_lio.log \
#       --export_colmap --colmap_out ./colmap_in/ --out /tmp/x.pcd
#
#   # Step 2 — train 3DGS:
#   git clone https://github.com/graphdeco-inria/gaussian-splatting
#   cd gaussian-splatting
#   pip install submodules/diff-gaussian-rasterization submodules/simple-knn
#   python train.py -s ../colmap_in/ --model_path ./gs_out/ --iterations 30000
#   # GPU time: ~2h on RTX 3090 for a 700m route with 6 cameras
#
#   # Step 3 — extract ghost-wall mask from learned opacity:
#   python3 - <<'EOF'
#   import torch, numpy as np, open3d as o3d
#   import sys; sys.path.insert(0, './gaussian-splatting')
#   from scene.gaussian_model import GaussianModel
#
#   g = GaussianModel(3)
#   g.load_ply('./gs_out/point_cloud/iteration_30000/point_cloud.ply')
#
#   xyz     = g._xyz.detach().cpu().numpy()
#   opacity = torch.sigmoid(g._opacity).squeeze().detach().cpu().numpy()
#
#   real_mask  = opacity > 0.3   # keep solid Gaussians
#   ghost_mask = opacity < 0.05  # transparent = ghost / dynamic
#
#   print(f"Real: {real_mask.sum()}   Ghost/dynamic: {ghost_mask.sum()}")
#
#   # Save opacity-filtered coloured PCD
#   pcd = o3d.geometry.PointCloud()
#   pcd.points = o3d.utility.Vector3dVector(xyz[real_mask])
#   o3d.io.write_point_cloud("gs_real_only.pcd", pcd)
#   EOF
#
#   # Step 4 — colorize the FAST-LIO map from rendered 3DGS views:
#   # For each map tile, render the 3DGS scene from the nearest keyframe
#   # camera pose, then project map points onto the rendered image and
#   # sample RGB — same project_points() math as Method A but colour comes
#   # from a photorealistic render, not the raw camera image.
# ─────────────────────────────────────────────────────────────────────────────

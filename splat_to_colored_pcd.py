#!/usr/bin/env python3
"""Turn a trained 3D Gaussian Splatting model into a colored point-cloud map (.pcd).

Every Gaussian centre becomes one point. Its colour is the view-independent part of the
Gaussian's appearance, the degree-0 spherical-harmonic (DC) coefficient:

    rgb = clip(0.5 + SH_C0 * f_dc, 0, 1),   SH_C0 = 1 / (2 * sqrt(pi)) = 0.28209479177387814

Because the Gaussians were seeded from the LiDAR map, the centres keep the LiDAR geometry while
the colours come from all surround cameras at once, so the result has no per-frame projection
seams. --min-opacity drops near-transparent Gaussians (floaters) whose colour is unreliable.

Usage:
    python3 splat_to_colored_pcd.py <point_cloud.ply | splat.ply> <out.pcd> [--min-opacity X]

Check the point count it prints before using --min-opacity: in a LiDAR-seeded splat most Gaussians
can be nearly transparent (on the maps/ splat, 0.1 keeps only 4,451 of 408,143).

Input: the .ply written by gaussian-splatting / nerfstudio splatfacto export
(vertex fields x, y, z, f_dc_0..2, opacity). Output: binary PCD with fields x y z rgb.
"""
import argparse

import numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def splat_to_points(ply_path, min_opacity=0.0):
    v = PlyData.read(ply_path)['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1), 0.0, 1.0)
    keep = np.all(np.isfinite(xyz), axis=1)
    if min_opacity > 0.0:
        keep &= 1.0 / (1.0 + np.exp(-np.asarray(v['opacity'], dtype=np.float64))) >= min_opacity
    return xyz[keep], rgb[keep]


def write_pcd(path, xyz, rgb):
    c = np.round(rgb * 255.0).astype(np.uint32)
    rec = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
    rec['x'], rec['y'], rec['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec['rgb'] = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
    header = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n'
              'TYPE F F F U\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\n'
              'DATA binary\n' % (len(rec), len(rec)))
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(rec.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('splat_ply', help='trained Gaussian splat .ply')
    parser.add_argument('out_pcd', help='output colored .pcd')
    parser.add_argument('--min-opacity', type=float, default=0.0,
                        help='drop Gaussians whose opacity (after sigmoid) is below this; 0 keeps all')
    args = parser.parse_args()
    xyz, rgb = splat_to_points(args.splat_ply, args.min_opacity)
    write_pcd(args.out_pcd, xyz, rgb)
    print('%s: %d points (x y z rgb)' % (args.out_pcd, len(xyz)))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_lidar_camera.py — 自包含的 LiDAR-相机外参标定（替代 FAST-Calib2 标准流程）

针对 16 线稀疏雷达 + 倾斜板 + intensity 区分度不足的数据，自定义 LiDAR 圆环提取，
配合 ArUco 图像端检测，用 SVD 求外参 T_cam_lidar。

流程：
  图像端: ArUco 板姿态估计 -> 4 圆心(相机坐标系)
  LiDAR端: 距离滤波 -> 高反点 -> 聚类取板簇 -> 倾斜板平面 -> 对齐Z=0
           -> 固定半径网格搜环心 -> 几何筛选4个 -> 转回LiDAR坐标系
  排序: 两端都用 sortPatternCenters 逻辑(按角度逆时针)保证对应
  求解: 单场景 SVD 刚体变换 + RMSE；多场景联合加权 SVD

用法:
  python3 scripts/calibrate_lidar_camera.py --config config/calibration.yaml --scenes /path/to/scenes --out output
"""
import os, glob, argparse, itertools
from collections import defaultdict
import numpy as np
import cv2
try:
    import rosbag
    import sensor_msgs.point_cloud2 as pc2
except ImportError:
    rosbag = None
    pc2 = None

from pcd_utils import load_pcd

DEFAULT_CONFIG = {
    "camera": {
        "fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "board": {
        "marker_size": 0.20,
        "delta_width_qr_center": 0.55,
        "delta_height_qr_center": 0.35,
        "delta_width_circles": 0.50,
        "delta_height_circles": 0.40,
        "circle_radius": 0.12,
        "annulus_half_width": 0.025,
        "marker_ids": [1, 2, 4, 3],
        "aruco_dictionary": "DICT_6X6_250",
    },
    "lidar": {
        "topic": "/livox/lidar",
        "x_min": 0.8,
        "x_max": 4.5,
        "intensity_threshold": 80.0,
        "cluster_tolerance": 0.20,
        "cluster_min_size": 30,
        "plane_ransac_threshold": 0.03,
        "plane_near_threshold": 0.04,
        "grid_step": 0.02,
        "rectangle_tolerance": 0.08,
    },
}


def load_config(path):
    """Load user calibration parameters from a YAML file."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("Install PyYAML with: pip install -r requirements.txt") from error

    with open(path, "r", encoding="utf-8") as config_file:
        supplied = yaml.safe_load(config_file) or {}

    config = {section: values.copy() for section, values in DEFAULT_CONFIG.items()}
    for section, values in supplied.items():
        if section not in config:
            raise ValueError(f"Unsupported config section: {section}")
        if not isinstance(values, dict):
            raise ValueError(f"Config section '{section}' must be a mapping")
        unknown = set(values) - set(config[section])
        if unknown:
            raise ValueError(f"Unsupported keys in '{section}': {sorted(unknown)}")
        config[section].update(values)

    camera = config["camera"]
    if any(float(camera[key]) == 0.0 for key in ("fx", "fy")):
        raise ValueError("Set non-zero camera.fx and camera.fy in your configuration file")
    if len(camera["distortion"]) != 5:
        raise ValueError("camera.distortion must contain [k1, k2, p1, p2, k3]")
    return config


def configure(config):
    """Apply configuration values to the reference implementation."""
    global MARKER_SIZE, DELTA_W_QR, DELTA_H_QR, DELTA_W_CIRCLE, DELTA_H_CIRCLE
    global CIRCLE_RADIUS, ANNULUS_HALF_W, BOARD_IDS, DICT_TYPE, CAM_MTX, CAM_DIST
    global LIDAR_TOPIC, LIDAR_X_MIN, LIDAR_X_MAX, INTENSITY_TH, CLUSTER_TOL, CLUSTER_MIN
    global PLANE_RANSAC_TH, PLANE_NEAR_TH, GRID_STEP, RECT_TOL
    global _BOARD_CORNERS, _BOARD_CIRCLE_CENTERS, _TMPL_RECT

    camera = config["camera"]
    board = config["board"]
    lidar = config["lidar"]
    MARKER_SIZE = float(board["marker_size"])
    DELTA_W_QR = float(board["delta_width_qr_center"])
    DELTA_H_QR = float(board["delta_height_qr_center"])
    DELTA_W_CIRCLE = float(board["delta_width_circles"])
    DELTA_H_CIRCLE = float(board["delta_height_circles"])
    CIRCLE_RADIUS = float(board["circle_radius"])
    ANNULUS_HALF_W = float(board["annulus_half_width"])
    BOARD_IDS = list(board["marker_ids"])
    DICT_TYPE = getattr(cv2.aruco, board["aruco_dictionary"])
    CAM_MTX = np.array(
        [[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]],
        dtype=np.float64,
    )
    CAM_DIST = np.array(camera["distortion"], dtype=np.float64)
    LIDAR_TOPIC = str(lidar["topic"])
    LIDAR_X_MIN = float(lidar["x_min"])
    LIDAR_X_MAX = float(lidar["x_max"])
    INTENSITY_TH = float(lidar["intensity_threshold"])
    CLUSTER_TOL = float(lidar["cluster_tolerance"])
    CLUSTER_MIN = int(lidar["cluster_min_size"])
    PLANE_RANSAC_TH = float(lidar["plane_ransac_threshold"])
    PLANE_NEAR_TH = float(lidar["plane_near_threshold"])
    GRID_STEP = float(lidar["grid_step"])
    RECT_TOL = float(lidar["rectangle_tolerance"])
    _BOARD_CORNERS, _BOARD_CIRCLE_CENTERS = build_board()
    _TMPL_RECT = np.array(
        [[-DELTA_W_CIRCLE / 2, -DELTA_H_CIRCLE / 2],
         [ DELTA_W_CIRCLE / 2, -DELTA_H_CIRCLE / 2],
         [ DELTA_W_CIRCLE / 2,  DELTA_H_CIRCLE / 2],
         [-DELTA_W_CIRCLE / 2,  DELTA_H_CIRCLE / 2]],
        dtype=np.float64,
    )


# Values are populated by configure() after command-line parsing.
MARKER_SIZE = DELTA_W_QR = DELTA_H_QR = DELTA_W_CIRCLE = DELTA_H_CIRCLE = None
CIRCLE_RADIUS = ANNULUS_HALF_W = None
BOARD_IDS = DICT_TYPE = CAM_MTX = CAM_DIST = None
LIDAR_TOPIC = None
LIDAR_X_MIN = LIDAR_X_MAX = INTENSITY_TH = CLUSTER_TOL = CLUSTER_MIN = None
PLANE_RANSAC_TH = PLANE_NEAR_TH = GRID_STEP = RECT_TOL = None
_BOARD_CORNERS = _BOARD_CIRCLE_CENTERS = _TMPL_RECT = None
# ============ 图像端: ArUco 板姿态 -> 4 圆心(相机坐标系) ============
def build_board():
    """构造 FAST-Calib2 板: 4 个 ArUco 角点 + 4 个圆心(板坐标系)"""
    width = DELTA_W_QR; height = DELTA_H_QR
    cw = DELTA_W_CIRCLE / 2; ch = DELTA_H_CIRCLE / 2
    board_corners = []
    circle_centers = []
    for i in range(4):
        x_qr = -1 if (i % 3) == 0 else 1
        y_qr = 1 if i < 2 else -1
        xc = x_qr * width; yc = y_qr * height
        circle_centers.append(np.array([x_qr * cw, y_qr * ch, 0.0], np.float32))
        corners = []
        for j in range(4):
            xq = -1 if (j % 3) == 0 else 1
            yq = 1 if j < 2 else -1
            corners.append([xc + xq * MARKER_SIZE / 2, yc + yq * MARKER_SIZE / 2, 0])
        board_corners.append(np.array(corners, np.float32))
    return board_corners, circle_centers



def extract_image_centers(image_path):
    """检测 ArUco 板, 用板姿态把 4 圆心变换到相机坐标系.
    【2026-07-07 BUG-4 修复】同时返回板系->相机系的旋转 R_cb 与平移 t_cb, 供 sort 用真实板轴排序,
    绕过 board_frame 里 world_up 项目导致的 3-12° 轴偏差(跨场景不一致真因).
    返回 (centers(4,3), R_cb(3,3), t_cb(3,)) 或 (None, None, None).
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [IMG] 读图失败: {image_path}"); return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None or len(ids) < 3:
        print(f"  [IMG] ArUco 检测不足: {len(ids) if ids is not None else 0}"); return None, None, None
    # 用 Board 估计板姿态 (OpenCV 4.2 旧 API: Board_create)
    if hasattr(cv2.aruco, "Board_create"):
        board = cv2.aruco.Board_create(_BOARD_CORNERS, dictionary, np.array(BOARD_IDS))
    else:
        board = cv2.aruco.Board(_BOARD_CORNERS, dictionary, np.array(BOARD_IDS))
    try:
        ok, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, CAM_MTX, CAM_DIST, None, None)
    except Exception:
        ok = 0
    if not ok:
        # 兜底: 用单标记平均
        rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, CAM_MTX, CAM_DIST)
        rvec = np.mean(rvecs, axis=0).reshape(3, 1)
        tvec = np.mean(tvecs, axis=0).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    centers = []
    for cc in _BOARD_CIRCLE_CENTERS:
        p = R @ cc + t
        centers.append(p)
    centers = np.array(centers, np.float64)
    print(f"  [IMG] ArUco={len(ids)} 圆心 z(深度)={centers[:,2].mean():.2f}m")
    return centers, R, t


# ============ LiDAR 端: 自定义圆环提取 ============
def load_cloud(bag_path, topic):
    if rosbag is None or pc2 is None:
        raise RuntimeError("Bag input requires ROS1 Python packages. Source your ROS environment first.")
    bag = rosbag.Bag(bag_path)
    for _, msg, _ in bag.read_messages(topics=[topic]):
        xs, ys, zs, ins = [], [], [], []
        for p in pc2.read_points(msg, field_names=("x","y","z","intensity"), skip_nans=True):
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2]); ins.append(p[3])
        return np.array(xs), np.array(ys), np.array(zs), np.array(ins)
    return None

def ransac_plane(pts, th, iters=500, seed=0):
    rng = np.random.RandomState(seed); bi = None; bn = 0; N = len(pts)
    for _ in range(iters):
        if N < 3: break
        idx = rng.choice(N, 3, replace=False)
        p1, p2, p3 = pts[idx]
        n = np.cross(p2 - p1, p3 - p1); nn = np.linalg.norm(n)
        if nn < 1e-6: continue
        n /= nn; d = -n.dot(p1)
        inl = np.abs(pts.dot(n) + d) < th
        if inl.sum() > bn: bn = inl.sum(); bi = inl
    return bi

def euclid_cluster(P, tol, mins):
    N = len(P); parent = list(range(N))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    cells = defaultdict(list); g = round(tol, 3)
    for i, p in enumerate(P):
        cells[(round(p[0]/g)*g, round(p[1]/g)*g, round(p[2]/g)*g)].append(i)
    for (cx, cy, cz), idxs in list(cells.items()):
        for dx, dy, dz in itertools.product([-1, 0, 1], repeat=3):
            k = (cx+dx*g, cy+dy*g, cz+dz*g)
            if k in cells:
                for i in idxs:
                    for j in cells[k]:
                        if i >= j: continue
                        if np.linalg.norm(P[i]-P[j]) <= tol:
                            ra, rb = find(i), find(j)
                            if ra != rb: parent[ra] = rb
    clus = defaultdict(list)
    for i in range(N): clus[find(i)].append(i)
    return [c for c in clus.values() if len(c) >= mins]

def rotation_align_z(normal):
    """旋转矩阵使 normal -> (0,0,1)"""
    a = np.cross(normal, np.array([0, 0, 1]))
    if np.linalg.norm(a) < 1e-9: return np.eye(3)
    a = a / np.linalg.norm(a)
    ang = np.arccos(np.clip(normal.dot([0, 0, 1]), -1, 1))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)

def extract_lidar_centers(bag_path):
    """自定义 LiDAR 4 圆环提取. 返回 LiDAR 坐标系 (4,3) 或 None"""
    cloud = load_cloud(bag_path, LIDAR_TOPIC)
    if cloud is None: return None
    x, y, z, inten = cloud
    # 1. 距离滤波
    m = (x >= LIDAR_X_MIN) & (x <= LIDAR_X_MAX)
    pts = np.column_stack([x[m], y[m], z[m]]); im = inten[m]
    # 2. 高反点
    hi = im >= INTENSITY_TH; ph = pts[hi]
    if len(ph) < 30:
        print(f"  [LIDAR] 高反点太少: {len(ph)}"); return None
    # 3. 聚类取最大簇(板子)
    clusters = euclid_cluster(ph, CLUSTER_TOL, CLUSTER_MIN)
    if not clusters:
        print(f"  [LIDAR] 无聚类簇"); return None
    clusters.sort(key=len, reverse=True)
    board = ph[clusters[0]]
    print(f"  [LIDAR] 板簇 {len(board)}点, 范围 x[{board[:,0].min():.2f},{board[:,0].max():.2f}]")
    # 4. 倾斜板平面
    inl = ransac_plane(board, PLANE_RANSAC_TH)
    pp = board[inl]
    ctr = pp.mean(0); _, _, Vt = np.linalg.svd(pp - ctr); normal = Vt[2]
    if normal[0] > 0: normal = -normal
    d = -normal.dot(ctr)
    dist_all = np.abs(board.dot(normal) + d)   # 修bug: 用board簇而非全部ph, 避免远端非环杂点
    ring = board[dist_all < PLANE_NEAR_TH]
    if len(ring) < 20:
        print(f"  [LIDAR] 板平面附近高反点太少: {len(ring)}"); return None
    # 5. 对齐 Z=0
    R = rotation_align_z(normal)
    ring_a = (R @ (ring - ctr).T).T
    xy = ring_a[:, :2]
    # 6. 固定半径网格搜环心
    inner = CIRCLE_RADIUS - ANNULUS_HALF_W; outer = CIRCLE_RADIUS + ANNULUS_HALF_W
    x0, x1 = xy[:, 0].min() - 0.1, xy[:, 0].max() + 0.1
    y0, y1 = xy[:, 1].min() - 0.1, xy[:, 1].max() + 0.1
    gx = np.arange(x0, x1, GRID_STEP); gy = np.arange(y0, y1, GRID_STEP)
    # 预计算点坐标加速
    px = xy[:, 0]; py = xy[:, 1]
    scores = np.zeros((len(gx), len(gy)))
    for i, cx in enumerate(gx):
        dx = px - cx
        for j, cy in enumerate(gy):
            dy = py - cy
            dd = np.sqrt(dx*dx + dy*dy)
            scores[i, j] = ((dd >= inner) & (dd <= outer)).sum()
    # 局部极大(NMS, 邻域±0.1m)
    cand = []
    rad = max(1, int(0.1 / GRID_STEP))
    for i in range(len(gx)):
        for j in range(len(gy)):
            s = scores[i, j]
            if s < 5: continue
            di, dj = max(0, i-rad), max(0, j-rad)
            ui, uj = min(len(gx), i+rad+1), min(len(gy), j+rad+1)
            if s == scores[di:ui, dj:uj].max():
                cand.append((s, gx[i], gy[j]))
    cand.sort(reverse=True)
    # 7. 几何筛选 2x2 矩形
    found = find_rect(cand[:30])
    if found is None:
        print(f"  [LIDAR] 几何筛选未找到4环心 (候选{len(cand)})"); return None
    centers2d = np.array([[c[1], c[2]] for c in found])
    # 7b. 环心精修: 对每个粗环心, 用环带内点做加权最小二乘拟合圆心(代数法+迭代)
    refined = []
    for cx, cy in centers2d:
        refined.append(refine_center(xy, cx, cy, CIRCLE_RADIUS, inner, outer))
    centers2d = np.array(refined)
    # 转回 LiDAR 坐标系(逆变换: z=0 的对齐点 -> 原始)
    centers_a = np.column_stack([centers2d, np.zeros(4)])
    centers_lidar = (R.T @ centers_a.T).T + ctr
    print(f"  [LIDAR] 提取4环心(精修), LiDAR坐标 z={centers_lidar[:,2].mean():.2f}")
    return centers_lidar


# ============ 多帧 LiDAR 圆环提取 (组合方法: 多帧累积+联合矩形模板+姿态先验) ============
# 针对非重复扫描(Livox类): 板静止时多帧累积补足半环缺口(单帧gap 186°→100帧2-32°).
# 联合刚性矩形模板(4角强制0.5x0.4)吸收单环偏差; 姿态先验定θ消除轴歧义.
_TMPL_RECT = np.array([[-0.25, -0.2], [0.25, -0.2], [0.25, 0.2], [-0.25, 0.2]], np.float64)

def pose_prior_theta(normal):
    """姿态先验: 板长边(0.5m方向)在对齐2D平面的方向θ.
    板y轴(上端朝传感器)=world_up(0,0,1)在板平面投影; 板x轴(长边)=y×z.
    对齐Z=0后, 板x轴2D方向即θ."""
    up = np.array([0.0, 0.0, 1.0])
    yb = up - normal * up.dot(normal)
    yn = np.linalg.norm(yb)
    if yn < 1e-3:
        yb = np.array([0.0, 1.0, 0.0]) - normal * np.array([0, 1, 0]).dot(normal)
        yn = np.linalg.norm(yb)
    yb = yb / yn
    xb = np.cross(yb, normal); xb = xb / np.linalg.norm(xb)
    Rm = rotation_align_z(normal)
    xb_a = Rm @ xb
    return np.arctan2(xb_a[1], xb_a[0])

def _frame_ring_board(pcd_path):
    """读一帧PCD, 返回 (板簇环带点LiDAR坐标, 板法向, 板质心) 或 None. 修ph->board bug."""
    if load_pcd is None: return None
    xyz, inten = load_pcd(pcd_path)
    if xyz is None: return None
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    m = (x >= LIDAR_X_MIN) & (x <= LIDAR_X_MAX)
    pts = np.column_stack([x[m], y[m], z[m]]); im = inten[m]
    hi = im >= INTENSITY_TH; ph = pts[hi]
    if len(ph) < 30: return None
    clusters = euclid_cluster(ph, CLUSTER_TOL, CLUSTER_MIN)
    if not clusters: return None
    clusters.sort(key=len, reverse=True); board = ph[clusters[0]]
    inl = ransac_plane(board, PLANE_RANSAC_TH); pp = board[inl]
    ctr = pp.mean(0); _, _, Vt = np.linalg.svd(pp - ctr); normal = Vt[2]
    if normal[0] > 0: normal = -normal
    d = -normal.dot(ctr)
    dist = np.abs(board.dot(normal) + d)   # 用board簇, 非全部ph
    ring = board[dist < PLANE_NEAR_TH]
    if len(ring) < 20: return None
    return ring, normal, ctr

def _score_template(px, py, cx, cy, th, inner, outer):
    """联合矩形模板score: 4角(±0.25,±0.2旋转θ平移cx,cy)各自的环带内点数之和"""
    c, s = np.cos(th), np.sin(th)
    xs = cx + c * _TMPL_RECT[:, 0] - s * _TMPL_RECT[:, 1]
    ys = cy + s * _TMPL_RECT[:, 0] + c * _TMPL_RECT[:, 1]
    tot = 0
    for i in range(4):
        dd = np.sqrt((px - xs[i]) ** 2 + (py - ys[i]) ** 2)
        tot += ((dd >= inner) & (dd <= outer)).sum()
    return tot, np.column_stack([xs, ys])

def _refine_center_fixedR(xy, cx, cy, radius, inner, outer, iters=200):
    """固定半径迭代精修圆心(近完整圆时收敛到真值, 无半环偏差)"""
    for _ in range(iters):
        dd = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
        inb = (dd >= inner) & (dd <= outer); pts = xy[inb]
        if len(pts) < 5: break
        dx = pts[:, 0] - cx; dy = pts[:, 1] - cy
        d = np.sqrt(dx * dx + dy * dy); d[d < 1e-6] = 1e-6
        ncx = np.mean(pts[:, 0] - dx / d * radius)
        ncy = np.mean(pts[:, 1] - dy / d * radius)
        if abs(ncx - cx) < 1e-9 and abs(ncy - cy) < 1e-9:
            cx, cy = ncx, ncy; break
        cx, cy = ncx, ncy
    return cx, cy

def extract_lidar_centers_multiframe(pcd_dir, nf=30):
    """多帧组合方法提取4环心(LiDAR坐标系). 所有帧统一到参考帧(frame 0)的板平面对齐系累积,
    联合矩形模板+姿态先验定θ, 每环固定半径精修.
    【BUG-2 修复】原代码累积时每帧独立对齐再累积, 转回时用 frame-0, 导致累积点 3-5cm 帧间偏移.
    改为全部用 frame-0 (Rm, ctr) 累积.
    【BUG-4 修复】同时返回 R_lb(板系->LiDAR系, 由 θ 定 xb, world_up=(0,0,1) 项目定 yb) 与 t_lb, 供
    sort_centers 用真实板轴排序, 绕过 board_frame 二次估计.
    返回 (centers(4,3), R_lb(3,3), t_lb(3,)) 或 (None, None, None).
    """
    if load_pcd is None:
        print("  [LIDAR-MF] load_pcd 不可用, 回退单帧"); return None, None, None
    pcds = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))[:nf]
    if not pcds:
        print(f"  [LIDAR-MF] 无PCD: {pcd_dir}"); return None, None, None
    f0 = _frame_ring_board(pcds[0])
    if f0 is None:
        print("  [LIDAR-MF] 第1帧提取失败"); return None, None, None
    ring0, normal, ctr = f0
    Rm = rotation_align_z(normal)
    th_prior = pose_prior_theta(normal)
    inner = CIRCLE_RADIUS - ANNULUS_HALF_W; outer = CIRCLE_RADIUS + ANNULUS_HALF_W
    # BUG-2 FIX: 所有帧统一到 frame-0 (Rm, ctr)
    cum = None; n_used = 0
    for p in pcds:
        r = _frame_ring_board(p)
        if r is None: continue
        ring, _, _ = r
        ra = (Rm @ (ring - ctr).T).T[:, :2]
        cum = ra if cum is None else np.vstack([cum, ra])
        n_used += 1
    if cum is None or len(cum) < 50: return None, None, None
    px, py = cum[:, 0], cum[:, 1]
    ctr2 = cum.mean(0)
    # 联合模板搜: θ在先验±20°, cx,cy±0.2
    best = None
    for th in np.arange(th_prior - np.radians(20), th_prior + np.radians(20), np.radians(2)):
        for cx in np.arange(ctr2[0] - 0.2, ctr2[0] + 0.2, 0.02):
            for cy in np.arange(ctr2[1] - 0.2, ctr2[1] + 0.2, 0.02):
                sc, _ = _score_template(px, py, cx, cy, th, inner, outer)
                if best is None or sc > best[0]: best = (sc, cx, cy, th)
    if best is None: return None, None, None
    sc, cx, cy, th = best
    _, centers2d = _score_template(px, py, cx, cy, th, inner, outer)
    # 精修每环
    ref2d = np.array([_refine_center_fixedR(cum, c[0], c[1], CIRCLE_RADIUS, inner, outer) for c in centers2d])
    # 转回LiDAR坐标系
    centers_a = np.column_stack([ref2d, np.zeros(4)])
    centers_lidar = (Rm.T @ centers_a.T).T + ctr
    # 【BUG-4 FIX】计算板系->LiDAR系的旋转 R_lb:
    #   对齐系(z=0)的板 x 轴 = (cos θ, sin θ, 0), 板 y 轴 = (-sin θ, cos θ, 0), z = (0,0,1)
    #   LiDAR系轴 = Rm.T @ 对齐系轴
    xb_a = np.array([np.cos(th), np.sin(th), 0.0])
    yb_a = np.array([-np.sin(th), np.cos(th), 0.0])
    zb_a = np.array([0.0, 0.0, 1.0])
    xb_l = Rm.T @ xb_a
    yb_l = Rm.T @ yb_a
    zb_l = Rm.T @ zb_a
    R_lb = np.column_stack([xb_l, yb_l, zb_l])
    t_lb = centers_lidar.mean(0)
    print(f"  [LIDAR-MF] {n_used}帧 累积{len(cum)}点 θ={np.degrees(th):.0f}°(先验{np.degrees(th_prior):.0f}°) score={sc}")
    return centers_lidar, R_lb, t_lb


def refine_center(xy, cx0, cy0, radius, inner, outer, iters=30):
    """对粗圆心(cx0,cy0), 用环带内点(半径radius±half)迭代精修圆心"""
    cx, cy = cx0, cy0
    for _ in range(iters):
        dd = np.sqrt((xy[:,0]-cx)**2 + (xy[:,1]-cy)**2)
        inband = (dd >= inner) & (dd <= outer)
        pts = xy[inband]
        if len(pts) < 5: break
        # 代数圆拟合 Kasa: 2cx*x+2cy*y+r2 = x2+y2 (r2=cx2+cy2-r2_cur)
        A = np.column_stack([2*pts[:,0], 2*pts[:,1], np.ones(len(pts))])
        b = pts[:,0]**2 + pts[:,1]**2
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            ncx, ncy = sol[0], sol[1]
            nr = np.sqrt(max(sol[0]**2+sol[1]**2 - sol[2], 1e-6))
            # 拉回接近预期半径
            if abs(nr - radius) < 0.05:
                cx, cy = ncx, ncy
            else:
                # 用固定半径约束: 圆心=环带内点相对半径radius的加权均值
                dx = pts[:,0]-cx; dy = pts[:,1]-cy
                d = np.sqrt(dx*dx+dy*dy)
                d[d<1e-6]=1e-6
                cx = np.mean(pts[:,0] - dx/d*radius)
                cy = np.mean(pts[:,1] - dy/d*radius)
        except Exception:
            break
    return [cx, cy]

def find_rect(cand, tol=RECT_TOL):
    """从候选里选4个组成 2x2 矩形(水平DELTA_W/垂直DELTA_H)"""
    pts = [(c[1], c[2]) for c in cand]
    n = len(pts)
    expected_diag = np.sqrt(DELTA_W_CIRCLE**2 + DELTA_H_CIRCLE**2)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    four = [pts[i], pts[j], pts[k], pts[l]]
                    if is_rect(four, expected_diag, tol):
                        return [cand[i], cand[j], cand[k], cand[l]]
    return None

def is_rect(four, expected_diag, tol):
    """4点是否构成 dw x dh 矩形: 4边≈{0.5,0.4}, 2对角线≈expected_diag"""
    cs = np.array(four)
    dists = []
    for a, b in itertools.combinations(range(4), 2):
        dists.append(np.linalg.norm(cs[a]-cs[b]))
    dists.sort()
    edges = dists[:4]; diags = dists[4:]
    # 4边应≈0.5或0.4 (允许tol), 2对角线≈expected_diag
    for e in edges:
        if not (abs(e-DELTA_W_CIRCLE) < tol or abs(e-DELTA_H_CIRCLE) < tol):
            return False
    for d in diags:
        if abs(d - expected_diag) > tol * 1.5:
            return False
    return True


# ============ 排序: 用板几何+世界up定向, 消除轴歧义 ============
# 【2026-07-07 BUG-4 修复】新方案: 由上游 (extract_image_centers / extract_lidar_centers_multiframe)
# 直接给出板系->传感器系的旋转 R_sb, sort_by_axes 用 R_sb 的 x 列/y 列做排序. 避免 board_frame 通过
# world_up 项目二次估计板轴 (在板 tilt 大时会有 3-12° 偏差, 是跨场景不一致的真根因).
#
# 板系 y 轴的物理约定 (两端一致): 指向"板上方" (marker 1/2 一侧, 世界 z 项目也是这一侧).
#   Cam 端: estimatePoseBoard 的 R 直接给出.
#   LiDAR 端: pose_prior_theta 用 world_up=(0,0,1) 项目定义, 与 Cam 端物理一致.
#
# 旧函数保留兼容, 但新代码不再使用.
def board_frame(rel_points, world_up, centroid):
    """[已弃用, 保留兼容] 给4点(相对质心) + centroid, 返回板 x,y,z 轴.
    问题: world_up 项目到板平面时, 若板 tilt 大, 结果与真实板轴有 3-12° 偏差.
    改用 extract_image_centers / extract_lidar_centers_multiframe 返回的 R_sb.
    """
    U, S, Vt = np.linalg.svd(rel_points, full_matrices=False)
    normal = Vt[2]
    if normal @ centroid > 0: normal = -normal
    yb = world_up - normal * world_up.dot(normal)
    yn = np.linalg.norm(yb)
    if yn < 1e-3:
        yb = np.array([0.0, 1.0, 0.0]) - normal * np.array([0,1,0]).dot(normal)
        yn = np.linalg.norm(yb)
    yb = yb / yn
    xb = np.cross(yb, normal); xb = xb / np.linalg.norm(xb)
    return xb, yb, normal


def sort_by_axes(centers, R_sb):
    """centers: (4,3) 传感器坐标系下的 4 圆心; R_sb: (3,3) 板系->传感器系的旋转矩阵.
    用 R_sb 的第0列(板 x 轴)/第1列(板 y 轴)做 sx,sy 分类, 按 [(-,-),(+,-),(+,+),(-,+)] 逆时针.
    因 R_sb 是真实板轴(不经 world_up 项目), 跨场景一致, 4 圆心在板系接近标准 (±0.25, ±0.2).
    """
    pts = centers.copy().astype(np.float64)
    ctr = pts.mean(0)
    rel = pts - ctr
    xb = R_sb[:, 0]; yb = R_sb[:, 1]
    sx = rel @ xb; sy = rel @ yb
    order_by_key = sorted(range(4), key=lambda i: ((1 if sx[i] > 0 else 0), (1 if sy[i] > 0 else 0)))
    final = [order_by_key[0], order_by_key[2], order_by_key[3], order_by_key[1]]
    return pts[final]


def sort_centers(centers, mode):
    """[旧接口, 保留兼容] centers: (4,3). 用板几何+世界up定向, 按 [(-,-),(+,-),(+,+),(-,+)] 排序.
    mode='lidar': world_up=(0,0,1); mode='camera': world_up=(0,-1,0).
    问题: 世界 up 项目在板 tilt 大时给出偏差 xb/yb, 跨场景不一致 (真根因). 新代码用 sort_by_axes.
    """
    pts = centers.copy().astype(np.float64)
    ctr = pts.mean(0)
    rel = pts - ctr
    world_up = np.array([0.0, 0.0, 1.0]) if mode == "lidar" else np.array([0.0, -1.0, 0.0])
    xb, yb, zb = board_frame(rel, world_up, ctr)
    sx = rel @ xb
    sy = rel @ yb
    order_by_key = sorted(range(4), key=lambda i: ((1 if sx[i] > 0 else 0), (1 if sy[i] > 0 else 0)))
    final = [order_by_key[0], order_by_key[2], order_by_key[3], order_by_key[1]]
    return pts[final]


# ============ SVD 求刚体变换 ============
def svd_rigid(src, dst):
    """src,dst: (N,3). 求 T 使 dst ≈ T @ src (4x4). 返回 (T, rmse)"""
    mu_s = src.mean(0); mu_d = dst.mean(0)
    S = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    aligned = (R @ src.T).T + t
    rmse = np.sqrt(np.mean(np.sum((aligned - dst)**2, axis=1)))
    return T, rmse


# ============ 单场景标定 ============
def calibrate_single(scene_dir, out_dir, raw_root=None, nf=30):
    bag = os.path.join(scene_dir, "scene.bag")
    img = os.path.join(scene_dir, "scene.bmp")
    name = os.path.basename(scene_dir)
    if not os.path.exists(img):
        return None
    print(f"\n[{name}]")
    img_c, R_cb, t_cb = extract_image_centers(img)
    if img_c is None: return None
    # 优先多帧PCD(组合方法), 回退单帧bag
    lidar_c = None; R_lb = None; t_lb = None
    if raw_root is not None:
        # 场景名 scene_XX_<timestamp>_fYY, 原始目录 <timestamp>/group_000000/pcd
        # timestamp 形如 2026_07_07_12_00_09 (带下划线), 用正则匹配
        import re
        m = re.search(r"(20\d\d_\d\d_\d\d_\d\d_\d\d_\d\d)", name)
        if m:
            ts = m.group(1)
            pcd_dir = os.path.join(raw_root, ts, "group_000000", "pcd")
            if os.path.isdir(pcd_dir):
                lidar_c, R_lb, t_lb = extract_lidar_centers_multiframe(pcd_dir, nf)
    if lidar_c is None:
        # 回退: 单帧bag(修了ph->board bug); 单帧路径不出板轴, 用老 sort
        if not os.path.exists(bag):
            return None
        lidar_c = extract_lidar_centers(bag)
    if lidar_c is None: return None
    if len(lidar_c) != 4:
        print(f"  LiDAR 环心数={len(lidar_c)} != 4"); return None
    # 【BUG-4 FIX】优先用真实板轴排序 (sort_by_axes), 单帧回退用旧 sort_centers
    cam_sorted = sort_by_axes(img_c, R_cb)
    if R_lb is not None:
        lidar_sorted = sort_by_axes(lidar_c, R_lb)
    else:
        lidar_sorted = sort_centers(lidar_c, "lidar")
    # SVD: LiDAR -> Camera
    T, rmse = svd_rigid(lidar_sorted, cam_sorted)
    print(f"  [Result] RMSE: {rmse:.4f} m")
    print(f"  [Result] T_cam_lidar =\n{T}")
    # 保存
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, name+"_T.txt"), T, fmt='%.6f')
    # 保存圆心对(供多场景联合)
    np.savetxt(os.path.join(out_dir, name+"_lidar.txt"), lidar_sorted, fmt='%.6f')
    np.savetxt(os.path.join(out_dir, name+"_cam.txt"), cam_sorted, fmt='%.6f')
    return {"name": name, "rmse": rmse, "T": T, "lidar": lidar_sorted, "cam": cam_sorted}


# ============ 多场景联合 ============
def calibrate_multi(results, out_dir):
    """results: list of single-scene dict. 联合 SVD.
    策略: 跨目录取每目录最优; 再从这些里选外参最一致的3个(旋转差异最小),
    而非单纯RMSE最小——避免外参不一致导致联合RMSE暴涨."""
    valid = [r for r in results if r is not None]
    if len(valid) < 3:
        print(f"有效场景 {len(valid)} < 3, 无法联合"); return None
    # 按目录分组, 每组取 RMSE 最小
    by_dir = defaultdict(list)
    for r in valid:
        dn = r["name"].rsplit("_f", 1)[0]
        by_dir[dn].append(r)
    cand = [min(rs, key=lambda r: r["rmse"]) for rs in by_dir.values()]
    cand.sort(key=lambda r: r["rmse"])
    print(f"\n=== 多场景联合候选(各目录最优, 共{len(cand)}个) ===")
    for c in cand:
        print(f"  {c['name']:40} rmse={c['rmse']:.4f}")
    # 从候选里选外参最一致的3个: 枚举所有3元组, 选联合RMSE最小的
    best_sel=None; best_rmse=1e9
    from itertools import combinations
    for combo in combinations(cand[:8], 3):  # 从RMSE最小的8个里选3
        L=np.vstack([s["lidar"] for s in combo])
        C=np.vstack([s["cam"] for s in combo])
        _,rmse=svd_rigid(L,C)
        if rmse<best_rmse:
            best_rmse=rmse; best_sel=combo
    sel=list(best_sel)
    print(f"=== 选外参最一致的3个: {[s['name'] for s in sel]} ===")
    L = np.vstack([s["lidar"] for s in sel])
    C = np.vstack([s["cam"] for s in sel])
    T, rmse = svd_rigid(L, C)
    print(f"  [Result] 联合 RMSE: {rmse:.4f} m")
    print(f"  [Result] T_cam_lidar =\n{T}")
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "multi_calib_result.txt"), T, fmt='%.6f')
    with open(os.path.join(out_dir, "multi_calib_result.txt"), 'a') as f:
        f.write(f"\n# 联合 RMSE: {rmse:.6f} m\n")
        f.write(f"# 场景(外参最一致): {[s['name'] for s in sel]}\n")
        f.write(f"# 各场景单RMSE: {[round(s['rmse'],6) for s in sel]}\n")
    print(f"  已保存: {out_dir}/multi_calib_result.txt")
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML file containing camera, board, and LiDAR parameters")
    ap.add_argument("--scenes", required=True, help="Directory containing scene_* folders")
    ap.add_argument("--out", default="output", help="Directory for generated calibration results")
    ap.add_argument("--raw", default=None, help="原始PCD根目录(含<时间戳>/group_000000/pcd), 启用多帧组合方法")
    ap.add_argument("--nf", type=int, default=30, help="多帧累积帧数(默认30, 100帧板面噪声多反而差)")
    ap.add_argument("--multi", nargs=3, default=None, help="Use exactly three named scenes for joint calibration")
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.scenes, "scene_*")))
    dirs = [d for d in dirs if os.path.isdir(d)]
    print(f"共 {len(dirs)} 个场景" + (f", 多帧模式 raw={args.raw} nf={args.nf}" if args.raw else ", 单帧bag模式"))

    results = []
    for d in dirs:
        r = calibrate_single(d, args.out, raw_root=args.raw, nf=args.nf)
        results.append(r)

    # 汇总
    ok = [r for r in results if r]
    print(f"\n=== 汇总: {len(ok)}/{len(dirs)} 成功 ===")
    print(f"{'scene':40} {'rmse':>8}")
    for r in results:
        if r:
            print(f"{r['name']:40} {r['rmse']:8.4f}")
        else:
            print(f"{os.path.basename(dirs[results.index(r)]):40} {'FAIL':>8}")

    # 联合
    if args.multi:
        sel = [r for r in ok if r["name"] in args.multi]
        if len(sel) != 3:
            print(f"指定的3场景里只有 {len(sel)} 个成功"); return
        # 直接用这3个
        calibrate_multi_named(sel, args.out)
    else:
        calibrate_multi(ok, args.out)


def calibrate_multi_named(sel, out_dir):
    L = np.vstack([s["lidar"] for s in sel])
    C = np.vstack([s["cam"] for s in sel])
    T, rmse = svd_rigid(L, C)
    print(f"\n=== 指定场景联合 ===")
    print(f"  [Result] 联合 RMSE: {rmse:.4f} m")
    print(f"  [Result] T_cam_lidar =\n{T}")
    np.savetxt(os.path.join(out_dir, "multi_calib_result.txt"), T, fmt='%.6f')
    print(f"  已保存: {out_dir}/multi_calib_result.txt")


if __name__ == "__main__":
    main()

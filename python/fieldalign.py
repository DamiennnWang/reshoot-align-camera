"""
fieldalign —— 靶标（二维码+圆点）+ 图像特征 的复拍照片对齐与对比核心算法。

坐标约定
--------
* 所有检测和计算都在“工作分辨率”下进行（长边 WORK_MAX_SIDE 像素）。
* 对齐矩阵 H 把“新照片工作坐标”映射到“参考照片工作坐标”。
* 安卓端 AlignEngine.kt 是本文件的逐行移植，修改算法时两边要同步。

流程
----
1. detect_markers(img)        找二维码（内容+四角）和圆点候选
2. select_reference_dots(...) 首次建档：从候选里挑出 3~6 个靶标圆点
3. align(ref, new)            码四角初始对齐 → 圆点配对 → 留一法剔除移动点
                              → ORB 特征独立求解 → 两者一致性判断 → 融合精算
4. compare(ref, aligned, ...) 光照归一化 → 抗 1px 误差差分 → 变化区域
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np

WORK_MAX_SIDE = 2000

# 圆点直径 / 二维码边长 的设计比例（make_target.py 中 16mm / 40mm）
DOT_TO_QR_RATIO = 0.4

# 判定阈值（单位：参考照片工作坐标像素，除非注明）
DOT_MOVE_TOL_FRAC = 0.30      # 圆点留一法残差 > 0.30 倍圆点直径 → 判为移动
FEATURE_MIN_INLIERS = 25      # 特征内点少于此数 → 纹理不足
FEATURE_MIN_COVERAGE = 0.10   # 特征内点覆盖测区面积比例下限
CONSISTENCY_TOL_PX = 6.0      # 靶标矩阵与特征矩阵在测区内最大偏差上限

STATUS_TEXT = {
    "confirmed": "已对齐（靶标与图像特征一致）",
    "markers_only": "对齐待确认（靶标正常，但表面纹理不足以复核）",
    "conflict": "配准需复核（靶标与表面纹理不一致，靶标可能被移动）",
    "dot_moved": "已对齐，但有圆点疑似移动或被遮挡",
    "features_only": "已按图像特征对齐（无靶标）",
    "failed": "无法对齐",
}


# ----------------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------------
@dataclass
class Markers:
    qr_text: Optional[str]
    qr_corners: Optional[np.ndarray]          # (4,2) 按二维码自身方向排列
    dot_candidates: List[Tuple[float, float, float, float]]  # (x, y, 直径, 得分)
    work_scale: float                         # 工作坐标 = 原图坐标 * work_scale
    work_size: Tuple[int, int]                # (w, h)

    def qr_side(self) -> float:
        c = self.qr_corners
        return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


@dataclass
class AlignResult:
    status: str
    message: str
    H: Optional[np.ndarray] = None            # 新 → 参考（工作坐标）
    qr_rotation: int = 0
    dots_matched: List[Optional[Tuple[float, float]]] = field(default_factory=list)
    dots_moved: List[int] = field(default_factory=list)
    dot_residuals: List[Optional[float]] = field(default_factory=list)
    marker_rms: Optional[float] = None
    feature_inliers: int = 0
    feature_coverage: float = 0.0
    consistency_px: Optional[float] = None
    roi: Optional[list] = None                # 可信对比区域（参考坐标多边形）：与对齐平面一致的区域
    k_new: float = 0.0                        # 镜头径向畸变系数（新照片 / 参考照片）；H 定义在去畸变坐标下
    k_ref: float = 0.0
    new_size: Tuple[int, int] = (0, 0)
    ref_size: Tuple[int, int] = (0, 0)

    def to_json(self) -> dict:
        d = asdict(self)
        d["H"] = None if self.H is None else self.H.tolist()
        return d


# ----------------------------------------------------------------------------
# 图像读取
# ----------------------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    """读取照片并按 EXIF 方向摆正（手机照片必须做这一步）。"""
    try:
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except ImportError:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img


def to_work(img: np.ndarray) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, WORK_MAX_SIDE / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def gray_of(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ----------------------------------------------------------------------------
# 1. 靶标检测
# ----------------------------------------------------------------------------
def detect_qr(gray: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """返回 (内容, 四角)。四角顺序以二维码自身方向为准（0=左上定位块）。"""
    for det in (cv2.QRCodeDetector(), cv2.QRCodeDetectorAruco()):
        for scale in (1.0, 0.5, 1.5):
            g = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale,
                                                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            try:
                text, pts, _ = det.detectAndDecode(g)
            except cv2.error:
                continue
            if pts is not None and text:
                return text, pts.reshape(4, 2).astype(np.float64) / scale
    return None, None


def refine_qr_corners(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """亚像素细化码角；细化后移动超过半个模块的角保持原值。"""
    side = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
    module = side / 25.0                      # 版本 2 码为 25 模块
    win = int(max(2, min(8, round(module * 0.6))))
    pts = corners.astype(np.float32).reshape(-1, 1, 2).copy()
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    try:
        cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), crit)
    except cv2.error:
        return corners
    out = corners.copy()
    for i in range(4):
        if np.linalg.norm(pts[i, 0] - corners[i]) <= module * 0.5:
            out[i] = pts[i, 0]
    return out


def point_in_quad(quad: np.ndarray, x: float, y: float, grow: float = 1.0) -> bool:
    c = quad.mean(axis=0)
    q = (quad - c) * grow + c
    return cv2.pointPolygonTest(q.astype(np.float32), (float(x), float(y)), False) >= 0


def detect_dot_candidates(gray: np.ndarray, qr_corners: Optional[np.ndarray],
                          qr_side: Optional[float]) -> List[Tuple[float, float, float, float]]:
    """找深色实心圆（透视后为椭圆）。返回 (x, y, 等效直径, 得分)，按得分降序。"""
    h, w = gray.shape
    if qr_side:
        exp_d = qr_side * DOT_TO_QR_RATIO
        d_min, d_max = exp_d * 0.35, exp_d * 2.5
    else:
        d_min, d_max = max(6, min(h, w) * 0.004), min(h, w) * 0.2
    block = int(max(15, (d_max * 2) // 2 * 2 + 1))
    bw = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 12)
    # RETR_LIST：纹理强时，阈值后的纹理连成大片，圆点落在它的“洞”（白圈）里，外轮廓模式会漏掉
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < math.pi * (d_min / 2) ** 2 or area > math.pi * (d_max / 2) ** 2:
            continue
        per = cv2.arcLength(c, True)
        if per <= 0 or len(c) < 8:
            continue
        circ = 4 * math.pi * area / (per * per)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area > 0 else 0
        if circ < 0.70 or solidity < 0.92:
            continue
        (ex, ey), (ma, mi), _ = cv2.fitEllipse(c)
        axis_ratio = min(ma, mi) / max(ma, mi)
        if axis_ratio < 0.35:
            continue
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        if qr_corners is not None and point_in_quad(qr_corners, cx, cy, 1.25):
            continue
        d = 2 * math.sqrt(area / math.pi)
        # 对比度：圆内应明显暗于外圈
        r = d / 2
        x0, x1 = int(max(0, cx - 2 * r)), int(min(w, cx + 2 * r + 1))
        y0, y1 = int(max(0, cy - 2 * r)), int(min(h, cy + 2 * r + 1))
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(xx - cx, yy - cy)
        inner = patch[rr < r * 0.6]
        ring = patch[(rr > r * 1.25) & (rr < r * 1.8)]
        if inner.size == 0 or ring.size == 0:
            continue
        contrast = (np.median(ring) - np.median(inner)) / 255.0
        if contrast < 0.15:
            continue
        size_score = 1.0
        if qr_side:
            size_score = math.exp(-abs(math.log(d / (qr_side * DOT_TO_QR_RATIO))))
        score = circ * solidity * min(1.0, contrast * 2) * size_score
        out.append((float(cx), float(cy), float(d), float(score)))
    # 同心候选只留最小的：RETR_LIST 还会返回白圈外沿（“洞”的轮廓），它与圆点同心、也“内暗外亮”，
    # 但质心受透视影响更大；真正的靶标是里面的黑圆。
    out.sort(key=lambda t: t[2])
    kept = []
    for c in out:
        if any(math.hypot(c[0] - k[0], c[1] - k[1]) < 0.5 * k[2] for k in kept):
            continue
        kept.append(c)
    kept.sort(key=lambda t: -t[3])
    return kept


def detect_markers(img_full: np.ndarray) -> Markers:
    work, s = to_work(img_full)
    g = gray_of(work)
    text, corners = detect_qr(g)
    side = None
    if corners is not None:
        corners = refine_qr_corners(g, corners)
        side = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
    dots = detect_dot_candidates(g, corners, side)
    return Markers(text, corners, dots, s, (g.shape[1], g.shape[0]))


def select_reference_dots(m: Markers, max_dots: int = 3) -> List[Tuple[float, float, float]]:
    """首次建档：自动挑选得分最高、彼此分散的圆点（手机上会让人确认）。"""
    chosen: List[Tuple[float, float, float]] = []
    for x, y, d, sc in m.dot_candidates:
        if sc < 0.35:
            continue
        if any(math.hypot(x - cx, y - cy) < 3 * d for cx, cy, _ in chosen):
            continue
        chosen.append((x, y, d))
        if len(chosen) >= max_dots:
            break
    return chosen


# ----------------------------------------------------------------------------
# 2. 对齐
# ----------------------------------------------------------------------------
def _ph(H: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def _fit_h(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(src.astype(np.float64), dst.astype(np.float64), 0)
    return H


def _match_dots(H0: np.ndarray, ref_dots, new_cands, new_qr_side: float):
    """用初始矩阵把参考圆点预测到新图，在半径内找最近候选。"""
    Hinv = np.linalg.inv(H0)
    pred = _ph(Hinv, [(x, y) for x, y, _ in ref_dots])
    radius = max(12.0, 0.6 * new_qr_side)
    exp_d = new_qr_side * DOT_TO_QR_RATIO
    used, out, dists = set(), [], []
    for px, py in pred:
        best, bd = None, radius
        for j, (cx, cy, cd, _) in enumerate(new_cands):
            if j in used or not (0.4 * exp_d <= cd <= 2.5 * exp_d):
                continue
            dd = math.hypot(cx - px, cy - py)
            if dd < bd:
                best, bd = j, dd
        if best is None:
            out.append(None)
            dists.append(None)
        else:
            used.add(best)
            out.append((new_cands[best][0], new_cands[best][1]))
            dists.append(bd)
    return out, dists


def _find_dot_rectified(new_g: np.ndarray, H0: np.ndarray, ref_dot) -> Optional[Tuple[float, float]]:
    """把新照片中参考圆点附近的区域“摆正”到参考视角（用码四角的初始矩阵），在那里找圆点。
    强透视下圆点被压成很扁的椭圆、全局检测会漏掉；摆正后它又是已知大小的圆。返回新照片坐标或 None。"""
    x, y, d = ref_dot
    half = int(round(3 * d))
    x0, y0 = x - half, y - half
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
    patch = cv2.warpPerspective(new_g, T @ H0, (2 * half, 2 * half), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    blk = int(max(15, (d * 2) // 2 * 2 + 1))
    bw = cv2.adaptiveThreshold(cv2.GaussianBlur(patch, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, blk, 12)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    best, best_dist = None, 1.5 * d
    exp_area = math.pi * (d / 2) ** 2
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.3 * exp_area or area > 2.5 * exp_area or len(c) < 8:
            continue
        per = cv2.arcLength(c, True)
        if per <= 0 or 4 * math.pi * area / (per * per) < 0.6:
            continue
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        r = math.sqrt(area / math.pi)
        yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
        rr = np.hypot(xx - cx, yy - cy)
        inner, ring = patch[rr < r * 0.6], patch[(rr > r * 1.25) & (rr < r * 1.8)]
        if inner.size == 0 or ring.size == 0 or (np.median(ring) - np.median(inner)) < 40:
            continue
        dist = math.hypot(cx - half, cy - half)
        if dist < best_dist:
            best, best_dist = (cx + x0, cy + y0), dist
    if best is None:
        return None
    q = _ph(np.linalg.inv(H0), [best])[0]
    return float(q[0]), float(q[1])


def _roi_grid(ref_qr: np.ndarray, ref_dots, size: Tuple[int, int], n: int = 9) -> np.ndarray:
    pts = np.vstack([ref_qr] + ([np.array([(x, y) for x, y, _ in ref_dots])] if ref_dots else []))
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    mx, my = (x1 - x0) * 0.25, (y1 - y0) * 0.25
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(size[0] - 1, x1 + mx), min(size[1] - 1, y1 + my)
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    return np.array([(x, y) for y in ys for x in xs])


def marker_mask(size: Tuple[int, int], qr: np.ndarray, dots_xy_d) -> np.ndarray:
    """255 = 可用于特征/差分；0 = 靶标区域（排除，避免“自己验证自己”）。"""
    w, h = size
    m = np.full((h, w), 255, np.uint8)
    c = qr.mean(axis=0)
    cv2.fillConvexPoly(m, ((qr - c) * 1.3 + c).astype(np.int32), 0)
    for x, y, d in dots_xy_d:
        cv2.circle(m, (int(round(x)), int(round(y))), int(d * 1.0 + 4), 0, -1)
    return m


def feature_homography(ref_g, new_g, ref_mask, new_mask, ransac_px: float = 4.0):
    """SIFT 特征 + 比值检验 + RANSAC（对齐用 SIFT：定位精度比 ORB 高一个量级，只慢约 30%）。
    返回 (H, 内点数, (新图内点, 参考内点), 参考特征数, 新图特征数, (全部候选匹配 新, 参考))。"""
    det = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = det.detectAndCompute(ref_g, ref_mask)
    k2, d2 = det.detectAndCompute(new_g, new_mask)
    n1, n2 = len(k1), len(k2)
    if d1 is None or d2 is None or n1 < 8 or n2 < 8:
        return None, 0, None, n1, n2, None
    good = []
    for pair in cv2.BFMatcher(cv2.NORM_L2).knnMatch(d2, d1, k=2):
        if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 8:
        return None, len(good), None, n1, n2, None
    src = np.float64([k2[m.queryIdx].pt for m in good])
    dst = np.float64([k1[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px, maxIters=5000, confidence=0.999)
    if H is None:
        return None, 0, None, n1, n2, (src, dst)
    inl = inl.ravel().astype(bool)
    return H, int(inl.sum()), (src[inl], dst[inl]), n1, n2, (src, dst)


# ----------------------------------------------------------------------------
# 镜头径向畸变：x_d = x_u·(1 + k·r²)，以画面中心为原点、长边归一化。
# 单应矩阵只描述理想针孔相机；手机广角/主摄边缘有几个像素的桶形畸变，
# 会让画面边缘出现大片“假变化”，甚至把圆点误判为移动。
# 做法：用特征匹配同时估计两张照片各自的 k，所有几何计算在“去畸变坐标”下进行。
# ----------------------------------------------------------------------------
DIST_K_MIN, DIST_K_MAX = -0.30, 0.20
DIST_MIN_INLIERS = 80          # 至少这么多特征内点才估计畸变
DIST_MIN_SPREAD = 0.25         # 内点需覆盖参考图面积的比例（太集中时畸变不可辨识）


def undistort_pts(pts, k: float, size: Tuple[int, int]) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if k == 0.0 or len(pts) == 0:
        return pts.copy()
    w, h = size
    f = float(max(w, h)); c = np.array([w / 2.0, h / 2.0])
    d = (pts - c) / f
    u = d.copy()
    for _ in range(10):                       # 不动点迭代求逆
        r2 = np.sum(u * u, axis=1, keepdims=True)
        u = d / (1.0 + k * r2)
    return u * f + c


def distort_pts(pts, k: float, size: Tuple[int, int]) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if k == 0.0 or len(pts) == 0:
        return pts.copy()
    w, h = size
    f = float(max(w, h)); c = np.array([w / 2.0, h / 2.0])
    u = (pts - c) / f
    r2 = np.sum(u * u, axis=1, keepdims=True)
    return u * (1.0 + k * r2) * f + c


def estimate_distortion(src, dst, size_new, size_ref) -> Tuple[float, float, float, float]:
    """由特征对应点估计 (k_new, k_ref)。返回 (k_new, k_ref, 原残差, 新残差)；改善不明显时返回 0。"""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)

    def cost(kn, kr):
        a = undistort_pts(src, kn, size_new); b = undistort_pts(dst, kr, size_ref)
        H, _ = cv2.findHomography(a, b, 0)
        if H is None:
            return 1e9
        e = np.linalg.norm(_ph(H, a) - b, axis=1)
        e = np.minimum(e, 6.0)                  # 截断，抑制残余误匹配
        # 轻微偏向 k=0：两张同样畸变、取景相近时 k 不可辨识，这时宁可不校正
        return float(np.sqrt(np.mean(e * e))) + 0.05 * (abs(kn) + abs(kr))

    c0 = cost(0.0, 0.0)
    ks = np.linspace(DIST_K_MIN, DIST_K_MAX, 11)
    best = min((cost(a, b), a, b) for a in ks for b in ks)
    # 局部细化（模式搜索）：当前步长下一直走到不再变好，再把步长减半
    step = (DIST_K_MAX - DIST_K_MIN) / 20
    while step > 2e-4:
        _, kn, kr = best
        cand = min((cost(kn + da, kr + db), kn + da, kr + db)
                   for da in (-step, 0, step) for db in (-step, 0, step) if da or db)
        if cand[0] < best[0]:
            best = cand
        else:
            step /= 2
    _, kn, kr = best
    c1 = best[0]
    if c1 < 0.8 * c0 and c0 - c1 > 0.15:
        return float(kn), float(kr), c0, c1
    return 0.0, 0.0, c0, c0


def distortion_two_pass(inl_loose, all_m, size_new, size_ref) -> Tuple[float, float]:
    """两遍估计：宽松内点 → 去畸变坐标下严格阈值重选内点 → 再估计（剔除残余误匹配）。"""
    kn, kr, _, _ = estimate_distortion(inl_loose[0], inl_loose[1], size_new, size_ref)
    if kn == 0.0 and kr == 0.0:
        return 0.0, 0.0
    a, b = undistort_pts(all_m[0], kn, size_new), undistort_pts(all_m[1], kr, size_ref)
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.999)
    if H is None or inl.sum() < DIST_MIN_INLIERS:
        return kn, kr
    inl = inl.ravel().astype(bool)
    kn2, kr2, _, _ = estimate_distortion(all_m[0][inl], all_m[1][inl], size_new, size_ref)
    return (kn, kr) if (kn2 == 0.0 and kr2 == 0.0) else (kn2, kr2)


def map_new_to_ref(res: "AlignResult", pts) -> np.ndarray:
    """新照片工作坐标 → 参考照片工作坐标（含畸变）。"""
    u = undistort_pts(pts, res.k_new, res.new_size)
    return distort_pts(_ph(res.H, u), res.k_ref, res.ref_size)


TEXTURE_RICH_KP = 800   # 两图遮掉靶标后都有这么多特征点，才算“纹理丰富”


def align(ref_img_full: np.ndarray, ref_markers: Markers, ref_dots,
          new_img_full: np.ndarray, new_markers: Optional[Markers] = None) -> Tuple[AlignResult, np.ndarray, np.ndarray]:
    """返回 (结果, 参考工作图, 新工作图)。"""
    ref_w, _ = to_work(ref_img_full)
    new_w, _ = to_work(new_img_full)
    ref_g, new_g = gray_of(ref_w), gray_of(new_w)
    nm = new_markers or detect_markers(new_img_full)
    rm = ref_markers

    if nm.qr_corners is None:
        return AlignResult("failed", "新照片中没有识别到二维码"), ref_w, new_w
    if nm.qr_text != rm.qr_text:
        return AlignResult("failed", f"二维码不同：{nm.qr_text!r} ≠ {rm.qr_text!r}"), ref_w, new_w

    ref_qr = rm.qr_corners
    side_new = nm.qr_side()
    side_ref = rm.qr_side()
    W, Hh = nm.work_size

    # --- 码四角初始对齐；四种循环顺序都试，取圆点吻合最好的（兼容不同扫码库的角点顺序）
    best = None
    for r in range(4):
        q = np.roll(nm.qr_corners, -r, axis=0)
        H0 = cv2.getPerspectiveTransform(q.astype(np.float32), ref_qr.astype(np.float32)).astype(np.float64)
        matched, dists = _match_dots(H0, ref_dots, nm.dot_candidates, side_new)
        n_ok = sum(m is not None for m in matched)
        err = sum(d for d in dists if d is not None) / max(1, n_ok)
        key = (n_ok, -err)
        if best is None or key > best[0]:
            best = (key, r, q, H0, matched)
    _, rot, new_qr, H0, matched = best
    # 全局检测没找到的圆点：摆正到参考视角再找一次（强透视下圆点很扁，全局检测会漏掉）。
    # 每找到新圆点，就用三个定位块角 + 已找到的圆点重算预测矩阵（码小时第 4 个推算角误差外推很大）。
    Hp = H0
    for pass_no in range(3):
        found_new = False
        for i, mt in enumerate(matched):
            if mt is None:
                px, py = _ph(np.linalg.inv(Hp), [ref_dots[i][:2]])[0]
                if 0 <= px < W and 0 <= py < Hh:
                    matched[i] = _find_dot_rectified(new_g, Hp, ref_dots[i])
                    found_new |= matched[i] is not None
        got = [i for i, mt in enumerate(matched) if mt is not None]
        if (pass_no > 0 and not found_new) or not got or all(m is not None for m in matched):
            break
        src = np.vstack([new_qr[[0, 1, 3]]] + [np.array([matched[i]]) for i in got])
        dst = np.vstack([ref_qr[[0, 1, 3]]] + [np.array([ref_dots[i][:2]]) for i in got])
        H1 = _fit_h(src, dst)
        if H1 is None:
            break
        Hp = H1
    matched_raw = list(matched)

    # --- 图像特征（遮掉靶标）：先用宽松阈值匹配，用来估计镜头畸变
    ref_mask = marker_mask(rm.work_size, ref_qr, ref_dots)
    new_dots_xyd = [(c[0], c[1], c[2]) for c in nm.dot_candidates[:12]]
    new_mask = marker_mask(nm.work_size, new_qr, new_dots_xyd)
    _, n_loose, inl_loose, kp_ref, kp_new, all_m = feature_homography(ref_g, new_g, ref_mask, new_mask, ransac_px=10.0)
    k_n = k_r = 0.0
    if inl_loose is not None and n_loose >= DIST_MIN_INLIERS:
        spread = cv2.contourArea(cv2.convexHull(inl_loose[1].astype(np.float32))) / (rm.work_size[0] * rm.work_size[1])
        if spread >= DIST_MIN_SPREAD:
            k_n, k_r = distortion_two_pass(inl_loose, all_m, nm.work_size, rm.work_size)
    Un = lambda p: undistort_pts(p, k_n, nm.work_size)
    Ur = lambda p: undistort_pts(p, k_r, rm.work_size)
    # 以下几何计算全部在去畸变坐标下进行
    new_qr, ref_qr = Un(new_qr), Ur(ref_qr)
    matched = [None if m is None else tuple(Un([m])[0]) for m in matched]
    ref_d = [tuple(p) for p in Ur([(x, y) for x, y, _ in ref_dots])] if ref_dots else []
    dot_d = [d for _, _, d in ref_dots]

    def pts_for(skip: Optional[int], include):
        # 二维码第 3 个角（右下，没有定位块）是检测器推算出来的，误差明显大于另外三个角；
        # 圆点够用时只用三个定位块角参与精算。
        n_dots = sum(1 for i in include if matched[i] is not None and i != skip)
        idx = [0, 1, 3] if n_dots >= 2 else [0, 1, 2, 3]
        src, dst = [new_qr[idx]], [ref_qr[idx]]
        for i in sorted(include):
            if matched[i] is None or i == skip:
                continue
            src.append(np.array([matched[i]]))
            dst.append(np.array([ref_d[i]]))
        return np.vstack(src), np.vstack(dst)

    # 每个圆点在新照片中的视直径（由初始矩阵在该点的局部尺度换算）。
    # 残差在新照片像素里量、阈值按视直径给：新照片拍得远/斜时，参考坐标里的误差会被放大，不能用固定阈值。
    H0inv = np.linalg.inv(H0)
    dot_d_new = []
    for (x, y), dd in zip(ref_d, dot_d):
        c0 = _ph(H0inv, [(x, y)])[0]
        ring = _ph(H0inv, [(x + dd / 2, y), (x - dd / 2, y), (x, y + dd / 2), (x, y - dd / 2)])
        dot_d_new.append(float(2 * np.mean(np.linalg.norm(ring - c0, axis=1))))

    all_valid = {i for i, mt in enumerate(matched) if mt is not None}
    H_all = _fit_h(*pts_for(None, all_valid))
    if H_all is None:
        H_all = H0

    # --- 图像特征：在去畸变坐标下用严格阈值重新 RANSAC，独立求解
    H_ft, n_inl, inl_pts = None, 0, None
    if all_m is not None and len(all_m[0]) >= 8:
        a_, b_ = Un(all_m[0]), Ur(all_m[1])
        H_ft, inl = cv2.findHomography(a_, b_, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.999)
        if H_ft is not None:
            inl = inl.ravel().astype(bool)
            n_inl, inl_pts = int(inl.sum()), (a_[inl], b_[inl])

    grid = _roi_grid(ref_qr, [(x, y, dd) for (x, y), dd in zip(ref_d, dot_d)], rm.work_size)
    roi_area = max(1.0, np.ptp(grid[:, 0]) * np.ptp(grid[:, 1]))

    def _cov(pts):
        return float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))) / roi_area)

    # 直接匹配不够（多半是视角太斜，描述子对不上）：先用靶标矩阵把新照片“摆正”到参考视角，再匹配一次。
    # 若靶标其实被搬到了别的表面，摆正后照样对不上，仍会判为“需复核”。
    if H_ft is None or n_inl < FEATURE_MIN_INLIERS or _cov(inl_pts[1]) < FEATURE_MIN_COVERAGE:
        rect = cv2.warpPerspective(new_g, H_all, rm.work_size, flags=cv2.INTER_LINEAR)
        rmask = cv2.warpPerspective(np.full(new_g.shape, 255, np.uint8), H_all, rm.work_size, flags=cv2.INTER_NEAREST)
        rmask = cv2.erode(rmask, np.ones((9, 9), np.uint8)) & ref_mask
        _, n2, inl2, _, _, _ = feature_homography(ref_g, rect, ref_mask, rmask, ransac_px=4.0)
        if inl2 is not None and n2 > n_inl:
            src_new = _ph(np.linalg.inv(H_all), inl2[0])          # 摆正图坐标 → 新照片坐标
            H2, m2 = cv2.findHomography(src_new, inl2[1], cv2.RANSAC, 4.0)
            if H2 is not None:
                m2 = m2.ravel().astype(bool)
                H_ft, n_inl, inl_pts = H2, int(m2.sum()), (src_new[m2], inl2[1][m2])
    coverage = _cov(inl_pts[1]) if (H_ft is not None and n_inl >= 4) else 0.0
    features_ok = H_ft is not None and n_inl >= FEATURE_MIN_INLIERS and coverage >= FEATURE_MIN_COVERAGE

    # --- 圆点是否移动
    valid = set(all_valid)
    moved: List[int] = []
    residuals: List[Optional[float]] = [None] * len(ref_dots)
    if features_ok:
        # 有可靠的表面纹理：直接用特征矩阵判断（独立于靶标，且不受小码外推误差影响）
        Hfi = np.linalg.inv(H_ft)
        for i in sorted(all_valid):
            residuals[i] = float(np.linalg.norm(_ph(Hfi, [ref_d[i]])[0] - matched[i]))
            if residuals[i] > DOT_MOVE_TOL_FRAC * dot_d_new[i]:
                moved.append(i)
                valid.discard(i)
    else:
        # 只能靠靶标：留一法。阈值考虑外推不确定度——给点加 0.7px 噪声重算，看预测漂多远。
        rng = np.random.default_rng(0)

        def loo(valid_set):
            out: List[Optional[float]] = [None] * len(ref_dots)
            tol: List[float] = [0.0] * len(ref_dots)
            for i in sorted(valid_set):
                s_, d_ = pts_for(i, valid_set)
                Hl = _fit_h(s_, d_)
                if Hl is None:
                    continue
                out[i] = float(np.linalg.norm(_ph(np.linalg.inv(Hl), [ref_d[i]])[0] - matched[i]))
                preds = []
                for _ in range(20):
                    Hn = _fit_h(s_ + rng.normal(0, 0.7, s_.shape), d_)
                    if Hn is not None:
                        preds.append(_ph(np.linalg.inv(Hn), [ref_d[i]])[0])
                spread = float(np.sqrt(np.mean(np.sum((np.array(preds) - np.mean(preds, axis=0)) ** 2, axis=1)))) if preds else 0.0
                tol[i] = max(DOT_MOVE_TOL_FRAC * dot_d_new[i], 3.0 * spread)
            return out, tol

        # 一次只剔除最离谱的一个；至少剩 3 个圆点才继续判断（再少就没有冗余）
        residuals, tols = loo(valid)
        while len(valid) >= 3:
            cand = [(residuals[i] / tols[i], i) for i in valid if residuals[i] is not None and residuals[i] > tols[i]]
            if not cand:
                break
            _, worst = max(cand)
            moved.append(worst)
            valid.discard(worst)
            residuals, tols = loo(valid)

    # 没找到的圆点：预测位置在画面外 → 只是没拍到；在画面内 → 可能被遮挡/移动
    H_tmp = _fit_h(*pts_for(None, valid))
    if H_tmp is None:
        H_tmp = H0
    pred_all = distort_pts(_ph(np.linalg.inv(H_tmp), ref_d), k_n, nm.work_size) if ref_d else np.zeros((0, 2))
    out_of_frame, missing = [], []
    for i, mt in enumerate(matched):
        if mt is None:
            px, py = pred_all[i]
            (out_of_frame if not (0 <= px < W and 0 <= py < Hh) else missing).append(i)

    s, d = pts_for(None, valid)
    H_mk = _fit_h(s, d)
    if H_mk is None:
        H_mk = H0
    marker_rms = float(np.sqrt(np.mean(np.sum((_ph(H_mk, s) - d) ** 2, axis=1))))
    consistency = None
    if H_ft is not None and n_inl >= 4:
        # 一致性：靶标点（码角 + 未移动的圆点）在“特征矩阵”下的最大残差；在靶标点本身上比较，不外推
        consistency = float(np.max(np.linalg.norm(_ph(H_ft, s) - d, axis=1)))
    # 新照片比参考小（拍得远）时，参考坐标里的测量误差按比例放大
    cons_tol = max(CONSISTENCY_TOL_PX, 0.06 * side_ref) * max(1.0, side_ref / max(side_new, 1e-6))

    res = AlignResult("confirmed", "", H=H_mk, qr_rotation=rot,
                      dots_matched=matched_raw, dots_moved=moved, k_new=k_n, k_ref=k_r,
                      new_size=nm.work_size, ref_size=rm.work_size,
                      dot_residuals=residuals, marker_rms=marker_rms,
                      feature_inliers=n_inl, feature_coverage=coverage,
                      consistency_px=consistency)

    note = ""
    if not features_ok:
        if kp_ref >= TEXTURE_RICH_KP and kp_new >= TEXTURE_RICH_KP:
            # 两边纹理都丰富却对不上 —— 靶标很可能被搬到了别处，或拍的不是同一表面
            res.status = "conflict"
            note = f"表面纹理丰富（{kp_ref}/{kp_new} 个特征点）但几乎无法匹配"
        else:
            res.status = "markers_only"
    elif consistency is not None and consistency > cons_tol:
        res.status = "conflict"
        note = f"靶标与表面纹理相差 {consistency:.1f}px"
    else:
        # 融合精算：靶标点 + 特征内点，每个点同等权重。
        # 圆点质心精度约 0.2~0.8px，而上千个特征点合起来精得多；给靶标加大权重反而会把矩阵带偏
        # （靶标点多在边缘，杠杆大）。特征少时靶标自然占更大比重。
        src_f, dst_f = inl_pts
        w = 1
        src_all = np.vstack([np.repeat(s, w, axis=0), src_f])
        dst_all = np.vstack([np.repeat(d, w, axis=0), dst_f])
        H_fused, _ = cv2.findHomography(src_all, dst_all, cv2.RANSAC, 3.0)
        if H_fused is not None:
            res.H = H_fused
            res.marker_rms = float(np.sqrt(np.mean(np.sum((_ph(H_fused, s) - d) ** 2, axis=1))))
        res.roi = distort_pts(hull_list(np.vstack([dst_f, d])), k_r, rm.work_size).tolist()
    if res.roi is None:
        # 没有可靠的特征：只在靶标附近（码和圆点外扩 25%）比较
        g = grid
        res.roi = distort_pts([[g[:, 0].min(), g[:, 1].min()], [g[:, 0].max(), g[:, 1].min()],
                               [g[:, 0].max(), g[:, 1].max()], [g[:, 0].min(), g[:, 1].max()]], k_r, rm.work_size).tolist()

    res.dots_moved = sorted(set(moved))
    if res.status == "confirmed" and (res.dots_moved or missing):
        res.status = "dot_moved"
    extra = [note] if note else []
    if missing:
        extra.append("画面内未找到圆点 " + ",".join(str(i + 1) for i in missing))
    if res.dots_moved:
        extra.append("疑似移动圆点 " + ",".join(str(i + 1) for i in res.dots_moved))
    if out_of_frame:
        extra.append("圆点 " + ",".join(str(i + 1) for i in out_of_frame) + " 不在画面内")
    if k_n or k_r:
        extra.append(f"已校正镜头畸变（k={k_n:+.3f}/{k_r:+.3f}）")
    res.message = STATUS_TEXT[res.status] + ("；" + "；".join(extra) if extra else "")
    return res, ref_w, new_w


# ----------------------------------------------------------------------------
# 2b. 无二维码：靠照片自身特征识别属于哪个系列，并对齐
# ----------------------------------------------------------------------------
ID_SIDE = 1000              # 识别用分辨率（长边），快速
ID_MIN_INLIERS = 40         # 最佳候选至少这么多 RANSAC 内点
ID_MIN_COVERAGE = 0.12      # 内点覆盖参考照片面积比例下限
ID_UNIQUE_RATIO = 2.0       # 最佳候选内点数 ≥ 第二名的 2 倍才算唯一
FEAT_ALIGN_MIN_INLIERS = 40


@dataclass
class Signature:
    """一张照片的特征签名（识别分辨率下的 ORB 关键点 + 描述子）。"""
    pts: np.ndarray        # (N,2) 识别分辨率坐标
    desc: Optional[np.ndarray]
    size: Tuple[int, int]  # 识别分辨率 (w, h)


def signature(img_full: np.ndarray, mask_fn=None) -> Signature:
    h, w = img_full.shape[:2]
    s = min(1.0, ID_SIDE / max(h, w))
    g = gray_of(cv2.resize(img_full, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA))
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
    kp, desc = orb.detectAndCompute(g, None)
    return Signature(np.float64([k.pt for k in kp]).reshape(-1, 2), desc, (g.shape[1], g.shape[0]))


def match_signatures(new: Signature, ref: Signature):
    """返回 (内点数, 覆盖率, H 新→参考[识别分辨率])。"""
    if new.desc is None or ref.desc is None or len(new.pts) < 8 or len(ref.pts) < 8:
        return 0, 0.0, None
    good = []
    for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(new.desc, ref.desc, k=2):
        if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 8:
        return 0, 0.0, None
    src = new.pts[[m.queryIdx for m in good]]
    dst = ref.pts[[m.trainIdx for m in good]]
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0, maxIters=3000, confidence=0.995)
    if H is None:
        return 0, 0.0, None
    inl = inl.ravel().astype(bool)
    n = int(inl.sum())
    if n < 4:
        return n, 0.0, H
    hull = cv2.convexHull(dst[inl].astype(np.float32))
    cov = float(cv2.contourArea(hull) / (ref.size[0] * ref.size[1]))
    return n, cov, H


@dataclass
class IdentifyResult:
    decision: str                  # "unique" | "ambiguous" | "no_match"
    best_key: Optional[object]
    ranking: List[Tuple[object, int, float]]   # (key, 内点数, 覆盖率)，降序
    message: str


def identify(new_sig: Signature, refs) -> IdentifyResult:
    """refs: [(key, Signature)]。判断新照片属于哪个系列；背景相似、分不清时返回 ambiguous。"""
    rank = []
    for key, sig in refs:
        n, cov, _ = match_signatures(new_sig, sig)
        rank.append((key, n, cov))
    rank.sort(key=lambda t: -t[1])
    if not rank or rank[0][1] < ID_MIN_INLIERS or rank[0][2] < ID_MIN_COVERAGE:
        return IdentifyResult("no_match", None, rank, "没有找到特征足够吻合的系列")
    best = rank[0]
    second = rank[1][1] if len(rank) > 1 else 0
    if second >= ID_MIN_INLIERS * 0.5 and best[1] < ID_UNIQUE_RATIO * second:
        return IdentifyResult("ambiguous", None, rank,
                              f"与多个系列都相似（{best[1]} vs {second} 个匹配点），"
                              "请贴二维码或手动选择")
    return IdentifyResult("unique", best[0], rank, f"按图像特征识别（{best[1]} 个匹配点）")


def align_features_only(ref_img_full: np.ndarray, new_img_full: np.ndarray):
    """无靶标时的纯特征对齐（工作分辨率，含镜头畸变估计）。返回 (AlignResult, 参考工作图, 新工作图)。"""
    ref_w, _ = to_work(ref_img_full)
    new_w, _ = to_work(new_img_full)
    rs = (ref_w.shape[1], ref_w.shape[0]); ns = (new_w.shape[1], new_w.shape[0])
    full = lambda img: np.full(img.shape[:2], 255, np.uint8)
    _, n0, inl0, _, _, all_m = feature_homography(gray_of(ref_w), gray_of(new_w), full(ref_w), full(new_w), ransac_px=10.0)
    res = AlignResult("failed", "", feature_inliers=n0, new_size=ns, ref_size=rs)
    if inl0 is None or n0 < FEAT_ALIGN_MIN_INLIERS:
        res.message = STATUS_TEXT["failed"] + f"：图像特征不足（{n0} 个匹配点）"
        return res, ref_w, new_w
    k_n = k_r = 0.0
    spread = cv2.contourArea(cv2.convexHull(inl0[1].astype(np.float32))) / (rs[0] * rs[1])
    if n0 >= DIST_MIN_INLIERS and spread >= DIST_MIN_SPREAD:
        k_n, k_r = distortion_two_pass(inl0, all_m, ns, rs)
    a, b = undistort_pts(all_m[0], k_n, ns), undistort_pts(all_m[1], k_r, rs)
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.999)
    if H is None:
        res.message = STATUS_TEXT["failed"]
        return res, ref_w, new_w
    inl = inl.ravel().astype(bool)
    n = int(inl.sum())
    if n < FEAT_ALIGN_MIN_INLIERS:
        res.message = STATUS_TEXT["failed"] + f"：图像特征不足（{n} 个匹配点）"
        return res, ref_w, new_w
    hull = cv2.convexHull(b[inl].astype(np.float32))
    res.feature_inliers = n
    res.feature_coverage = float(cv2.contourArea(hull) / (rs[0] * rs[1]))
    res.H, res.k_new, res.k_ref = H, k_n, k_r
    res.roi = distort_pts(hull_list(b[inl]), k_r, rs).tolist()
    res.status = "features_only"
    res.message = STATUS_TEXT["features_only"] + f"（{n} 个匹配点，覆盖 {res.feature_coverage:.0%}）"
    if k_n or k_r:
        res.message += f"；已校正镜头畸变（k={k_n:+.3f}/{k_r:+.3f}）"
    return res, ref_w, new_w


def recover_qr(ref_img_full: np.ndarray, rm: Markers, new_img_full: np.ndarray) -> Optional[Markers]:
    """新照片的码读不出（太小/太糊/太斜）但参考照片有码：先按特征对齐，把参考码所在区域
    “摆正并放大”后再解码；内容一致就把四角映射回新照片，返回新照片的 Markers。"""
    if rm.qr_corners is None:
        return None
    fo, ref_w, new_w = align_features_only(ref_img_full, new_img_full)
    if fo.H is None:
        return None
    side = rm.qr_side()
    q = rm.qr_corners
    x0, y0 = int(max(0, q[:, 0].min() - side)), int(max(0, q[:, 1].min() - side))
    x1, y1 = int(min(rm.work_size[0] - 1, q[:, 0].max() + side)), int(min(rm.work_size[1] - 1, q[:, 1].max() + side))
    aligned, _ = warp_to_ref(new_w, fo, rm.work_size)
    scale = max(1.0, 300.0 / side)                          # 放大到码边长约 300 像素
    big = cv2.resize(aligned[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    text, corners = detect_qr(gray_of(big))
    if text is None or text != rm.qr_text:
        return None
    ref_pts = corners / scale + np.array([x0, y0])
    u = undistort_pts(ref_pts, fo.k_ref, fo.ref_size)
    new_pts = distort_pts(_ph(np.linalg.inv(fo.H), u), fo.k_new, fo.new_size)
    g = gray_of(new_w)
    new_pts = refine_qr_corners(g, new_pts)
    side_new = float(np.mean([np.linalg.norm(new_pts[i] - new_pts[(i + 1) % 4]) for i in range(4)]))
    return Markers(text, new_pts, detect_dot_candidates(g, new_pts, side_new), to_work(new_img_full)[1],
                   (g.shape[1], g.shape[0]))


# ----------------------------------------------------------------------------
# 3. 对比
# ----------------------------------------------------------------------------
@dataclass
class CompareResult:
    changed_ratio: float
    regions: List[Tuple[int, int, int, int]]   # x, y, w, h（参考工作坐标）
    overlap_ratio: float
    similarity: float = 1.0                     # 对比区内两图（光照归一化后）的相关系数，<0.3 表示整体差异大
    light_ratio: float = 0.0                    # 明暗变化（阴影、水渍、光照不均）占比，不计入 changed_ratio
    light_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    glare_ratio: float = 0.0                    # 反光/过曝/死黑而未参与比较的面积占比
    blur_sigma: float = 0.0                     # 为匹配清晰度对较清晰一张施加的模糊（>0 表示两张清晰度不同）


def hull_list(pts: np.ndarray) -> list:
    return cv2.convexHull(np.asarray(pts, np.float32)).reshape(-1, 2).astype(float).tolist()


def comparison_mask(ref_size: Tuple[int, int], res: AlignResult, ref_qr=None, ref_dots=()) -> np.ndarray:
    """对比时使用的区域：对齐平面上的可信区域（略外扩），再去掉靶标本身。
    匹配点只落在与单应矩阵一致的平面上，所以墙外背景、远处景物等视差区会被排除。"""
    w, h = ref_size
    m = np.zeros((h, w), np.uint8)
    if res.roi:
        poly = np.array(res.roi, np.float64)
        c = poly.mean(axis=0)
        cv2.fillConvexPoly(m, poly.astype(np.int32), 255)   # 不外扩：外扩会漏到平面外的背景
    else:
        m[:] = 255
    if ref_qr is not None:
        m &= marker_mask(ref_size, ref_qr, ref_dots)
    return m


def warp_to_ref(new_w: np.ndarray, H, ref_size: Tuple[int, int], k_new: float = 0.0, k_ref: float = 0.0):
    """把新照片变换到参考照片坐标。H 可以是矩阵，也可以是 AlignResult（自动带上畸变参数）。"""
    if isinstance(H, AlignResult):
        k_new, k_ref, H = H.k_new, H.k_ref, H.H
    w, h = ref_size
    if k_new == 0.0 and k_ref == 0.0:
        aligned = cv2.warpPerspective(new_w, H, ref_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.warpPerspective(np.full(new_w.shape[:2], 255, np.uint8), H, ref_size, flags=cv2.INTER_NEAREST)
    else:
        # 参考像素 → 去畸变 → H⁻¹ → 加新照片畸变 → 新照片像素；映射平滑，先在 1/8 网格上算再插值
        ns = (new_w.shape[1], new_w.shape[0])
        step = 8
        gx, gy = np.meshgrid(np.arange(0, w + step, step, dtype=np.float64), np.arange(0, h + step, step, dtype=np.float64))
        pts = np.c_[gx.ravel(), gy.ravel()]
        q = distort_pts(_ph(np.linalg.inv(H), undistort_pts(pts, k_ref, ref_size)), k_new, ns)
        mx, my = _grid_upsample(q, gx.shape, step, w, h)
        aligned = cv2.remap(new_w, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.remap(np.full(new_w.shape[:2], 255, np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    k = max(7, int(0.008 * max(ref_size))) | 1     # 重叠区边缘插值不可靠，收进约 0.8% 画面尺寸
    valid = cv2.erode(valid, np.ones((k, k), np.uint8))
    return aligned, valid


def _grid_upsample(q: np.ndarray, shape, step: int, w: int, h: int):
    """网格点 (i*step, j*step) 上的映射值 → 每个像素的映射（双线性）。"""
    gy, gx = shape
    mx = q[:, 0].reshape(shape); my = q[:, 1].reshape(shape)
    xs = np.arange(w) / step; ys = np.arange(h) / step
    x0 = np.minimum(xs.astype(int), gx - 2); y0 = np.minimum(ys.astype(int), gy - 2)
    fx = (xs - x0)[None, :]; fy = (ys - y0)[:, None]
    def interp(m):
        a = m[np.ix_(y0, x0)]; b = m[np.ix_(y0, x0 + 1)]; c = m[np.ix_(y0 + 1, x0)]; d = m[np.ix_(y0 + 1, x0 + 1)]
        return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy).astype(np.float32)
    return interp(mx), interp(my)


ILLUM_SIGMA = 10        # 光照归一化窗口：越小越能压住阴影，但大面积变化的内部会被“归一”掉
LIGHT_MEDIAN_HALF = 21  # 明暗/结构分类的中值窗口（半分辨率像素）：保留阴影台阶，去掉裂缝这类细线
STRUCT_MIN = 0.10       # 结构残差（对数亮度比）下限


def _light_residual(ref_w: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    """对数亮度比去掉“台阶状”分量后的残差。
    阴影、水渍是乘性的明暗变化：对数比在区域内近似常数，边界是台阶，大窗口中值能保留台阶；
    裂缝、剥落等结构变化是细线/纹理改变，中值会把它去掉 → 残差大。"""
    ga = gray_of(ref_w).astype(np.float32); gb = gray_of(aligned).astype(np.float32)
    lr = np.log(gb + 4.0) - np.log(ga + 4.0)
    small = cv2.resize(lr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    q = np.clip((small + 2.0) * (255.0 / 4.0), 0, 255).astype(np.uint8)
    med = cv2.medianBlur(q, LIGHT_MEDIAN_HALF).astype(np.float32) * (4.0 / 255.0) - 2.0
    med = cv2.resize(med, (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.abs(lr - med)


GLARE_SAT = 245          # 灰度 ≥ 此值视为过曝/反光（纹理已丢失）
GLARE_DARK = 4           # 灰度 ≤ 此值视为死黑
GLARE_HALO = np.log(1.3) # 与饱和核相连、局部亮度比整体偏离超过 30% 的区域算反光光晕


def _glare_mask(ref_w: np.ndarray, aligned: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """两张照片中任一张过曝/反光/死黑的区域（不可比较）。
    镜面反光是“加性”的：中心饱和，周围一圈纹理对比度被冲淡，既不是结构变化也不是乘性阴影，只能排除。
    光晕的判定：局部平均亮度比整体明显偏离，且与饱和核连通。"""
    ga = gray_of(ref_w); gb = gray_of(aligned)
    v = valid > 0
    op = lambda x: cv2.morphologyEx(x.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    sat_a = op((ga >= GLARE_SAT) & v); sat_b = op((gb >= GLARE_SAT) & v)
    dark = op(((ga <= GLARE_DARK) | (gb <= GLARE_DARK)) & v)   # 死黑只排除本身，不向外生长（否则会吞掉阴影）
    if not (sat_a.any() or sat_b.any() or dark.any()):
        return np.zeros(v.shape, np.uint8)
    vf = v.astype(np.float32)
    den = cv2.GaussianBlur(vf, (0, 0), 6) + 1e-3
    la = np.log(cv2.GaussianBlur(ga.astype(np.float32) * vf, (0, 0), 6) / den + 4.0)
    lb = np.log(cv2.GaussianBlur(gb.astype(np.float32) * vf, (0, 0), 6) / den + 4.0)
    d = lb - la
    d -= float(np.median(d[v]))
    out = dark.copy()
    # 光晕只朝“变亮”的方向生长：新照片反光 → 新照片更亮（d>0）；参考照片反光 → d<0
    for core, elev in ((sat_b, d > GLARE_HALO), (sat_a, d < -GLARE_HALO)):
        if not core.any():
            continue
        grow = ((elev & v) | core).astype(np.uint8)
        n, lab = cv2.connectedComponents(grow, connectivity=8)
        keep = np.zeros(n, bool); keep[np.unique(lab[core])] = True; keep[0] = False
        out |= keep[lab]
    out = out.astype(np.uint8)
    return cv2.dilate(out, np.ones((7, 7), np.uint8))


BLUR_SIGMAS = (0.5, 0.8, 1.1, 1.5, 2.0, 2.6, 3.4)


def _match_sharpness(a: np.ndarray, b: np.ndarray, m: np.ndarray):
    """两张清晰度不同（对焦、手抖、远近）时，把较清晰的一张模糊到与另一张一致。
    在半分辨率上搜索使两图平均差最小的高斯 σ。返回 (a', b', σ)，σ>0 表示 a 被模糊，<0 表示 b 被模糊。"""
    sa = cv2.resize(a, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    sb = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    sm = cv2.resize(m.astype(np.uint8), (sa.shape[1], sa.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if sm.sum() < 100:
        return a, b, 0.0
    best, best_s = float(np.mean(np.abs(sa - sb)[sm])), 0.0
    for sign, src, other in ((1, sa, sb), (-1, sb, sa)):
        for s in BLUR_SIGMAS:
            e = float(np.mean(np.abs(cv2.GaussianBlur(src, (0, 0), s / 2) - other)[sm]))
            if e < best * 0.97:
                best, best_s = e, sign * s
    if best_s > 0:
        a = cv2.GaussianBlur(a, (0, 0), best_s)
    elif best_s < 0:
        b = cv2.GaussianBlur(b, (0, 0), -best_s)
    return a, b, best_s


def _illum_norm(g: np.ndarray, valid: np.ndarray) -> np.ndarray:
    f = g.astype(np.float32) + 1.0
    v = (valid > 0).astype(np.float32)
    # 只在有效区域内求局部均值，避免黑边把亮度拉低
    num = cv2.GaussianBlur(f * v, (0, 0), ILLUM_SIGMA)
    den = cv2.GaussianBlur(v, (0, 0), ILLUM_SIGMA) + 1e-3
    return f / np.maximum(num / den, 1.0)


def resolution_ratio(res: "AlignResult") -> float:
    """参考照片每像素对应新照片多少像素的倒数：>1 表示新照片拍得更远/更小，放大后比参考模糊。"""
    if res.H is None or not res.roi:
        return 1.0
    c = np.mean(np.array(res.roi), axis=0)
    sq = np.array([c + (-20, -20), c + (20, -20), c + (20, 20), c + (-20, 20)])
    u = undistort_pts(sq, res.k_ref, res.ref_size)
    back = distort_pts(_ph(np.linalg.inv(res.H), u), res.k_new, res.new_size)
    a_new = cv2.contourArea(back.astype(np.float32))
    return float(np.sqrt(1600.0 / max(a_new, 1e-6)))


def compare(ref_w: np.ndarray, aligned: np.ndarray, valid: np.ndarray,
            exclude_mask: Optional[np.ndarray] = None,
            sensitivity: float = 6.0, min_area_frac: float = 2e-5, res_ratio: float = 1.0):
    """返回 (CompareResult, 变化掩膜 uint8：255=结构变化，128=明暗变化, 差异图)。
    res_ratio：resolution_ratio(res)。新照片分辨率更低（拍得远）时，在两者中较低的分辨率下比较：
    参考坐标里 1 个新照片像素可能对应 3 个参考像素，按参考分辨率比较会把配准误差和细纹理当成变化。"""
    if res_ratio > 1.2:
        f = 1.0 / min(res_ratio, 4.0)
        h0, w0 = ref_w.shape[:2]
        sz = (max(16, int(w0 * f)), max(16, int(h0 * f)))
        rs = lambda im, interp: cv2.resize(im, sz, interpolation=interp)
        c, m_small, d_small = compare(rs(ref_w, cv2.INTER_AREA), rs(aligned, cv2.INTER_AREA),
                                      rs(valid, cv2.INTER_NEAREST),
                                      None if exclude_mask is None else rs(exclude_mask, cv2.INTER_NEAREST),
                                      sensitivity, min_area_frac, 1.0)
        sx, sy = w0 / sz[0], h0 / sz[1]
        up = lambda rr: [(int(x * sx), int(y * sy), int(np.ceil(w * sx)), int(np.ceil(h * sy))) for x, y, w, h in rr]
        c.regions, c.light_regions = up(c.regions), up(c.light_regions)
        mask = cv2.resize(m_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
        return c, mask, cv2.resize(d_small, (w0, h0), interpolation=cv2.INTER_LINEAR)
    glare = _glare_mask(ref_w, aligned, valid)
    m = valid > 0
    if exclude_mask is not None:
        m &= exclude_mask > 0
    m_all = m.copy()
    m &= glare == 0
    a = _illum_norm(gray_of(ref_w), valid)
    b = _illum_norm(gray_of(aligned), valid)
    a, b, blur_s = _match_sharpness(a, b, m)
    a = cv2.GaussianBlur(a, (3, 3), 0)
    b = cv2.GaussianBlur(b, (3, 3), 0)
    # 抗配准误差：取 3x3 邻域平移后的最小差（双向）
    diff = None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            Mt = np.float32([[1, 0, dx], [0, 1, dy]])
            bs = cv2.warpAffine(b, Mt, (b.shape[1], b.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            d = np.abs(a - bs)
            diff = d if diff is None else np.minimum(diff, d)
    diff2 = None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            Mt = np.float32([[1, 0, dx], [0, 1, dy]])
            as_ = cv2.warpAffine(a, Mt, (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            d = np.abs(b - as_)
            diff2 = d if diff2 is None else np.minimum(diff2, d)
    diff = np.maximum(diff, diff2)
    glare_ratio = float((m_all & ~m).sum() / max(1, m_all.sum()))
    if m.sum() < 100:
        return (CompareResult(0.0, [], float(m.mean()), 0.0, glare_ratio=glare_ratio, blur_sigma=blur_s),
                np.zeros(valid.shape, np.uint8), diff)
    # 整体相似度：自适应阈值只能发现“局部”变化，画面整体都不同时要靠它提示
    av, bv = a[m] - a[m].mean(), b[m] - b[m].mean()
    sim = float((av * bv).sum() / (np.sqrt((av * av).sum() * (bv * bv).sum()) + 1e-9))
    vals = diff[m]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-6
    thr = max(med + sensitivity * mad, 0.12)
    mask = ((diff > thr) & m).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(12, int(min_area_frac * mask.size))
    out = np.zeros_like(mask)
    regions, light = [], []
    resid = _light_residual(ref_w, aligned)
    rv = resid[m]
    r_med = float(np.median(rv)); r_mad = float(np.median(np.abs(rv - r_med))) * 1.4826 + 1e-6
    struct_thr = max(r_med + 6 * r_mad, STRUCT_MIN)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        sel = lab == i
        # 区域内残差的高分位：有一截细线/结构改变就算结构变化
        score = float(np.percentile(resid[sel], 90))
        if score > struct_thr:
            out[sel] = 255
            regions.append((int(x), int(y), int(w), int(h)))
        else:
            out[sel] = 128
            light.append((int(x), int(y), int(w), int(h)))
    total = max(1, m.sum())
    ratio = float((out == 255).sum() / total)
    lratio = float((out == 128).sum() / total)
    out[(m_all & ~m) & (out == 0)] = 64     # 64 = 反光/过曝，未比较
    return CompareResult(ratio, regions, float(m.mean()), sim, lratio, light, glare_ratio, blur_s), out, diff


def quality_warnings(c: "CompareResult", new_w: Optional[np.ndarray] = None) -> List[str]:
    """拍摄质量提示：反光/过曝、清晰度不一致、偏暗。提示用户是否需要重拍。"""
    w = []
    if c.glare_ratio > 0.025:
        w.append(f"反光/过曝区域 {c.glare_ratio:.0%} 未参与比较，建议换个角度或避开强光重拍")
    if c.blur_sigma >= 1.5:
        w.append("新照片比参考模糊（对焦或手抖），细小裂缝可能漏检，建议重拍")
    elif c.blur_sigma <= -1.5:
        w.append("参考照片比新照片模糊，可考虑把更清晰的一张设为参考")
    if new_w is not None and float(np.mean(gray_of(new_w))) < 45:
        w.append("照片偏暗，建议补光后重拍")
    return w


# ----------------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------------
def draw_markers(img_w: np.ndarray, qr: Optional[np.ndarray], dots, color=(0, 200, 0)) -> np.ndarray:
    vis = img_w.copy()
    if qr is not None:
        cv2.polylines(vis, [qr.astype(np.int32)], True, (255, 120, 0), 3)
        cv2.circle(vis, tuple(qr[0].astype(int)), 8, (0, 0, 255), -1)
    for i, dt in enumerate(dots):
        if dt is None:
            continue
        x, y = dt[0], dt[1]
        r = int(dt[2] / 2) if len(dt) > 2 else 12
        cv2.circle(vis, (int(x), int(y)), r + 6, color, 3)
        cv2.putText(vis, str(i + 1), (int(x + r + 8), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return vis


def make_views(ref_w, aligned, valid, change_mask, regions, light_regions=()):
    h, w = ref_w.shape[:2]
    side = np.hstack([ref_w, aligned])
    overlay = cv2.addWeighted(ref_w, 0.5, aligned, 0.5, 0)
    cell = max(40, w // 16)
    yy, xx = np.mgrid[0:h, 0:w]
    chk = (((yy // cell) + (xx // cell)) % 2).astype(bool)
    checker = ref_w.copy()
    checker[chk] = aligned[chk]
    heat = ref_w.copy()
    red = np.zeros_like(heat); red[:, :, 2] = 255
    blue = np.zeros_like(heat); blue[:, :, 0] = 255; blue[:, :, 1] = 140
    k = np.ones((7, 7), np.uint8)
    lt = cv2.dilate((change_mask == 128).astype(np.uint8), k) > 0
    heat[lt] = cv2.addWeighted(heat, 0.6, blue, 0.4, 0)[lt]          # 明暗变化：浅蓝
    st_ = cv2.dilate((change_mask == 255).astype(np.uint8), k) > 0
    heat[st_] = cv2.addWeighted(heat, 0.35, red, 0.65, 0)[st_]       # 结构变化：红
    gl = change_mask == 64
    heat[gl] = ((heat[gl].astype(np.uint16) + 2 * np.array([200, 200, 200], np.uint16)) // 3).astype(np.uint8)  # 反光：灰白
    heat[valid == 0] = (heat[valid == 0] * 0.4).astype(np.uint8)
    for x, y, bw, bh in light_regions:
        cv2.rectangle(heat, (x - 6, y - 6), (x + bw + 6, y + bh + 6), (255, 200, 0), 2)
    for x, y, bw, bh in regions:
        cv2.rectangle(heat, (x - 6, y - 6), (x + bw + 6, y + bh + 6), (0, 255, 255), 3)
    return {"side_by_side": side, "overlay": overlay, "checker": checker, "changes": heat}


def markers_to_json(m: Markers, ref_dots=None) -> dict:
    return {
        "qr_text": m.qr_text,
        "qr_corners": None if m.qr_corners is None else m.qr_corners.tolist(),
        "work_scale": m.work_scale,
        "work_size": list(m.work_size),
        "dots": ref_dots,
    }


def markers_from_json(d: dict) -> Tuple[Markers, list]:
    m = Markers(d["qr_text"], None if d["qr_corners"] is None else np.array(d["qr_corners"]),
                [], d["work_scale"], tuple(d["work_size"]))
    return m, [tuple(x) for x in (d.get("dots") or [])]

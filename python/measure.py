"""
病害参数测量：比例尺（二维码尺寸 → 毫米）、裂缝提取与测量、前后对比、文字结论。

所有测量都在“参考照片工作图坐标”下进行（复拍照片已对齐到这个坐标），因此
  * 比例尺只需参考照片中有二维码即可，复拍照片里没有码也能换算毫米；
  * 前后两次的裂缝可以逐像素比较，得到新增长度。

算法（全部为 OpenCV 基础操作，手机端 Kotlin 版逐步对应）：
  1. 光照归一化：灰度 / 局部均值（σ=15）。
  2. 黑帽（闭运算 − 原图，13×13 椭圆核）：细长的暗线变亮，宽于 ~10px 的暗块被抑制。
  3. 双阈值（中位数 + k·MAD）+ 连通性：强响应像素所在的弱响应连通区才保留。
  4. 细化（Zhang–Suen）得到骨架；骨架长度 = 横竖相邻 1 + 斜向相邻 √2。
  5. 过滤：骨架长 ≥ 30px 且平均宽 ≤ 8px（骨料斑点、污点被去掉）。
  6. 宽度（面积法，可到亚像素）：裂缝区域内黑帽响应之和 / (裂缝深度 c × 长度)；
     局部宽度用 11×11 窗口内的同一比值，最大宽度取 95 分位。
  7. 新增：复拍骨架中离参考裂缝 > 4px 的部分。
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_QR_MM = 40.0
MIN_LEN_PX = 30.0
MAX_MEAN_W_PX = 8.0
GROW_TOL_PX = 4
SQRT2 = 2 ** 0.5
RES_LIMIT_PX = 2.0            # 平均宽度小于此值（像素）时测不准：照片模糊会把更细的线“摊宽”到约 2 像素


# ------------------------------------------------------------------ 比例尺
def qr_size_mm(text: Optional[str]) -> float:
    """二维码内容里写了尺寸（如 'Z023;40mm'）就用它，否则按默认 40mm。"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm", text or "", re.I)
    if m:
        v = float(m.group(1))
        if 5 <= v <= 500:
            return v
    return DEFAULT_QR_MM


@dataclass
class Scale:
    H: np.ndarray            # 图像像素 → 墙面毫米（二维码所在平面）
    qr_mm: float
    qr_side_px: float

    def mm_per_px(self, at) -> float:
        """某点处的局部比例（透视下各处不同）：1 像素对应多少毫米。"""
        p = np.array(at, np.float64)

        def m(q):
            v = self.H @ np.array([q[0], q[1], 1.0])
            return v[:2] / v[2]
        a = m(p + [1, 0]) - m(p)
        b = m(p + [0, 1]) - m(p)
        return float(np.sqrt(abs(a[0] * b[1] - a[1] * b[0])))


def scale_from_qr(qr_corners, qr_mm: float) -> Optional[Scale]:
    if qr_corners is None:
        return None
    src = np.asarray(qr_corners, np.float32).reshape(4, 2)
    dst = np.float32([[0, 0], [qr_mm, 0], [qr_mm, qr_mm], [0, qr_mm]])
    H = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    side = float(np.mean([np.linalg.norm(src[i] - src[(i + 1) % 4]) for i in range(4)]))
    return Scale(H, qr_mm, side)


# ------------------------------------------------------------------ 裂缝
def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang–Suen 细化。输入 0/1 uint8，输出 0/1 uint8。"""
    img = (mask > 0).astype(np.uint8)
    img = np.pad(img, 1)
    while True:
        changed = False
        for step in (0, 1):
            P = img
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            c = P[1:-1, 1:-1]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(x.astype(np.int32) for x in nb)
            seq = nb + [p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int32) for i in range(8))
            if step == 0:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0
            rm = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1]


def neighbor_count(sk: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.float32); k[1, 1] = 0
    return cv2.filter2D(sk.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT).astype(np.int32)


def prune(sk: np.ndarray, n: int = 8) -> np.ndarray:
    """去掉骨架上短于 n 像素的毛刺（粘连的斑点、裂缝边缘的小凸起），再把主干两端长回来。"""
    s = (sk > 0).astype(np.uint8)
    for _ in range(n):
        nb = neighbor_count(s)
        s[(s == 1) & (nb <= 1)] = 0
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(n):
        nb = neighbor_count(s)
        ends = ((s == 1) & (nb == 1)).astype(np.uint8)
        s |= (cv2.dilate(ends, k3) & (sk > 0) & (s == 0)).astype(np.uint8)
    return s


def link_length(skel: np.ndarray, labels: Optional[np.ndarray] = None, n: int = 0):
    """骨架长度：每对相邻骨架像素计一次（横竖 1、斜向 √2）。给 labels 时按连通区分别统计。"""
    s = skel > 0
    pairs = [(s[:, :-1] & s[:, 1:], 1.0, (slice(None), slice(None, -1))),
             (s[:-1, :] & s[1:, :], 1.0, (slice(None, -1), slice(None))),
             (s[:-1, :-1] & s[1:, 1:], SQRT2, (slice(None, -1), slice(None, -1))),
             (s[:-1, 1:] & s[1:, :-1], SQRT2, (slice(None, -1), slice(1, None)))]
    # 同时存在横/竖连接时，斜向连接是多余的（三角形），去掉以免高估
    h, v = pairs[0][0], pairs[1][0]
    d1 = pairs[2][0] & ~(h[:-1, :] & v[:, 1:]) & ~(v[:, :-1] & h[1:, :])
    d2 = pairs[3][0] & ~(h[:-1, :] & v[:, :-1]) & ~(v[:, 1:] & h[1:, :])
    pairs[2] = (d1, SQRT2, pairs[2][2]); pairs[3] = (d2, SQRT2, pairs[3][2])
    if labels is None:
        return float(sum(p.sum() * w for p, w, _ in pairs))
    out = np.zeros(n, np.float64)
    for p, w, sl in pairs:
        lab = labels[sl][p]
        out += np.bincount(lab, minlength=n)[:n] * w
    return out


def crack_response(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) + 1.0
    m = (mask > 0).astype(np.float32)
    bg = cv2.GaussianBlur(g * m, (0, 0), 15) / np.maximum(cv2.GaussianBlur(m, (0, 0), 15), 1e-3)
    r = np.clip(g / np.maximum(bg, 1.0), 0, 2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    bh = cv2.morphologyEx(r, cv2.MORPH_CLOSE, k) - r
    bh = cv2.GaussianBlur(bh, (0, 0), 0.8)
    bh[m == 0] = 0
    return bh


LINE_LEN = 15
LINE_DIRS = 12


def line_kernels():
    ks = []
    c = LINE_LEN // 2
    for i in range(LINE_DIRS):
        a = np.pi * i / LINE_DIRS
        k = np.zeros((LINE_LEN, LINE_LEN), np.float32)
        dx, dy = np.cos(a) * c, np.sin(a) * c
        cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1, cv2.LINE_AA)
        ks.append(k / k.sum())
    return ks


_KS = None


def line_response(bh: np.ndarray) -> np.ndarray:
    """沿 12 个方向、长 15px 的线段求平均，取最大：细长的裂缝保持高响应，圆形斑点被摊薄。"""
    global _KS
    if _KS is None:
        _KS = line_kernels()
    out = np.zeros_like(bh)
    for k in _KS:
        np.maximum(out, cv2.filter2D(bh, -1, k, borderType=cv2.BORDER_CONSTANT), out=out)
    return out


@dataclass
class Crack:
    length_px: float
    mean_w_px: float
    max_w_px: float
    angle_deg: float          # 走向：与水平方向夹角（0~180）
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    grow_px: float = 0.0      # 相对参考新增的长度
    is_new: bool = False      # 参考里完全没有
    limited: bool = False     # 宽度接近分辨率极限（约 2 像素），实际可能更细


@dataclass
class CrackMap:
    cracks: List[Crack]
    mask: np.ndarray          # 裂缝区域（0/255）
    skel: np.ndarray          # 骨架（0/1）
    total_px: float
    max_w_px: float
    evid: Optional[np.ndarray] = None   # 裂缝响应强度图（各尺度线段响应取最大，原图分辨率）：比较两张照片“那里本来有没有”用
    bg_med: float = 0.0                 # 响应图的背景水平（中位数）和波动（MAD）
    bg_mad: float = 0.0


K_HI = 6.0                    # 强阈值：中位数 + K_HI × MAD（线段响应）
K_LO = 3.0                    # 弱阈值
SCALES = (1, 3, 9)            # 多尺度：原图检细裂缝；缩小 3、9 倍后检宽裂缝（宽于 ~10px 的暗带在原图尺度会被当成背景）
MESH_DENSITY = 0.08           # 骨架长度 / 外接框面积 超过此值且很曲折：暗色粗糙区域里的“网状”假裂缝
MIN_ELONG = 5.0               # 长度至少是平均宽度的 5 倍


def _detect_scale(gray: np.ndarray, mask: np.ndarray, min_len: float):
    """单一尺度检测。返回候选列表：每个为 dict(L, mean_w, max_w, ang, ys, xs, x0, y0, reg)，坐标为该尺度像素。"""
    bh = crack_response(gray, mask)
    ln = line_response(bh)
    vals = ln[mask > 0][::7]
    if vals.size < 100:
        return [], ln
    med = float(np.median(vals)); mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-4
    t_hi = max(med + K_HI * mad, 0.20)
    t_lo = max(med + K_LO * mad, 0.10)
    lo = ((ln > t_lo) & (bh > t_lo)).astype(np.uint8)
    lo = cv2.morphologyEx(lo, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(lo, connectivity=8)
    strong = np.zeros(n, bool); strong[np.unique(lab[ln > t_hi])] = True; strong[0] = False
    region = strong[lab].astype(np.uint8)
    skel = prune(thin(region), 8)
    n2, slab = cv2.connectedComponents(skel, connectivity=8)
    lens = link_length(skel, slab, n2)
    rlab = cv2.connectedComponents(region, connectivity=8)[1]
    # 面积法只积分“高出背景”的部分，并只取骨架两侧 5px 以内（排除粘连的骨料斑点）
    base = float(np.median(bh[mask > 0][::7]))
    near_sk = cv2.dilate(skel, np.ones((11, 11), np.uint8))
    resp = np.maximum(bh - base, 0) * region * near_sk
    ys_all, xs_all = np.nonzero(skel)
    out = []
    for i in range(1, n2):
        L = lens[i]
        if L < min_len:
            continue
        sel = slab[ys_all, xs_all] == i
        ys, xs = ys_all[sel], xs_all[sel]
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        # 网状假裂缝：骨架又密又曲折
        diag = float(np.hypot(x1 - x0, y1 - y0))
        if L / max((x1 - x0) * (y1 - y0), 1) > MESH_DENSITY and L > 2.5 * diag:
            continue
        rid = np.bincount(rlab[ys, xs]).argmax()
        pad = 12
        X0, Y0 = max(x0 - pad, 0), max(y0 - pad, 0)
        X1, Y1 = min(x1 + pad, skel.shape[1]), min(y1 + pad, skel.shape[0])
        reg = rlab[Y0:Y1, X0:X1] == rid
        c = max(float(np.percentile(bh[ys, xs], 90)) - base, 0.25)   # 裂缝“深度”（相对背景变暗的比例）
        rr = (resp[Y0:Y1, X0:X1] * reg).astype(np.float32)
        mean_w = float(rr.sum()) / (c * L)
        if mean_w > MAX_MEAN_W_PX or L < MIN_ELONG * mean_w:
            continue
        # 局部宽度：21×21 窗口内 响应和 /(c × 骨架长度)
        ss = np.zeros(rr.shape, np.float32); ss[ys - Y0, xs - X0] = L / len(xs)   # 每个骨架像素平均代表的长度
        S_r = cv2.boxFilter(rr, -1, (21, 21), normalize=False)
        S_s = cv2.boxFilter(ss, -1, (21, 21), normalize=False)
        loc = S_r[ys - Y0, xs - X0] / np.maximum(c * S_s[ys - Y0, xs - X0], 1e-6)
        max_w = float(np.percentile(loc, 90)) if loc.size else mean_w
        # 走向：骨架点的主方向
        pts = np.stack([xs, ys], 1).astype(np.float64)
        ev, evec = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
        d = evec[:, 1]
        ang = float(np.degrees(np.arctan2(-d[1], d[0])) % 180)
        out.append(dict(L=float(L), mean_w=mean_w, max_w=max(max_w, mean_w), ang=ang, ys=ys, xs=xs,
                        X0=int(X0), Y0=int(Y0), reg=reg))
    return out, ln


def detect_cracks(gray: np.ndarray, mask: np.ndarray) -> CrackMap:
    """多尺度检测并合并：重叠的候选取“在自己尺度下宽度最接近 4 像素”的那个（测得最准）。"""
    h, w = gray.shape
    cands = []
    for f in SCALES:
        if f > 1 and min(h, w) // f < 100:
            break
        if f == 1:
            g, m = gray, mask
        else:
            g = cv2.resize(gray, (w // f, h // f), interpolation=cv2.INTER_AREA)
            m = cv2.resize(mask, (w // f, h // f), interpolation=cv2.INTER_NEAREST)
        found, ln = _detect_scale(g, m, MIN_LEN_PX if f == 1 else 20.0)
        evid = ln.copy() if f == 1 else np.maximum(evid, cv2.resize(ln, (w, h), interpolation=cv2.INTER_LINEAR))
        for c in found:
            cands.append((abs(np.log(max(c["mean_w"], 0.5) / 4.0)), f, c))
    cands.sort(key=lambda t: (t[0], t[1]))
    region = np.zeros((h, w), np.uint8); skel = np.zeros((h, w), np.uint8)
    cracks: List[Crack] = []
    for _, f, c in cands:
        # 候选的骨架、区域放大回原图
        rh, rw = c["reg"].shape
        X0, Y0 = c["X0"] * f, c["Y0"] * f
        X1, Y1 = min(X0 + rw * f, w), min(Y0 + rh * f, h)
        sk_s = np.zeros((rh, rw), np.uint8); sk_s[c["ys"] - c["Y0"], c["xs"] - c["X0"]] = 1
        if f == 1:
            sk, reg = sk_s, c["reg"].astype(np.uint8)
        else:
            sk = cv2.resize(sk_s, (rw * f, rh * f), interpolation=cv2.INTER_NEAREST)[:Y1 - Y0, :X1 - X0]
            sk = thin(sk)
            reg = cv2.resize(c["reg"].astype(np.uint8), (rw * f, rh * f), interpolation=cv2.INTER_NEAREST)[:Y1 - Y0, :X1 - X0]
        n_sk = int(sk.sum())
        if n_sk == 0:
            continue
        # 已被接受的裂缝覆盖了一半以上：重复，跳过
        kk = 2 * f + 3
        near = cv2.dilate(region[Y0:Y1, X0:X1], np.ones((kk, kk), np.uint8))
        if int((sk & near).sum()) > 0.5 * n_sk:
            continue
        ys, xs = np.nonzero(sk)
        cracks.append(Crack(c["L"] * f, c["mean_w"] * f, c["max_w"] * f, c["ang"],
                            (int(X0 + xs.min()), int(Y0 + ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
                            (float(X0 + xs.mean()), float(Y0 + ys.mean())),
                            limited=(f == 1 and c["mean_w"] < RES_LIMIT_PX)))
        region[Y0:Y1, X0:X1] |= reg
        skel[Y0:Y1, X0:X1] |= sk
    cracks.sort(key=lambda k: -k.length_px)
    total = sum(k.length_px for k in cracks)
    mw = max((k.max_w_px for k in cracks), default=0.0)
    ev = evid[mask > 0][::7]
    bmed = float(np.median(ev)) if ev.size else 0.0
    bmad = float(np.median(np.abs(ev - bmed))) * 1.4826 if ev.size else 0.0
    return CrackMap(cracks, region * 255, skel, total, mw, evid, bmed, bmad)


@dataclass
class CrackChange:
    ref: CrackMap
    new: CrackMap
    grow_px: float            # 新增长度（复拍中离参考裂缝 > 4px 的骨架）
    grow_skel: np.ndarray     # 新增部分的骨架
    lost_px: float            # 参考有、复拍没有（修补、遮挡或拍摄问题）


PRESENT_RATIO = 0.5           # 另一张照片同一位置的响应达到这一比例，就认为“那里本来就有”


def _drop_short(sk: np.ndarray, min_px: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sk, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab].astype(np.uint8)


BLUR_SIGMAS = (0.5, 0.8, 1.1, 1.5, 2.0, 2.6, 3.4)


def _local_ratio(gray, mask, sigma=15.0):
    g = gray.astype(np.float32) + 1.0
    m = (mask > 0).astype(np.float32)
    bg = cv2.GaussianBlur(g * m, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-3)
    return np.clip(g / np.maximum(bg, 1.0), 0, 2)


def match_sharpness(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    """两张照片清晰度不同（对焦、手抖、远近）时，把较清晰的一张模糊到与另一张一致。
    否则清晰那张噪声大、阈值高，同一条淡裂缝一张检出、一张检不出，会被误判成新增或消失。
    在半分辨率、去掉明暗差异后搜索使两图差别最小的高斯 σ。返回 (a', b', σ)：σ>0 表示 a 被模糊，<0 表示 b。"""
    ha = cv2.resize(a, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    hb = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    sm8 = cv2.resize(mask, (ha.shape[1], ha.shape[0]), interpolation=cv2.INTER_NEAREST)
    ra, rb = _local_ratio(ha, sm8, 7.5), _local_ratio(hb, sm8, 7.5)
    sm = sm8 > 0
    if sm.sum() < 100:
        return a, b, 0.0
    best, best_s = float(np.mean(np.abs(ra - rb)[sm])), 0.0
    for sign, src, other in ((1, ra, rb), (-1, rb, ra)):
        for sg in BLUR_SIGMAS:
            e = float(np.mean(np.abs(cv2.GaussianBlur(src, (0, 0), sg / 2) - other)[sm]))
            if e < best * 0.97:
                best, best_s = e, sign * sg
    if best_s > 0:
        a = cv2.GaussianBlur(a, (0, 0), best_s)
    elif best_s < 0:
        b = cv2.GaussianBlur(b, (0, 0), -best_s)
    return a, b, best_s


def compare_cracks(ref_gray, new_gray, mask) -> CrackChange:
    ref_gray, new_gray, _ = match_sharpness(ref_gray, new_gray, mask)
    r = detect_cracks(ref_gray, mask)
    n = detect_cracks(new_gray, mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW_TOL_PX + 1, 2 * GROW_TOL_PX + 1))
    near_ref = cv2.dilate(r.mask, k) > 0
    near_new = cv2.dilate(n.mask, k) > 0
    # 新增 = 这次有、上次那里“确实没有”：不仅上次没检出，而且上次该处的裂缝响应不到这次的一半。
    # （若隐若现的细裂缝可能一张刚过阈值、另一张差一点，只看“检出与否”会把它误算成新增/消失。）
    ke = np.ones((2 * GROW_TOL_PX + 1, 2 * GROW_TOL_PX + 1), np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    # 只比较“高出背景”的部分：邻域取最大会把背景噪声也抬高，所以被比较的一方减去 中位数 + 2.5×MAD
    def excess_near(c):
        return cv2.dilate(c.evid, ke) - (c.bg_med + 2.5 * c.bg_mad)

    def excess_at(c):
        return np.maximum(cv2.dilate(c.evid, k3) - c.bg_med, 0)
    absent_in_ref = excess_near(r) < PRESENT_RATIO * excess_at(n)
    absent_in_new = excess_near(n) < PRESENT_RATIO * excess_at(r)
    # 比较区边缘 8px 内不算：裂缝在一张照片里刚好被边界截断，会被误当成新增/消失
    inner = cv2.erode((mask > 0).astype(np.uint8), np.ones((17, 17), np.uint8), borderValue=0) > 0
    grow = ((n.skel > 0) & ~near_ref & absent_in_ref & inner).astype(np.uint8)
    lost = ((r.skel > 0) & ~near_new & absent_in_new & inner).astype(np.uint8)
    # 零散的几个像素不算（对齐误差、噪声）
    grow = _drop_short(grow, 6); lost = _drop_short(lost, 6)
    # 每条新裂缝的新增长度、是否全新
    for c in n.cracks:
        x, y, w, h = c.bbox
        g = grow[y:y + h, x:x + w]
        sk = n.skel[y:y + h, x:x + w]
        c.grow_px = link_length(g)
        c.is_new = c.grow_px > 0.8 * link_length(sk)
    return CrackChange(r, n, link_length(grow), grow, link_length(lost))


# ------------------------------------------------------------------ 结论
@dataclass
class Rules:
    """判定阈值（可按工程要求调整）。"""
    grow_min_px: float = 40.0        # 新增长度至少这么多像素才算“延长/新增”（避免对齐误差、若隐若现的细纹引起的误判）
    grow_min_ratio: float = 0.05     # 且至少为原长的 5%
    widen_min_px: float = 1.5        # 最大宽度增加至少 1.5 像素
    widen_min_ratio: float = 0.3     # 且至少 30%
    widen_mean_ratio: float = 0.15   # 同时整体平均宽度增加至少 15%（最大宽度是局部估计，单看它容易受清晰度影响）
    attention_mm: float = 0.3        # 最大宽度超过此值提示重点关注（有毫米比例时）


def fmt_len(px: float, mmpp: Optional[float]) -> str:
    return f"{px * mmpp:.1f} mm" if mmpp else f"{px:.0f} px"


def fmt_w(px: float, mmpp: Optional[float]) -> str:
    return f"{px * mmpp:.2f} mm" if mmpp else f"{px:.1f} px"


def conclude(ch: CrackChange, mmpp: Optional[float], light_regions: int = 0, light_ratio: float = 0.0,
             rules: Rules = Rules()) -> Tuple[str, str]:
    """返回 (等级, 文字结论)。等级：无裂缝 / 稳定 / 延长 / 变宽 / 新增 / 延长且变宽。"""
    r, n = ch.ref, ch.new
    parts = []
    new_cracks = [c for c in n.cracks if c.is_new and c.grow_px >= rules.grow_min_px]
    longer = ch.grow_px >= max(rules.grow_min_px, rules.grow_min_ratio * r.total_px)
    def mean_w(cm):
        tot = sum(c.length_px for c in cm.cracks)
        return sum(c.mean_w_px * c.length_px for c in cm.cracks) / tot if tot > 0 else 0.0
    wider = (r.cracks and n.cracks
             and (n.max_w_px - r.max_w_px) >= max(rules.widen_min_px, rules.widen_min_ratio * r.max_w_px)
             and mean_w(n) >= (1 + rules.widen_mean_ratio) * mean_w(r))
    if not n.cracks and not r.cracks:
        level = "无裂缝"
        parts.append("对比区域内未检出裂缝。")
    else:
        parts.append(f"检出裂缝 {len(n.cracks)} 条，总长 {fmt_len(n.total_px, mmpp)}"
                     f"（上次 {len(r.cracks)} 条、{fmt_len(r.total_px, mmpp)}），"
                     f"最大宽度 {fmt_w(n.max_w_px, mmpp)}（上次 {fmt_w(r.max_w_px, mmpp)}）。")
        changes = []
        if new_cracks:
            changes.append(f"新增裂缝 {len(new_cracks)} 条（共 {fmt_len(sum(c.length_px for c in new_cracks), mmpp)}）")
        if longer:
            pct = f"，+{ch.grow_px / r.total_px:.0%}" if r.total_px > 0 else ""
            changes.append(f"裂缝延长 {fmt_len(ch.grow_px, mmpp)}{pct}")
        if wider:
            changes.append(f"最大宽度增加 {fmt_w(n.max_w_px - r.max_w_px, mmpp)}")
        if changes:
            level = "新增" if new_cracks and not (longer and r.total_px > 0) else \
                ("延长且变宽" if longer and wider else ("延长" if longer else ("变宽" if wider else "新增")))
            parts.append("变化：" + "；".join(changes) + "。")
        else:
            level = "稳定"
            parts.append("与上次相比未见明显发展。")
        if n.cracks and all(c.limited for c in n.cracks):
            lim = f"约 {fmt_w(RES_LIMIT_PX, mmpp)}"
            parts.append(f"注意：裂缝宽度接近照片分辨率极限（{lim}），实际宽度可能更细；需要测细裂缝宽度时请靠近拍摄。")
        if ch.lost_px >= rules.grow_min_px:
            parts.append(f"有 {fmt_len(ch.lost_px, mmpp)} 上次的裂缝这次未检出（可能已修补、被遮挡或拍摄不清）。")
        if mmpp and n.max_w_px * mmpp >= rules.attention_mm and not all(c.limited for c in n.cracks):
            parts.append(f"最大宽度已达 {fmt_w(n.max_w_px, mmpp)}，超过 {rules.attention_mm} mm，建议重点关注。")
        elif level not in ("稳定",):
            parts.append("建议持续观察，缩短复拍间隔。")
    if light_regions > 0:
        parts.append(f"另有 {light_regions} 处明暗变化（占 {light_ratio:.1%}，可能为渗水、水渍或阴影），请现场确认。")
    if not mmpp:
        parts.append("（参考照片中没有可用的二维码，尺寸以像素表示。）")
    return level, "".join(parts)


def draw_cracks(img_bgr: np.ndarray, ch: CrackChange, mmpp: Optional[float]) -> np.ndarray:
    """在复拍（已对齐）图上画出裂缝：原有部分黄色，新增部分红色，并标注编号、长度、最大宽度。"""
    out = img_bgr.copy()
    k = np.ones((3, 3), np.uint8)
    old = cv2.dilate(((ch.new.skel > 0) & (ch.grow_skel == 0)).astype(np.uint8), k) > 0
    new = cv2.dilate(ch.grow_skel, k) > 0
    lost = cv2.dilate(((ch.ref.skel > 0) & ~(cv2.dilate(ch.new.mask, np.ones((9, 9), np.uint8)) > 0)).astype(np.uint8), k) > 0
    out[lost] = (255, 160, 60)
    out[old] = (0, 220, 255)
    out[new] = (40, 40, 255)
    for i, c in enumerate(ch.new.cracks):
        x, y, w, h = c.bbox
        label = f"#{i + 1} L{fmt_len(c.length_px, mmpp)} W{'<=' if c.limited else ''}{fmt_w(c.max_w_px, mmpp)}"
        fs = max(out.shape[:2]) / 1300.0
        org = (int(x), int(max(y - 12, 30 * fs)))
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(int(6 * fs), 3), cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(int(2 * fs), 1), cv2.LINE_AA)
    return out

"""
小场景病害检测基元（与 App 的 Defects.kt 同一套算法和阈值）。输入都是已对齐到参考坐标的两张照片。

  面状 areas()        剥落/掉块、锈蚀、渗水/湿痕、泛碱/白化：颜色和纹理相对参考的变化
  位移 displacement() 伸缩缝张开、错动/错台、支座位移：分块相位相关，找“整块一起移动”的区域
  标记线 marks()      螺栓防松标记线：每条标记的方向是否变了、是否断了
  有无 parts()        部件缺失/新增：结构变化区域里，原来的东西是不是没了

手机上只给大致判断；精细分析请导出档案后在电脑上处理。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

LEVELS = ["无变化", "轻微", "明显", "建议重点关注"]
F = 4                         # 面状检测在 1/4 分辨率上做


def _design(h, w, lref):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xs, ys = xx / w - 0.5, yy / h - 0.5
    return np.stack([np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys, lref], -1)


def _mad(v):
    return 1.4826 * float(np.median(np.abs(v - np.median(v)))) + 1e-6


def _local_ncc(a, b, k=7):
    box = lambda x: cv2.boxFilter(x, -1, (k, k))
    ma, mb = box(a), box(b)
    va = box(a * a) - ma * ma; vb = box(b * b) - mb * mb
    cov = box(a * b) - ma * mb
    return cov / np.sqrt(np.maximum(va, 1e-3) * np.maximum(vb, 1e-3)), np.sqrt(np.maximum(va, 0)), np.sqrt(np.maximum(vb, 0))


@dataclass
class Area:
    cls: str
    box: tuple            # x, y, w, h（参考工作坐标）
    px: float
    mm2: float | None
    dL: float
    contour: np.ndarray = field(repr=False, default=None)


def areas(ref, al, valid_mask, change, mmpp_at=None, debug=None):
    """ref/al: BGR 工作图；valid_mask: 255 = 参与比较；change: fa.compare 的变化图（64 = 反光）。"""
    h0, w0 = ref.shape[:2]
    sz = (w0 // F, h0 // F)
    lab = lambda im: cv2.cvtColor(cv2.resize(im, sz, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
    A, B = lab(ref), lab(al)
    m = cv2.resize(((valid_mask > 0) & (change != 64)).astype(np.uint8) * 255, sz, interpolation=cv2.INTER_AREA) > 250
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    h, w = m.shape
    X = _design(h, w, A[..., 0] / 255)
    idx = np.flatnonzero(m.ravel())[::2]
    res = []
    for ch in range(3):
        d = (B[..., ch] - A[..., ch]).ravel()
        Xi = X.reshape(-1, 7)[idx]
        cc, *_ = np.linalg.lstsq(Xi, d[idx], rcond=None)
        keep = np.ones(len(idx), bool)
        for _ in range(3):
            cc, *_ = np.linalg.lstsq(Xi[keep], d[idx][keep], rcond=None)
            r = d[idx] - Xi @ cc
            keep = np.abs(r) < 2.5 * _mad(r[keep])
        res.append((d - X.reshape(-1, 7) @ cc).reshape(h, w))
    rL, ra, rb = res
    sL, sa, sb = (_mad(r[m]) for r in res)
    chroma_new = np.hypot(B[..., 1] - 128, B[..., 2] - 128)
    ncc, sdA, sdB = _local_ncc(A[..., 0], B[..., 0])
    tL = max(10.0, 4 * sL)
    dark = rL < -tL
    white = (rL > tL) & (chroma_new < 20)
    rusty = (ra > max(5.0, 4 * sa)) & (rb > max(4.0, 3 * sb)) & (B[..., 1] - 128 > 6)
    tex = (ncc < 0.2) & (sdA > 3) & (np.abs(rL) > 0.5 * tL)          # 纹理被替换（剥落、涂层脱落、新物体）
    u = ((dark | white | rusty | tex) & m).astype(np.uint8) * 255
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    u = cv2.morphologyEx(u, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(u, connectivity=8)
    dist = cv2.distanceTransform(u, cv2.DIST_L2, 3)
    if debug is not None:
        debug.update(sL=sL, sa=sa, sb=sb, ncc_med=float(np.median(ncc[m])), dark=int(dark[m].sum()), white=int(white[m].sum()),
                     rusty=int(rusty[m].sum()), tex=int(tex[m].sum()))
    out = []
    for i in range(1, n):
        x, y, bw, bh, npx = st[i]
        comp = lab_ == i
        if dist[comp].max() < 2.0 or min(bw, bh) < 0.15 * max(bw, bh):
            continue                                                        # 细线、细长条（裂缝、缝）交给线状/位移基元
        cx, cy = (x + bw / 2) * F, (y + bh / 2) * F
        mmpp = mmpp_at((cx, cy)) if mmpp_at else None
        area_px = npx * F * F
        mm2 = area_px * mmpp * mmpp if mmpp else None
        if (mm2 if mm2 is not None else area_px / 400) < 100:              # < 1 cm²（无比例时 < 2000 像素）不报
            continue
        f = lambda k: float(k[comp].mean())
        if f(rusty) >= 0.3:
            cls = "锈蚀"
        elif f(white) >= 0.4:
            cls = "泛碱/白化"
        elif f(tex) >= 0.25:
            cls = "剥落/掉块" if float(rL[comp].mean()) < 0 else "表面缺损"
        elif f(dark) >= 0.4:
            cls = "渗水/湿痕"
        else:
            cls = "表面变化"
        cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out.append(Area(cls, (x * F, y * F, bw * F, bh * F), area_px, mm2, float(rL[comp].mean()),
                        max(cs, key=cv2.contourArea).reshape(-1, 2) * F))
    return out


@dataclass
class Disp:
    dx: float; dy: float              # 像素（参考坐标），移动一侧相对不动一侧
    mm: float | None
    opening: float | None             # 沿缝法向（正 = 张开）
    shear: float | None               # 沿缝方向（错动）
    blocks: int
    box: tuple
    cells: list = field(repr=False, default=None)
    cdx: float = 0.0; cdy: float = 0.0   # 搬回原位用的实际偏移


def displacement(rg, ng, mask, mmpp_at=None, B=128, step=64, debug=None):
    a = rg.astype(np.float32); b = ng.astype(np.float32)
    a -= cv2.GaussianBlur(a, (0, 0), 8); b -= cv2.GaussianBlur(b, (0, 0), 8)
    win = cv2.createHanningWindow((B, B), cv2.CV_32F)
    H, W = a.shape
    cells = {}
    for gy, y in enumerate(range(0, H - B + 1, step)):
        for gx, x in enumerate(range(0, W - B + 1, step)):
            if (mask[y:y + B, x:x + B] > 0).mean() < 0.95:
                continue
            pa, pb = a[y:y + B, x:x + B], b[y:y + B, x:x + B]
            if pa.std() < 4 or pb.std() < 4:
                continue
            (dx, dy), r = cv2.phaseCorrelate(pa, pb, win)
            if r < 0.08:
                continue
            cells[(gx, gy)] = (x + B / 2, y + B / 2, dx, dy, r)
    if len(cells) < 8:
        return None
    keys = list(cells)
    P_ = np.array([[cells[k][0], cells[k][1]] for k in keys]); V = np.array([[cells[k][2], cells[k][3]] for k in keys])
    u_, v_ = (P_[:, 0] - W / 2) / W, (P_[:, 1] - H / 2) / H
    U = np.stack([np.ones(len(keys)), u_, v_, u_ * u_, v_ * v_, u_ * v_], 1)          # 二次位移场
    # 不动的平面：RANSAC 拟合二次位移场（吸收对齐时的小偏差），残差大的块才算“移动”
    rng = np.random.default_rng(1)
    best_in, tol = None, 0.6
    for _ in range(500):
        s3 = rng.choice(len(keys), 6, replace=False)
        try:
            C = np.linalg.solve(U[s3], V[s3])
        except np.linalg.LinAlgError:
            continue
        inl = np.hypot(*(V - U @ C).T) < tol
        if best_in is None or inl.sum() > best_in.sum():
            best_in = inl
    C, *_ = np.linalg.lstsq(U[best_in], V[best_in], rcond=None)
    R = V - U @ C
    mag = np.hypot(*R.T)
    thr = max(0.8, 3 * _mad(mag[best_in]))
    if debug is not None:
        debug.update(cells=len(cells), inliers=int(best_in.sum()), thr=round(thr, 2))
    res_ = {k: R[i] for i, k in enumerate(keys)}
    med = np.zeros(2)
    moved = {k for k, mg in zip(keys, mag) if mg >= thr}
    seen, best = set(), []
    for k in moved:
        if k in seen:
            continue
        comp, q = [], [k]; seen.add(k)
        while q:
            c = q.pop(); comp.append(c)
            vc = res_[c]
            for dx_ in (-1, 0, 1):
                for dy_ in (-1, 0, 1):
                    nb = (c[0] + dx_, c[1] + dy_)
                    if nb in moved and nb not in seen and np.hypot(*(res_[nb] - vc)) < max(1.0, 0.35 * np.hypot(*vc)):
                        seen.add(nb); q.append(nb)
        if len(comp) > len(best):
            best = comp
    if len(best) < 4:
        return None
    vv = np.median([res_[k] for k in best], 0)                       # 相对不动平面的位移
    raw = np.median([cells[k][2:4] for k in best], 0)                 # 对齐图里的实际偏移（用于搬回原位）
    cxy = np.mean([cells[k][:2] for k in best], 0)
    # 缝两侧：离缝 2~5 个网格的块（紧贴缝的块同时含两侧纹理，不用），各自拟合平面，在缝的位置求差
    bs = set(best)

    def ring(k, other):
        for d in range(1, 6):
            for i in range(-d, d + 1):
                for j in range(-d, d + 1):
                    if (abs(i) == d or abs(j) == d) and other((k[0] + i, k[1] + j)):
                        return d
        return 99
    d_stat = {k: ring(k, lambda n: n in bs) for k in keys if k not in moved}
    d_mov = {k: ring(k, lambda n: n in cells and n not in moved) for k in best}
    stat_adj = [k for k, d in d_stat.items() if 2 <= d <= 5]; mov_adj = [k for k, d in d_mov.items() if 2 <= d <= 5]
    if len(stat_adj) < 3 or len(mov_adj) < 3:
        stat_adj = [k for k, d in d_stat.items() if d <= 5]; mov_adj = [k for k, d in d_mov.items() if d <= 5]
    opening = shear = None
    if len(stat_adj) >= 3 and len(mov_adj) >= 3:
        s_ = np.mean([cells[k][:2] for k in stat_adj], 0); m_ = np.mean([cells[k][:2] for k in mov_adj], 0)
        pb = (s_ + m_) / 2

        def plane_at(ks):
            A_ = np.array([[1, (cells[k][0] - pb[0]) / W, (cells[k][1] - pb[1]) / H] for k in ks])
            Y_ = np.array([cells[k][2:4] for k in ks])
            c_, *_ = np.linalg.lstsq(A_, Y_, rcond=None)
            return c_[0]
        vv = plane_at(mov_adj) - plane_at(stat_adj)
        nvec = (m_ - s_) / (np.linalg.norm(m_ - s_) + 1e-9)
        edge = [k for k, d in d_mov.items() if d == 1]                    # 缝的走向：紧贴缝的移动块的主方向
        if len(edge) >= 3:
            _, _, t_ = _pca(np.array([cells[k][0] for k in edge], float), np.array([cells[k][1] for k in edge], float))
            e_ = np.array([-t_[1], t_[0]])
            nvec = e_ if e_ @ nvec >= 0 else -e_
        opening = float(vv @ nvec); shear = float(abs(vv[0] * nvec[1] - vv[1] * nvec[0]))
    mmpp = mmpp_at(tuple(cxy)) if mmpp_at else None
    xs = [cells[k][0] for k in best]; ys = [cells[k][1] for k in best]
    box = (int(min(xs) - B / 2), int(min(ys) - B / 2), int(max(xs) - min(xs) + B), int(max(ys) - min(ys) + B))
    k_ = mmpp or 1.0
    return Disp(float(vv[0]), float(vv[1]), float(np.hypot(*vv)) * mmpp if mmpp else None,
                opening * k_ if opening is not None else None, shear * k_ if shear is not None else None, len(best), box,
                [cells[k] for k in best], float(raw[0]), float(raw[1]))


def compensate(al, d: Disp):
    """把整块移动的区域按位移搬回原位，之后的面状/部件分析就不会把“整体移动”误判为表面变化。"""
    if d is None or not d.cells:
        return al
    region = np.zeros(al.shape[:2], np.uint8)
    half = 64
    for (x, y, *_r) in d.cells:
        region[int(y - half):int(y + half), int(x - half):int(x + half)] = 255
    M = np.float32([[1, 0, -d.cdx], [0, 1, -d.cdy]])
    shifted = cv2.warpAffine(al, M, (al.shape[1], al.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    out = al.copy()
    out[region > 0] = shifted[region > 0]
    return out


@dataclass
class Mark:
    cx: float; cy: float; ang: float; length: float
    status: str           # 正常 / 疑似松动 / 标记残缺 / 标记缺失
    dang: float
    coverage: float


def _mark_mask(bgr, mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mm_ = ((hsv[..., 1] >= 100) & (hsv[..., 2] >= 70) & (mask > 0)).astype(np.uint8) * 255
    return cv2.morphologyEx(mm_, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _pca(xs, ys):
    c = np.cov(np.stack([xs, ys]))
    ev, evec = np.linalg.eigh(c)
    u = evec[:, 1]
    return np.sqrt(12 * max(ev[1], 0)), np.sqrt(12 * max(ev[0], 0)), u


def marks(ref, al, mask):
    mr, mn = _mark_mask(ref, mask), _mark_mask(al, mask)
    n, lab, st, cen = cv2.connectedComponentsWithStats(mr, connectivity=8)
    out = []
    for i in range(1, n):
        if st[i, 4] < 40:
            continue
        ys, xs = np.nonzero(lab == i)
        L, wd, u = _pca(xs.astype(np.float64), ys.astype(np.float64))
        if L < 25 or L < 3 * max(wd, 1):
            continue
        c = np.array([xs.mean(), ys.mean()])
        ang = np.degrees(np.arctan2(u[1], u[0])) % 180
        R = L / 2 + 12
        x0, y0 = int(max(c[0] - R, 0)), int(max(c[1] - R, 0)); x1, y1 = int(c[0] + R) + 1, int(c[1] + R) + 1
        crop = mn[y0:y1, x0:x1].copy()
        yy, xx = np.mgrid[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]]
        crop[np.hypot(xx - c[0], yy - c[1]) > R] = 0
        nn = int((crop > 0).sum())
        if nn < 0.25 * st[i, 4]:
            out.append(Mark(c[0], c[1], ang, L, "标记缺失", 0.0, nn / st[i, 4])); continue
        # 新照片里落在原标记线段上的比例
        ny, nx = np.nonzero(crop)
        p = np.stack([nx + x0, ny + y0], 1) - c
        along = p @ u; perp = np.abs(p[:, 0] * u[1] - p[:, 1] * u[0])
        on = (perp <= max(3.0, wd)) & (np.abs(along) <= L / 2 + 3)
        cov = min(on.sum() / st[i, 4], 1.0)
        # 标记线沿原方向分成两半，各自拟合方向：螺母那一半转了、钢板那一半没转（小角度时两段仍连在一起）
        dmax = 0.0
        for half in (along < 0, along >= 0):
            if half.sum() < max(12, 0.12 * st[i, 4]):
                continue
            Lh, wh, uh = _pca(p[half, 0], p[half, 1])
            if Lh < 10 or Lh < 2 * max(wh, 1):
                continue
            ah = np.degrees(np.arctan2(uh[1], uh[0])) % 180
            dmax = max(dmax, min(abs(ah - ang), 180 - abs(ah - ang)))
        dev = float(np.percentile(perp, 95)) - wd / 2                        # 偏离原标记线的横向距离
        loose = dmax >= 6 or dev > max(5.0, 0.12 * L)
        status = "疑似松动" if loose else ("标记残缺" if cov < 0.7 else "正常")
        out.append(Mark(c[0], c[1], ang, L, status, dmax, cov))
    return out


@dataclass
class Part:
    box: tuple
    status: str           # 疑似缺失 / 新增物体 / 外观改变
    ncc: float
    e_ratio: float
    d_mean: float


def parts(rg, ng, change, mask, min_px=300):
    s = ((change == 255) & (mask > 0)).astype(np.uint8) * 255
    s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(s, connectivity=8)
    dist = cv2.distanceTransform(s, cv2.DIST_L2, 3)
    a = rg.astype(np.float32); b = ng.astype(np.float32)
    ga = cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    gb = cv2.magnitude(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1))
    H, W = rg.shape
    out = []
    for i in range(1, n):
        x, y, w, h, npx = st[i]
        comp = lab == i
        if npx < min_px or dist[comp].max() < 4:
            continue
        p = int(0.3 * max(w, h)) + 6
        X0, Y0, X1, Y1 = max(x - p, 0), max(y - p, 0), min(x + w + p, W), min(y + h + p, H)
        inner = np.zeros((Y1 - Y0, X1 - X0), bool); inner[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = True
        ring = ~inner
        pa, pb = a[Y0:Y1, X0:X1], b[Y0:Y1, X0:X1]
        za = (pa - pa.mean()) / (pa.std() + 1e-6); zb = (pb - pb.mean()) / (pb.std() + 1e-6)
        ncc = float((za[inner] * zb[inner]).mean())
        if ncc > 0.6:
            continue
        ea, eb = float(ga[Y0:Y1, X0:X1][inner].mean()), float(gb[Y0:Y1, X0:X1][inner].mean())
        er = eb / (ea + 1e-6) / ((gb[Y0:Y1, X0:X1][ring].mean() + 1e-6) / (ga[Y0:Y1, X0:X1][ring].mean() + 1e-6))
        dm = float((pb[inner].mean() - pb[ring].mean()) - (pa[inner].mean() - pa[ring].mean()))
        status = "疑似缺失" if (er < 0.65 or dm < -25) else ("新增物体" if er > 1.5 or dm > 25 else "外观改变")
        out.append(Part((int(x), int(y), int(w), int(h)), status, ncc, er, dm))
    return out

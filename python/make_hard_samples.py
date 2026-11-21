"""
更难的合成样例：弯曲裂缝（蜿蜒、分叉、粗细渐变、网状）和剥落（浅色新断面、小块、露筋、阴影干扰）。
和 make_defect_samples.py 一样：前后两张换角度、换光线、偏色。

  ../.venv/bin/python make_hard_samples.py     → samples/hard/<用例>_ref.jpg / _new.jpg / truth.json

真值单位都是毫米（平面 8 px/mm）。
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_defect_samples as mds  # noqa: E402
import synth_test as st  # noqa: E402

OUT = os.path.join(st.OUT, "hard")
PX = st.PX_PER_MM
W, H = st.PLANE
mm = mds.mm


# ------------------------------------------------------------------ 弯曲裂缝
def curve(seed, start, heading, length_mm, wiggle=0.10, step=1.5):
    """平滑随机游走的裂缝中线（mm）。wiggle：每步转角的标准差（弧度），会被平滑，所以是缓慢弯曲 + 小锯齿。"""
    rng = np.random.default_rng(seed)
    n = int(length_mm / step)
    turn = np.convolve(rng.normal(0, wiggle, n + 20), np.ones(9) / 9 * 2.2, "same")[10:n + 10]
    jag = rng.normal(0, 0.25, n)                                 # 裂缝边缘的小锯齿（mm）
    a = heading + np.cumsum(turn)
    p = np.array(start, float) + np.cumsum(np.stack([np.cos(a), np.sin(a)], 1) * step, 0)
    nrm = np.stack([-np.sin(a), np.cos(a)], 1)
    return p + nrm * jag[:, None]


def length_mm(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def draw_curve(alpha, pts, w0, w1, upto=1.0):
    """沿中线画裂缝，宽度从 w0 渐变到 w1（mm）；只画前 upto 比例。alpha 为 0..1 浮点“暗度”图。"""
    n = max(2, int(len(pts) * upto))
    for i in range(n - 1):
        t = i / max(len(pts) - 1, 1)
        wpx = max(1, int(round((w0 + (w1 - w0) * t) * PX)))
        a, b = (int(round(pts[i][0] * PX)), int(round(pts[i][1] * PX))), (int(round(pts[i + 1][0] * PX)), int(round(pts[i + 1][1] * PX)))
        cv2.line(alpha, a, b, 1.0, wpx, cv2.LINE_AA)


def crack_plane(base_bgr, segs, depth=0.72):
    """segs: [(pts, w0, w1, upto)]。裂缝 = 变暗（深处更暗）。"""
    a = np.zeros((H, W), np.float32)
    for pts, w0, w1, upto in segs:
        draw_curve(a, pts, w0, w1, upto)
    a = cv2.GaussianBlur(a, (0, 0), 0.6)
    return base_bgr * (1 - depth * np.clip(a, 0, 1))[..., None]


def avoid_target(pts):
    """裂缝不要穿过二维码和圆点（它们会被排除在比较区外，真值就不准了）。返回是否安全。"""
    qx, qy = st.QR_POS_MM; q = st.QR_MM
    for x, y in pts:
        if qx - 14 < x < qx + q + 14 and qy - 14 < y < qy + q + 14:
            return False
        if any((x - dx) ** 2 + (y - dy) ** 2 < 22 ** 2 for dx, dy in st.DOTS_MM):
            return False
        if not (12 < x < W / PX - 12 and 12 < y < H / PX - 12):
            return False
    return True


def safe_curve(seed, start, heading, L, wiggle):
    for k in range(200):
        c = curve(seed + 1000 * k, start, heading, L, wiggle)
        if avoid_target(c):
            return c
    raise RuntimeError("找不到避开靶标的裂缝")


# ------------------------------------------------------------------ 剥落
def spall_light(p_bgr, m, seed):
    """新鲜断面：比原表面更亮、更粗糙（骨料外露），边缘有一圈断口阴影。"""
    rough = st.texture(seed, 2.6).astype(np.float32) * 0.55 + 95
    rng = np.random.default_rng(seed)
    for _ in range(300):
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.circle(rough, (int(x), int(y)), int(rng.integers(5, 18)), float(rng.uniform(120, 215)), -1)
    edge = cv2.GaussianBlur(m, (0, 0), 4) - cv2.GaussianBlur(cv2.erode(m, np.ones((11, 11), np.uint8)), (0, 0), 4)
    a = cv2.GaussianBlur(m, (0, 0), 1.2)[..., None]
    out = p_bgr * (1 - a) + mds.to_bgr(np.clip(rough, 0, 255), np.array([0.98, 1.0, 1.02])) * a
    return out * (1 - 0.5 * np.clip(edge, 0, 1))[..., None], a[..., 0]


def spall_rebar(p_bgr, m, seed):
    """剥落露筋：暗粗糙断面 + 一根锈色钢筋横穿。"""
    out, a = mds.spall(p_bgr, m, seed)
    ys, xs = np.nonzero(m > 0.5)
    y0 = int(ys.mean()); x0, x1 = int(xs.min()), int(xs.max())
    bar = np.zeros((H, W), np.float32)
    cv2.line(bar, (x0 - 40, y0), (x1 + 40, y0 + 10), 1.0, mm(10), cv2.LINE_AA)
    bar = cv2.GaussianBlur(bar * m, (0, 0), 1.5)[..., None]
    out = out * (1 - bar) + np.array([40.0, 70.0, 130.0]) * bar
    return out, a


def soft_shadow(p_bgr, cx, cy, r, k=0.45):
    """大片柔和阴影（构件、人、树影），只是变暗、边缘很软。"""
    s = np.zeros((H, W), np.float32)
    cv2.ellipse(s, (mm(cx), mm(cy)), (mm(r), mm(r * 0.6)), 20, 0, 360, 1.0, -1)
    s = cv2.GaussianBlur(s, (0, 0), mm(14))[..., None]
    return p_bgr * (1 - k * s)


def stain(p_bgr, m):
    """光滑的深色污渍（不是剥落）：纹理保留，只是变暗、略偏色。"""
    a = cv2.GaussianBlur(m, (0, 0), 3)[..., None]
    return p_bgr * (1 - 0.45 * a) * np.array([0.98, 1.0, 1.04]) ** a


# ------------------------------------------------------------------ 主程序
def main():
    os.makedirs(OUT, exist_ok=True)
    H_ref = st.camera_h(100, tilt=0.05, rot_deg=5)
    truth = {}

    def save(name, ref_plane, new_plane, t, seed, warm=(0.9, 1.0, 1.12), gain=0.9):
        Hn = st.camera_h(seed, tilt=0.11, rot_deg=16, zoom=0.88)
        cv2.imwrite(os.path.join(OUT, f"{name}_ref.jpg"), mds.render(ref_plane, H_ref, 100))
        cv2.imwrite(os.path.join(OUT, f"{name}_new.jpg"), mds.render(new_plane, Hn, seed, light=0.35, blur=1.0, tint=warm, gain=gain))
        truth[name] = t
        print(f"{name:<20}", {k: v for k, v in t.items()})

    p0 = mds.to_bgr(mds.base(1))

    # 1 蜿蜒裂缝延长：上次画 55%，这次全长，并多出一条分叉
    c1 = safe_curve(11, (40, 255), -0.55, 300, 0.10)
    br = safe_curve(12, tuple(c1[int(len(c1) * 0.30)]), -0.55 + 1.1, 60, 0.12)
    ref = crack_plane(p0, [(c1, 0.55, 0.25, 0.55)])
    new = crack_plane(p0, [(c1, 0.55, 0.25, 1.0), (br, 0.35, 0.2, 1.0)])
    grow = length_mm(c1[int(len(c1) * 0.55):]) + length_mm(br)
    save("curve_grow", ref, new, {"kind": "crack_grow", "grow_mm": round(grow, 1), "total_mm": round(length_mm(c1) + length_mm(br), 1)}, 701)

    # 2 蜿蜒裂缝无变化（光线、清晰度都不同）
    c2 = safe_curve(21, (30, 60), 0.35, 330, 0.13)
    pc = crack_plane(p0, [(c2, 0.5, 0.3, 1.0)])
    save("curve_nochange", pc, pc, {"kind": "crack_same", "total_mm": round(length_mm(c2), 1)}, 702)

    # 3 蜿蜒裂缝变宽（长度不变，宽度 ×2.2）
    c3 = safe_curve(31, (35, 220), -0.3, 280, 0.09)
    save("curve_widen", crack_plane(p0, [(c3, 0.30, 0.25, 1.0)]), crack_plane(p0, [(c3, 0.70, 0.55, 1.0)]),
         {"kind": "crack_widen", "w_ref_mm": 0.3, "w_new_mm": 0.7, "total_mm": round(length_mm(c3), 1)}, 703)

    # 4 网状（龟裂）无变化：主裂缝 + 多条分叉
    segs = []
    main_c = safe_curve(41, (25, 200), -0.15, 340, 0.11)
    segs.append((main_c, 0.45, 0.3, 1.0))
    for k, (f, d) in enumerate(((0.18, 1.3), (0.4, -1.4), (0.62, 1.2), (0.8, -1.2))):
        b = safe_curve(42 + k, tuple(main_c[int(len(main_c) * f)]), -0.15 + d, 70, 0.14)
        segs.append((b, 0.3, 0.18, 1.0))
    pm = crack_plane(p0, segs)
    save("map_nochange", pm, pm, {"kind": "crack_same", "total_mm": round(sum(length_mm(s[0]) for s in segs), 1)}, 704)

    # 5 细而淡的裂缝延长（0.3 mm、颜色浅，约 1.3 像素宽，接近分辨率极限）
    c5 = safe_curve(51, (40, 250), -0.4, 260, 0.10)
    save("hairline_grow", crack_plane(p0, [(c5, 0.3, 0.25, 0.5)], 0.5), crack_plane(p0, [(c5, 0.3, 0.25, 1.0)], 0.5),
         {"kind": "crack_grow", "grow_mm": round(length_mm(c5[int(len(c5) * 0.5):]), 1), "total_mm": round(length_mm(c5), 1)}, 705)

    # 6 裂缝不变 + 新出现一片污渍（裂缝应判稳定，污渍报渗水/湿痕而不是剥落）
    c6 = safe_curve(61, (30, 70), 0.3, 300, 0.1)
    pc6 = crack_plane(p0, [(c6, 0.45, 0.3, 1.0)])
    m6 = mds.blob(310, 215, 38, 26, 606)
    save("curve_stain", pc6, stain(pc6, m6), {"kind": "crack_same", "total_mm": round(length_mm(c6), 1),
                                                "area_cls": "渗水/湿痕", "area_mm2": round(float((m6 > 0.5).sum()) / PX ** 2, 1)}, 706)

    # 7 浅色新断面剥落（不能当成泛碱）
    m7 = mds.blob(300, 205, 32, 22, 707)
    n7, a7 = spall_light(p0, m7, 707)
    save("spall_light", p0, n7, {"kind": "spall", "area_mm2": round(float((a7 >= 0.5).sum()) / PX ** 2, 1)}, 707)

    # 8 小块剥落（约 2 cm²）
    m8 = mds.blob(310, 200, 9, 7, 708, rough=0.2)
    n8, a8 = mds.spall(p0, m8, 708)
    save("spall_small", p0, n8, {"kind": "spall", "area_mm2": round(float((a8 >= 0.5).sum()) / PX ** 2, 1)}, 708)

    # 9 剥落露筋
    m9 = mds.blob(295, 200, 40, 24, 709)
    n9, a9 = spall_rebar(p0, m9, 709)
    save("spall_rebar", p0, n9, {"kind": "spall", "area_mm2": round(float((a9 >= 0.5).sum()) / PX ** 2, 1)}, 709)

    # 10 剥落 + 另一处大片柔和阴影（阴影不应报病害）
    m10 = mds.blob(310, 205, 30, 22, 710)
    n10, a10 = mds.spall(p0, m10, 710)
    save("spall_shadow", p0, soft_shadow(n10, 90, 170, 60), {"kind": "spall", "area_mm2": round(float((a10 >= 0.5).sum()) / PX ** 2, 1)}, 710)

    # 11 有裂缝的墙面上剥落：剥落面积要报，裂缝稳定
    c11 = safe_curve(71, (30, 60), 0.32, 250, 0.1)
    p11 = crack_plane(p0, [(c11, 0.45, 0.3, 1.0)])
    m11 = mds.blob(300, 215, 30, 20, 711)
    n11, a11 = mds.spall(p11, m11, 711)
    save("spall_on_crack", p11, n11, {"kind": "spall", "area_mm2": round(float((a11 >= 0.5).sum()) / PX ** 2, 1),
                                      "crack": "same"}, 711)

    with open(os.path.join(OUT, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

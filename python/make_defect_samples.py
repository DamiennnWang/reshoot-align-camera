"""
小场景病害合成样例：在“混凝土 / 钢板”平面上造前后两张照片（换角度、换光线、偏色），
覆盖面状（剥落、锈蚀、渗水、泛碱）、位移（错台、接缝张开）、螺栓（标记线松动、缺失）和无变化对照。

  ../.venv/bin/python make_defect_samples.py     → samples/defects/<用例>_ref.jpg / _new.jpg / truth.json

真值单位都是毫米（平面 8 px/mm）。
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_test as st  # noqa: E402

OUT = os.path.join(st.OUT, "defects")
PX = st.PX_PER_MM
W, H = st.PLANE
SHOT = (3000, 2250)                       # 输出照片尺寸
CONCRETE = np.array([0.97, 1.0, 1.03])    # BGR 偏暖灰


def mm(v):
    return int(round(v * PX))


def to_bgr(gray, tint=CONCRETE):
    return np.clip(gray[..., None].astype(np.float32) * tint, 0, 255)


def blob(cx, cy, rx, ry, seed, rough=0.25):
    """不规则斑块掩膜（0..1 浮点），中心和半径单位 mm。"""
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    r = 1 + rough * np.convolve(rng.normal(0, 1, 64 + 6), np.ones(7) / 7, "valid")[:64] * 2
    pts = np.stack([mm(cx) + np.cos(a) * mm(rx) * r, mm(cy) + np.sin(a) * mm(ry) * r], 1).astype(np.int32)
    m = np.zeros((H, W), np.float32)
    cv2.fillPoly(m, [pts], 1.0, cv2.LINE_AA)
    return m


def base(seed=1, dots=st.DOTS_MM, strength=1.0):
    p = st.texture(seed, strength)
    st.paste_target(p, "B01-Z023", dots)
    return p


def render(plane_bgr, Hc, seed, light=0.25, blur=0.8, tint=(1.0, 1.0, 1.0), gain=1.0):
    """彩色版 synth_test.render：透视 + 光照梯度 + gamma + 偏色 + 噪声 + JPEG。"""
    rng = np.random.default_rng(seed)
    PW, PH = st.PHOTO
    out = np.empty((PH, PW, 3), np.float32)
    for c in range(3):
        out[..., c] = cv2.warpPerspective(np.ascontiguousarray(plane_bgr[..., c]), Hc, (PW, PH), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=70)
    yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
    g = 1 + light * ((xx / PW - 0.5) * rng.uniform(-1, 1) + (yy / PH - 0.5) * rng.uniform(-1, 1))
    gamma = rng.uniform(0.85, 1.2)
    f = 255 * (np.clip(out, 0, 255) / 255) ** gamma * g[..., None] * gain * np.array(tint)
    f = cv2.GaussianBlur(f, (0, 0), blur) + rng.normal(0, 3, f.shape)
    f = cv2.resize(np.clip(f, 0, 255).astype(np.uint8), SHOT, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ 面状
def spall(p_bgr, m, seed):
    """剥落：斑块内换成更粗糙、更暗的骨料面，边缘一圈阴影。"""
    rough = st.texture(seed, 2.4).astype(np.float32) * 0.72
    rng = np.random.default_rng(seed)
    for _ in range(260):
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.circle(rough, (int(x), int(y)), int(rng.integers(6, 22)), float(rng.uniform(60, 170)), -1)
    edge = cv2.GaussianBlur(m, (0, 0), 6) - cv2.GaussianBlur(cv2.erode(m, np.ones((15, 15), np.uint8)), (0, 0), 6)
    a = cv2.GaussianBlur(m, (0, 0), 1.5)[..., None]
    out = p_bgr * (1 - a) + to_bgr(rough, np.array([0.95, 1.0, 1.04])) * a
    return out * (1 - 0.45 * np.clip(edge, 0, 1))[..., None], a[..., 0]


def rust(p_bgr, m, seed):
    rng = np.random.default_rng(seed)
    streak = cv2.GaussianBlur(m, (0, 0), 3)
    k = np.zeros((161, 9), np.float32); k[80:, 4] = 1; k /= k.sum()     # 向下流淌的锈水痕
    streak = np.maximum(m, cv2.filter2D(streak, -1, k) * 1.6)
    a = np.clip(cv2.GaussianBlur(streak, (0, 0), 2), 0, 1)[..., None] * 0.85
    noise = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 4)[..., None] * 25
    col = np.array([38.0, 78.0, 150.0]) + noise                          # BGR 红褐
    return p_bgr * (1 - a) + col * a, a[..., 0]


def seep(p_bgr, m, seed):
    a = cv2.GaussianBlur(m, (0, 0), 14)[..., None]
    return p_bgr * (1 - 0.40 * a) * np.array([1.03, 1.0, 0.97]) ** a, a[..., 0]      # 湿痕：变暗、略偏冷


def effl(p_bgr, m, seed):
    rng = np.random.default_rng(seed)
    a = np.clip(cv2.GaussianBlur(m, (0, 0), 6) * (0.75 + 0.25 * rng.random((H, W))), 0, 1)[..., None] * 0.75
    return p_bgr * (1 - a) + 232.0 * a, a[..., 0]


# ------------------------------------------------------------------ 位移
JOINT_X = 300.0                                   # 缝的位置（mm），右侧那块会移动
OFFSET_DOTS = [(40, 40), (250, 40), (80, 230)]    # 圆点都在左侧不动的那块上


def segments(dx_mm=0.0, dy_mm=0.0, gap_mm=2.0):
    p = st.texture(3).astype(np.float32)
    right = st.texture(4).astype(np.float32)            # 右侧另一块管片/梁体
    j = mm(JOINT_X)
    M = np.float32([[1, 0, mm(dx_mm)], [0, 1, mm(dy_mm)]])
    right = cv2.warpAffine(right, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    p[:, j:] = right[:, j:]
    g0 = j - mm(gap_mm / 2); g1 = j + mm(gap_mm / 2) + max(mm(dx_mm), 0)
    p[:, g0:g1] = 38                                     # 缝
    p = p.astype(np.uint8)
    st.paste_target(p, "B01-Z023", OFFSET_DOTS)
    return to_bgr(p)


# ------------------------------------------------------------------ 螺栓
BOLTS = [(110, 90, 30), (300, 115, 120), (120, 205, 200), (300, 210, 300)]     # 中心 mm, 标记线角度（度）
RED = np.array([40.0, 45.0, 205.0])


def hexagon(cx, cy, r, ang):
    a = np.deg2rad(ang + np.arange(6) * 60)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1).astype(np.int32)


def bolts(rot=None, missing=None):
    """钢板 + 4 个螺栓；每个螺栓画一条横跨螺母和钢板的红色防松标记线。
    rot={i: 度} 让第 i 个螺母（连同螺母上那段标记）转动；missing={i} 去掉第 i 个螺栓，只剩孔和钢板上那段标记。"""
    rot = rot or {}; missing = missing or set()
    plate = st.texture(9, 0.45).astype(np.float32) * 0.8 + 18
    p = plate.copy()
    st.paste_target(pu := p.astype(np.uint8), "B01-Z023", st.DOTS_MM)
    p = to_bgr(pu, np.array([1.02, 1.0, 0.98]))
    for i, (cx_mm, cy_mm, ang) in enumerate(BOLTS):
        cx, cy = mm(cx_mm), mm(cy_mm)
        L = mm(30); th = mm(2.6)
        e = (int(cx + L * np.cos(np.deg2rad(ang))), int(cy + L * np.sin(np.deg2rad(ang))))
        cv2.line(p, (cx, cy), e, RED.tolist(), th, cv2.LINE_AA)            # 钢板上的一段（螺母外）
        if i in missing:
            cv2.circle(p, (cx, cy), mm(16), (95, 95, 95), -1, cv2.LINE_AA)   # 垫片印痕
            cv2.circle(p, (cx, cy), mm(8), (28, 28, 30), -1, cv2.LINE_AA)    # 螺栓孔
            continue
        a2 = ang + rot.get(i, 0)
        nut = np.zeros((mm(40), mm(40), 3), np.float32); c0 = mm(20)
        cv2.circle(nut, (c0, c0), mm(17), (165, 165, 168), -1, cv2.LINE_AA)  # 垫片
        cv2.fillPoly(nut, [hexagon(c0, c0, mm(13), a2 + 15)], (128, 130, 134), cv2.LINE_AA)
        cv2.fillPoly(nut, [hexagon(c0, c0, mm(10), a2 + 15)], (150, 152, 156), cv2.LINE_AA)
        cv2.circle(nut, (c0, c0), mm(6), (175, 176, 178), -1, cv2.LINE_AA)   # 螺杆端头
        e2 = (int(c0 + mm(17) * np.cos(np.deg2rad(a2))), int(c0 + mm(17) * np.sin(np.deg2rad(a2))))
        cv2.line(nut, (c0, c0), e2, RED.tolist(), th, cv2.LINE_AA)           # 螺母和垫片上的一段（随螺母转动）
        msk = np.zeros((mm(40), mm(40)), np.float32); cv2.circle(msk, (c0, c0), mm(17), 1.0, -1, cv2.LINE_AA)
        reg = p[cy - c0:cy + c0, cx - c0:cx + c0]
        reg[:] = reg * (1 - msk[..., None]) + nut * msk[..., None]
        shadow = np.zeros((H, W), np.float32); cv2.circle(shadow, (cx + mm(2), cy + mm(2)), mm(18), 1.0, -1)
    return p


# ------------------------------------------------------------------ 主程序
def main():
    os.makedirs(OUT, exist_ok=True)
    H_ref = st.camera_h(100, tilt=0.05, rot_deg=5)
    truth = {}

    def save(name, ref_plane, new_plane, t, seed, warm=(0.9, 1.0, 1.12), gain=0.9):
        Hn = st.camera_h(seed, tilt=0.11, rot_deg=16, zoom=0.88)
        cv2.imwrite(os.path.join(OUT, f"{name}_ref.jpg"), render(ref_plane, H_ref, 100))
        cv2.imwrite(os.path.join(OUT, f"{name}_new.jpg"), render(new_plane, Hn, seed, light=0.35, blur=1.0, tint=warm, gain=gain))
        truth[name] = t
        print(f"{name:<16}", t)

    p0 = to_bgr(base(1))
    for name, fn, (cx, cy, rx, ry), seed, kind in (
            ("spall", spall, (300, 195, 34, 24), 501, "spall"),
            ("rust", rust, (85, 150, 22, 14), 502, "rust"),
            ("seep", seep, (300, 200, 45, 32), 503, "seep"),
            ("effl", effl, (100, 165, 30, 20), 504, "effl")):
        m = blob(cx, cy, rx, ry, seed)
        newp, alpha = fn(p0, m, seed)
        area = float((alpha >= 0.5 * alpha.max()).sum()) / PX ** 2          # 真值：实际画上去的范围（含锈水痕）
        save(name, p0, newp, {"kind": kind, "area_mm2": round(area, 1),
                                          "center_mm": [cx, cy]}, seed)
    save("plain_nochange", p0, p0, {"kind": "none"}, 601)

    save("offset", segments(), segments(dy_mm=3.0), {"kind": "offset", "disp_mm": 3.0, "dir": "dy"}, 511)
    save("joint", segments(), segments(dx_mm=2.0), {"kind": "joint", "disp_mm": 2.0, "dir": "dx"}, 512)
    save("segments_nochange", segments(), segments(), {"kind": "none"}, 602)
    save("joint_1mm", segments(), segments(dx_mm=1.0), {"kind": "joint", "disp_mm": 1.0, "dir": "dx"}, 513)   # 检测下限附近

    save("bolts", bolts(), bolts(rot={1: 25}, missing={3}), {"kind": "bolt", "loose": [1], "loose_deg": 25, "missing": [3],
                                                               "bolts_mm": [b[:2] for b in BOLTS]}, 521)
    save("bolts_nochange", bolts(), bolts(), {"kind": "none", "bolts_mm": [b[:2] for b in BOLTS]}, 603)
    save("bolts_10deg", bolts(), bolts(rot={2: 10}), {"kind": "bolt_small", "loose": [2], "loose_deg": 10,
                                                      "bolts_mm": [b[:2] for b in BOLTS]}, 522)                 # 检测下限附近
    with open(os.path.join(OUT, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

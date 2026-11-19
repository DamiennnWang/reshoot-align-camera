"""
宣传片渲染器：横版 1920×1080 与竖版 1080×1920 共用一套分镜（film_script.py）。

  ../.venv/bin/python make_film.py h            # 横版 → 宣传片_横版.mp4（画面 + 素材/音轨.wav，ffmpeg 一次编码）
  ../.venv/bin/python make_film.py v            # 竖版
  ../.venv/bin/python make_film.py h --preview  # 每个分镜抽几帧拼成一张图，检查用

画面里的对齐、特征匹配连线、变形、裂缝骨架、读数都来自 samples/promo 这对样板照片的真实处理结果；
手机画面是模拟器上 App 的真实截图。隧道、两次机位、扫码、手机展示是 Blender 三维镜头（blender/scene.py，
先运行 blender/render_all.sh），衬砌上的裂缝与二维码就是样板照片里那一块墙面。
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "python"))
import film_script as fs  # noqa: E402
import fieldalign as fa  # noqa: E402
import measure as ms  # noqa: E402

ORI = sys.argv[1] if len(sys.argv) > 1 else "h"
PREVIEW = "--preview" in sys.argv
PORT = ORI == "v"
W, H = (1080, 1920) if PORT else (1920, 1080)
FPS = 30
U = min(W, H) / 1080.0
SAMPLES = os.path.join(HERE, "..", "samples")
SHOTS = os.path.join(HERE, "素材")
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
INK = (22, 32, 42); AMBER = (255, 176, 32); GOOD = (34, 200, 120); RED = (240, 72, 62); WHITE = (255, 255, 255)
GREY = (170, 182, 195); CYAN = (110, 205, 255); YEL = (255, 220, 0)
rng = np.random.default_rng(11)


# ====================================================================== 基础工具
def ease(t):
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def eout(t):
    t = min(max(float(t), 0.0), 1.0)
    return 1 - (1 - t) ** 3


def px(v):
    return int(round(v * U))


_fonts, _spr = {}, {}


def font(size, bold=True):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT, size, index=2 if bold else 0)
    return _fonts[k]


def tsprite(s, size, fill=WHITE, bold=True, stroke=0, track=0.0):
    k = (s, size, fill, bold, stroke, track)
    if k not in _spr and track > 0:
        f = font(size, bold)
        d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
        adv = [d0.textlength(c, font=f) + size * track for c in s]
        l, t, r, b = d0.textbbox((0, 0), "国Ag", font=f)
        pad = 6 + stroke
        im = Image.new("RGBA", (int(sum(adv) - size * track) + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
        dd = ImageDraw.Draw(im); x = pad
        for c, a_ in zip(s, adv):
            dd.text((x, pad - t), c, font=f, fill=fill + (255,), stroke_width=stroke, stroke_fill=(0, 0, 0, 255)); x += a_
        _spr[k] = np.asarray(im).copy()
    if k not in _spr:
        f = font(size, bold)
        l, t, r, b = ImageDraw.Draw(Image.new("RGBA", (4, 4))).textbbox((0, 0), s, font=f, stroke_width=stroke)
        pad = 6 + stroke
        im = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((pad - l, pad - t), s, font=f, fill=fill + (255,), stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        _spr[k] = np.asarray(im).copy()
        if len(_spr) > 4000:
            _spr.clear()
    return _spr[k]


def pill_sprite(s, size, bg, fg=WHITE):
    k = ("pill", s, size, bg, fg)
    if k not in _spr:
        f = font(size, False)
        l, t, r, b = ImageDraw.Draw(Image.new("RGBA", (4, 4))).textbbox((0, 0), s, font=f)
        ph, pv = int(size * 0.8), int(size * 0.42)
        w, h = r - l + 2 * ph, b - t + 2 * pv
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        dark = sum(bg) < 160
        line, txt = ((96, 108, 122) if dark else bg), (fg if dark or fg != WHITE else bg)
        if fg == INK:
            txt = bg
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=(10, 14, 20, 200), outline=line + (255,), width=max(2, size // 14))
        d.text((ph - l, pv - t), s, font=f, fill=txt + (255,))
        _spr[k] = np.asarray(im).copy()
    return _spr[k]


def blit(fr, spr, cx, cy, alpha=1.0, scale=1.0, anchor="c"):
    """把 RGBA 小图合成到画面上。anchor: c 居中 / l 左对齐（纵向居中）。"""
    if alpha <= 0.003:
        return
    if abs(scale - 1) > 1e-3:
        spr = cv2.resize(spr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = spr.shape[:2]
    x = int(cx - w / 2) if anchor == "c" else int(cx)
    y = int(cy - h / 2)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4].astype(np.float32) * (alpha / 255.0)
    reg = fr[y0:y1, x0:x1]
    reg[:] = (reg * (1 - a) + s[..., :3] * a).astype(np.uint8)


def put(fr, s, cx, cy, size, fill=WHITE, alpha=1.0, scale=1.0, bold=True, anchor="c", stroke=0, track=0.0):
    blit(fr, tsprite(s, px(size), fill, bold, stroke, track), cx, cy, alpha, scale, anchor)


def pill(fr, s, cx, cy, size, bg, fg=WHITE, alpha=1.0, scale=1.0):
    blit(fr, pill_sprite(s, px(size), bg, fg), cx, cy, alpha, scale)


_rm = {}


def round_mask(w, h, r):
    k = (w, h, r)
    if k not in _rm:
        m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, fill=255)
        _rm[k] = (np.asarray(m).astype(np.float32) / 255.0)[..., None]
        if len(_rm) > 200:
            _rm.clear()
    return _rm[k]


def fit(img, box):
    """图像等比放进 box=(x,y,w,h)，返回 (缩放后的图, x0, y0, 比例)。"""
    bx, by, bw, bh = box
    s = min(bw / img.shape[1], bh / img.shape[0])
    w, h = int(img.shape[1] * s), int(img.shape[0] * s)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR), int(bx + (bw - w) / 2), int(by + (bh - h) / 2), s


def paste(fr, img, x, y, alpha=1.0, radius=0):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0 or alpha <= 0.003:
        return
    src = img[y0 - y:y1 - y, x0 - x:x1 - x]
    if radius <= 0 and alpha >= 0.999:
        fr[y0:y1, x0:x1] = src; return
    m = (round_mask(w, h, radius)[y0 - y:y1 - y, x0 - x:x1 - x] if radius > 0 else 1.0) * alpha
    reg = fr[y0:y1, x0:x1]
    reg[:] = (reg * (1 - m) + src * m).astype(np.uint8)


def glow(fr, layer, k=4, sigma=5.0, gain=1.6, core=True):
    """把发光层叠加到画面：核心 + 模糊光晕。"""
    small = cv2.resize(layer, (W // k, H // k), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma)
    up = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    if core:
        cv2.add(fr, layer, dst=fr)
    cv2.addWeighted(fr, 1.0, up, gain, 0, dst=fr)


_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.55 * np.clip(((_xx - W / 2) / (W * 0.62)) ** 2 + ((_yy - H / 2) / (H * 0.62)) ** 2, 0, 1))[..., None].astype(np.float32)
GRAIN = rng.normal(0, 5.0, (H, W, 1)).astype(np.float32)
_g = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
BG = (np.array(INK, np.float32) * 0.75 * (1 - _g) + np.array((30, 46, 62), np.float32) * _g)
BG = np.broadcast_to(BG, (H, W, 3)).astype(np.uint8)
RAD = np.clip(1 - np.hypot(_xx - W / 2, _yy - H / 2) / (0.75 * max(W, H)), 0, 1)[..., None].astype(np.float32)


def post(fr, t, vig=1.0, grain=1.0):
    f = fr.astype(np.float32)
    if vig > 0:
        f *= (1 - vig) + vig * VIG
    if grain > 0:
        f += np.roll(GRAIN, (int(t * 977) % H, int(t * 613) % W), (0, 1)) * grain
    return np.clip(f, 0, 255).astype(np.uint8)


def bg(t, grid=False):
    """动态背景：渐变 + 缓慢移动的斜向网格 + 漂浮光点。"""
    fr = BG.copy()
    if grid:
        lay = np.zeros((H, W, 3), np.uint8)
        step = px(120); off = int(t * 24 * U) % step
        for x in range(-H, W + step, step):
            cv2.line(lay, (x + off, 0), (x + off + H // 3, H), (26, 40, 56), 1, cv2.LINE_AA)
        for y in range(0, H + step, step):
            cv2.line(lay, (0, y - off // 2), (W, y - off // 2), (22, 34, 48), 1, cv2.LINE_AA)
        cv2.add(fr, lay, dst=fr)
    lay = np.zeros((H, W, 3), np.uint8)
    for i in range(26):
        x = (DUST[i, 0] * W + t * DUST[i, 2] * 40 * U) % W
        y = (DUST[i, 1] * H - t * (12 + DUST[i, 3] * 26) * U) % H
        cv2.circle(lay, (int(x), int(y)), max(int(DUST[i, 4] * 3 * U), 1), (30, 48, 64), -1, cv2.LINE_AA)
    glow(fr, lay, 4, 6, 0.8)
    return fr


DUST = rng.random((60, 5))


def glitch(fr, strength, seed):
    """故障效果：横向错位条带 + 红蓝通道分离。"""
    if strength <= 0:
        return fr
    r = np.random.default_rng(seed)
    out = fr.copy()
    d = int(strength * 18 * U) + 1
    out[..., 0] = np.roll(fr[..., 0], d, 1); out[..., 2] = np.roll(fr[..., 2], -d, 1)
    for _ in range(int(3 + strength * 9)):
        y = int(r.integers(0, H - 10)); hh = int(r.integers(px(6), px(70)))
        out[y:y + hh] = np.roll(out[y:y + hh], int(r.integers(-1, 2) * strength * r.integers(20, 160) * U), 1)
    return out


def flash(fr, a, color=(255, 255, 255)):
    if color == (255, 255, 255):
        a *= 0.3
    if a > 0:
        cv2.addWeighted(fr, 1 - a, np.full_like(fr, color), a, 0, dst=fr)


def ring(lay, cx, cy, r, color, th=3):
    if r > 1:
        cv2.circle(lay, (int(cx), int(cy)), int(r), color, max(int(th * U), 1), cv2.LINE_AA)


def title(fr, step, main, t, sub=None):
    """分镜标题：字距拉开的小字 + 两侧细线 + 细体大标题，缓缓浮现。"""
    a = eout(t / 0.7); dy = int((1 - a) * 16 * U)
    y0 = H * (0.07 if PORT else 0.08)
    if step:
        put(fr, step, W / 2, y0 + dy, 24, AMBER, a, bold=False, track=0.45)
        ln = int(px(70) * a)
        for sgn in (-1, 1):
            x_ = int(W / 2 + sgn * px(95))
            cv2.line(fr, (x_, int(y0 + dy)), (x_ + sgn * ln, int(y0 + dy)), tuple(int(v * 0.7 * a) for v in AMBER), 1, cv2.LINE_AA)
    put(fr, main, W / 2, y0 + px(70 if PORT else 62) + dy, 54 if PORT else 50, WHITE, a, bold=False, track=0.06)
    if sub:
        put(fr, sub, W / 2, y0 + px(132 if PORT else 118) + dy, 28, GREY, a, bold=False, track=0.12)


def scrim(fr, y0, y1, k=0.6, up=True):
    """竖向渐变压暗一条区域，让叠在三维画面上的字更清楚。"""
    y0, y1 = int(max(y0, 0)), int(min(y1, H))
    if y1 <= y0 or k <= 0:
        return
    g = np.linspace(1, 0, y1 - y0, dtype=np.float32) if up else np.linspace(0, 1, y1 - y0, dtype=np.float32)
    fr[y0:y1] = (fr[y0:y1] * (1 - k * g[:, None, None])).astype(np.uint8)


BL = os.path.join(HERE, "blender", "render")
BL_N = dict(tunnel=180, poses=105, scan=180, phones=180, end=180)
_LIFT = np.array([0.0, 6.0, 14.0], np.float32); _WARM = np.array([10.0, 3.0, -8.0], np.float32)


def bl(shot, t, bloom=0.9, shift=0.0):
    """读取 Blender 渲染帧，加柔光、冷暖分离调色。"""
    i = min(max(int(t * FPS) + 1, 1), BL_N[shot])
    im = cv2.imread(os.path.join(BL, f"{shot}_{ORI}", f"{i:04d}.png"))
    if im is None:
        fr = np.full((H, W, 3), 18, np.uint8)
        put(fr, f"三维镜头 {shot} 第 {i} 帧尚未渲染", W / 2, H / 2, 40, GREY, bold=False)
        return fr
    if im.shape[1] != W:
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    fr = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    hi = cv2.subtract(fr, np.full_like(fr, 150))
    if shift:
        fr = cv2.warpAffine(fr, np.float32([[1, 0, 0], [0, 1, shift * H]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
    glow(fr, cv2.multiply(hi if not shift else cv2.subtract(fr, np.full_like(fr, 150)), np.full_like(fr, 2)), 4, 7, bloom, core=False)
    f = fr.astype(np.float32)
    lum = f.mean(2, keepdims=True) / 255.0
    f += (1 - lum) ** 2 * _LIFT + lum ** 2 * _WARM
    return np.clip(f, 0, 255).astype(np.uint8)


def box_main():
    return (W * 0.05, H * 0.21, W * 0.90, H * 0.60) if PORT else (W * 0.14, H * 0.235, W * 0.72, H * 0.62)


def box_pair():
    if PORT:
        return (W * 0.07, H * 0.205, W * 0.86, H * 0.30), (W * 0.07, H * 0.535, W * 0.86, H * 0.30)
    return (W * 0.035, H * 0.26, W * 0.445, H * 0.58), (W * 0.52, H * 0.26, W * 0.445, H * 0.58)


# ====================================================================== 素材与真实处理结果
print("准备素材…")
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
ref_full = fa.load_image(os.path.join(SAMPLES, "promo", "last_month.jpg"))
new_full = fa.load_image(os.path.join(SAMPLES, "promo", "today.jpg"))
rm = fa.detect_markers(ref_full); dots = fa.select_reference_dots(rm)
res, rw_bgr, nw_bgr = fa.align(ref_full, rm, dots, new_full)
al_bgr, valid = fa.warp_to_ref(nw_bgr, res, rm.work_size)
excl = fa.comparison_mask(rm.work_size, res, rm.qr_corners, dots)
cmask = ((excl > 0) & (valid > 0)).astype(np.uint8)
ch = ms.compare_cracks(fa.gray_of(rw_bgr), fa.gray_of(al_bgr), cmask)
Hm, n_inl, (src_in, dst_in), _, _, _ = fa.feature_homography(fa.gray_of(rw_bgr), fa.gray_of(nw_bgr), None, None)
N_MATCH = res.feature_inliers
RW, NW, AL = rgb(rw_bgr), rgb(nw_bgr), rgb(al_bgr)
WW, WH = RW.shape[1], RW.shape[0]
QR = np.asarray(rm.qr_corners, np.float32).reshape(4, 2)
HNEW2REF = np.asarray(res.H, np.float64)
sel = rng.choice(len(src_in), size=min(260, len(src_in)), replace=False)
M_NEW, M_REF = src_in[sel], dst_in[sel]
order = np.argsort(M_REF[:, 1] + rng.normal(0, 200, len(sel)))
M_NEW, M_REF = M_NEW[order], M_REF[order]
OLD = np.array(sorted([(y, x) for y, x in zip(*np.nonzero((ch.new.skel > 0) & (ch.grow_skel == 0)))]))
GRW = np.array(sorted([(y, x) for y, x in zip(*np.nonzero(ch.grow_skel))]))
ALLP = np.vstack([OLD, GRW])
print(f"对齐 {res.status}，特征匹配 {N_MATCH} 对；裂缝骨架 原有 {len(OLD)} 点、新增 {len(GRW)} 点")

thumbs_src = [f for pat in ("*.jpg", "lighting/*.jpg", "robustness/*.jpg", "series/*.jpg", "measure/*_ref.jpg", "measure/*_new.jpg", "promo/*.jpg")
              for f in sorted(glob.glob(os.path.join(SAMPLES, pat))) if "_changes" not in f]
COLS = 4 if PORT else 7
TS = int(W / COLS) - px(10)
THUMBS = []
for f in thumbs_src:
    im = cv2.imread(f)
    if im is None:
        continue
    h, w = im.shape[:2]; s = min(h, w)
    im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    THUMBS.append(rgb(cv2.resize(im, (TS, TS), interpolation=cv2.INTER_AREA)))
TH_ORDER = rng.permutation(len(THUMBS) * 6) % len(THUMBS)
shot = lambda n: rgb(cv2.imread(os.path.join(SHOTS, n)))
SHOT_CMP, SHOT_CRACK, SHOT_ZONE = shot("截图_对比结论.png"), shot("截图_裂缝标注.png"), shot("截图_系列页.png")
try:
    SUBS = json.load(open(os.path.join(SHOTS, "旁白时间.json"), encoding="utf-8"))
except Exception:
    SUBS = [{"start": a, "dur": d, "text": s} for a, s, _, d in fs.NARRATION]


def phone(img, height):
    """把截图装进手机外框，返回 RGBA。"""
    k = ("phone", id(img), height)
    if k not in _spr:
        bz = max(int(height * 0.014), 6)
        s = (height - 2 * bz) / img.shape[0]
        sc = cv2.resize(img, (int(img.shape[1] * s), height - 2 * bz), interpolation=cv2.INTER_AREA)
        w = sc.shape[1] + 2 * bz
        im = Image.new("RGBA", (w, height), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, w - 1, height - 1), radius=int(height * 0.06), fill=(8, 12, 18, 255), outline=(90, 104, 120, 255), width=2)
        arr = np.asarray(im).copy()
        m = round_mask(sc.shape[1], sc.shape[0], int(height * 0.048))
        reg = arr[bz:bz + sc.shape[0], bz:bz + sc.shape[1]]
        reg[..., :3] = (reg[..., :3] * (1 - m) + sc * m).astype(np.uint8)
        _spr[k] = arr
    return _spr[k]


def rot_sprite(spr, deg):
    if abs(deg) < 0.05:
        return spr
    h, w = spr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(spr, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


# ====================================================================== 分镜
# ---- 1. 隧道冷开场（Blender）：沿隧道推进，停在侧壁那条裂缝前
def s_tunnel(t, d):
    fr = bl("tunnel", t)
    scrim(fr, 0, H * 0.42, 0.55)
    hud = ease((t - 0.6) / 0.8) * 0.75 * (1 - ease((t - 3.4) / 0.5))
    put(fr, "K128+460  ·  二衬  ·  裂缝观测点", px(64) if not PORT else W / 2, px(70) if not PORT else H * 0.06, 24, (150, 175, 195), hud,
        bold=False, anchor="l" if not PORT else "c", track=0.2)
    a = ease((t - 3.6) / 0.9)
    put(fr, "它，在长吗？", W / 2, H * (0.20 if PORT else 0.19), 104 if PORT else 112, WHITE, a, 1.04 - 0.04 * a, bold=False, track=0.12 + 0.06 * a)
    fr = post(fr, t, 0.9, 0.9)
    flash(fr, 1 - ease(t / 1.2), (0, 0, 0))
    return fr


# ---- 2. 相册：上千张照片
def s_album(t, d):
    fr = np.zeros((H, W, 3), np.uint8)
    gap = px(10); cell = TS + gap
    off = (60 + 520 * t + 300 * t * t) * U
    r0 = int(off // cell)
    for r in range(r0 - 1, r0 + H // cell + 2):
        y = int(r * cell - off + H * 0.02)
        for c in range(COLS):
            paste(fr, THUMBS[TH_ORDER[(r * COLS + c) % len(TH_ORDER)]], int(c * cell + gap / 2), y, 1.0, px(10))
    blur = ease((t - 1.6) / 1.4)
    if blur > 0:
        k = 1 + 2 * int(blur * 14 * U)
        fr = cv2.GaussianBlur(fr, (k, k), 0)
    fr = (fr * (1 - 0.72 * ease(t / 0.8))).astype(np.uint8)
    a = ease(t / 0.5)
    put(fr, "上个月，你拍过它", W / 2, H * 0.42, 84 if PORT else 88, WHITE, a * (1 - ease((t - 2.0) / 0.4)), bold=False, track=0.08)
    put(fr, "可它在哪？", W / 2, H * 0.52, 60, AMBER, ease((t - 1.0) / 0.5) * (1 - ease((t - 2.0) / 0.4)), bold=False, track=0.12)
    put(fr, "相册里，上千张几乎一样的墙", W / 2, H * 0.48, 58 if PORT else 64, WHITE, ease((t - 2.3) / 0.5), bold=False, track=0.08)
    return post(fr, t, 0.9, 1.0)


# ---- 3. 两次到访（Blender）：上个月正面冷光 → 今天斜拍侧光，视锥标出两次机位
def s_pair(t, d):
    fr = bl("poses", t)
    cut = 51 / FPS
    scrim(fr, 0, H * 0.30, 0.6)
    title(fr, None, "同一处裂缝，两次拍摄", t)
    first = t < cut
    k = eout((t if first else t - cut) / 0.45)
    lab, det, col = ("上个月", "正面  ·  冷光  ·  1 米", CYAN) if first else ("今天", "斜拍  ·  侧面暖光  ·  更低更近", AMBER)
    x0 = W * 0.08 if not PORT else W / 2
    y0 = H * 0.70 if not PORT else H * 0.74
    an = "l" if not PORT else "c"
    put(fr, lab, x0, y0, 76, col, k, bold=False, anchor=an, track=0.1)
    put(fr, det, x0, y0 + px(78), 30, GREY, eout(((t if first else t - cut) - 0.15) / 0.4), bold=False, anchor=an, track=0.15)
    for t0 in (0.2, cut):                                   # 快门闪光
        if 0 <= t - t0 < 0.25:
            flash(fr, 0.9 * (1 - (t - t0) / 0.25))
    return post(fr, t, 0.8, 0.8)


# ---- 4. 直接叠在一起：完全对不上（两张真实照片半透明叠加，重影缓慢漂移）
def s_chaos(t, d):
    fr = bg(t)
    box = box_main()
    a_im, x, y, s = fit(RW, box); b_im = cv2.resize(NW, (a_im.shape[1], a_im.shape[0]), interpolation=cv2.INTER_AREA)
    drift = np.sin(t * 1.4) * 10 * U
    M = np.float32([[1, 0, drift], [0, 1, -drift * 0.4]])
    b_sh = cv2.warpAffine(b_im, M, (b_im.shape[1], b_im.shape[0]), borderMode=cv2.BORDER_REFLECT)
    mix = cv2.addWeighted(a_im, 0.5, b_sh, 0.5, 0)
    sp = int((2 + 6 * ease(t / d)) * U)                     # 轻微色散，越来越不稳
    mix[..., 0] = np.roll(mix[..., 0], sp, 1); mix[..., 2] = np.roll(mix[..., 2], -sp, 1)
    paste(fr, mix, x, y, ease(t / 0.4), px(18))
    lay = np.zeros((H, W, 3), np.uint8)
    y0, x0 = ALLP.min(0); y1, x1 = ALLP.max(0)
    quad_ref = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    quad_new = cv2.perspectiveTransform(quad_ref[None], np.linalg.inv(HNEW2REF))[0]
    k = ease((t - 0.3) / 0.6)
    for q, col, lab in ((quad_ref, CYAN, "上个月它在这"), (quad_new, AMBER, "今天它在这")):
        p = (q * s + [x, y]).astype(np.int32)
        cv2.polylines(lay, [p], True, col, max(px(2), 1), cv2.LINE_AA)
        put(lay, lab, int(p[:, 0].mean()), int(p[:, 1].min() - px(30)), 26, col, k, bold=False, track=0.1)
    cv2.addWeighted(fr, 1.0, lay, k, 0, dst=fr)
    a = eout((t - 0.4) / 0.5)
    put(fr, "叠在一起", W / 2, H * (0.12 if PORT else 0.10), 60, WHITE, a, bold=False, track=0.15)
    put(fr, "完全对不上", W / 2, H * (0.88 if PORT else 0.905), 92 if PORT else 84, RED, eout((t - 1.0) / 0.5), bold=False, track=0.12)
    fr = post(fr, t, 0.8, 1.0)
    flash(fr, ease((t - (d - 0.5)) / 0.45), (0, 0, 0))
    return fr


# ---- 5. 口号：三个词依次浮现，下方细线展开
def s_slogan(t, d):
    fr = bg(t)
    words = [("一码", WHITE, 0.45), ("一拍", WHITE, 0.95), ("出结果", AMBER, 1.45)]
    for i, (wd, col, t0) in enumerate(words):
        a = eout((t - t0) / 0.5)
        cxw = W / 2 if PORT else W * (0.25 + 0.25 * i + (0.02 if i == 2 else 0))
        cyw = H * (0.34 + 0.12 * i) if PORT else H * 0.45
        put(fr, wd, cxw, cyw + (1 - a) * px(24), 170 if PORT else 176, col, a, bold=False, track=0.1)
        ln = int(px(150) * ease((t - t0 - 0.15) / 0.6))
        if ln > 0:
            yl = int(cyw + px(118))
            cv2.line(fr, (int(cxw - ln), yl), (int(cxw + ln), yl), tuple(int(v * 0.6) for v in col), max(px(2), 1), cv2.LINE_AA)
    a = ease((t - 2.2) / 0.7)
    put(fr, "手机拍照  ·  自动对齐  ·  裂缝变化直接读数", W / 2, H * (0.78 if PORT else 0.72), 34 if PORT else 38, GREY, a, bold=False, track=0.15)
    fr = post(fr, t, 0.9, 0.8)
    flash(fr, 1 - ease(t / 0.4), (0, 0, 0))
    return fr


# ---- 6. 扫码锁定测点（Blender）：手机抬起对准二维码，扫描光带掠过
def s_scan(t, d):
    fr = bl("scan", t)
    scrim(fr, 0, H * 0.28, 0.65)
    title(fr, "第 1 步", "贴一个二维码，扫一下", t)
    lines = [("测点编号", "B01-Z023", GOOD), ("比例尺", "40 mm 二维码 → 0.228 mm / 像素", WHITE),
             ("位置 · 朝向", "自动记录，下次按此归档", WHITE), ("没法贴码？", "按位置 + 朝向 + 画面特征识别", GREY)]
    if PORT:
        px0, py0, step = W * 0.09, H * 0.735, px(88)
        scrim(fr, H * 0.68, H, 0.8, up=False)
    else:
        px0, py0, step = W * 0.64, H * 0.36, px(112)
        pa = ease((t - 3.2) / 0.5) * 0.55
        if pa > 0:
            x_ = int(W * 0.615)
            reg = fr[int(H * 0.30):int(H * 0.80), x_:int(W * 0.96)]
            reg[:] = (reg * (1 - pa)).astype(np.uint8)
            cv2.line(fr, (x_, int(H * 0.30)), (x_, int(H * 0.80)), tuple(int(v * pa) for v in AMBER), 1, cv2.LINE_AA)
    for i, (kq, v, col) in enumerate(lines):
        tt_ = t - 3.3 - 0.45 * i
        if tt_ <= 0:
            continue
        yy = py0 + i * step
        put(fr, kq, px0, yy, 24, (130, 155, 178), ease(tt_ / 0.3), bold=False, anchor="l", track=0.2)
        put(fr, v, px0, yy + px(42), 34 if PORT else 32, col, ease((tt_ - 0.1) / 0.4), bold=False, anchor="l")
    return post(fr, t, 0.8, 0.8)


# ---- 7. 特征匹配
def pair_layout():
    A, B = box_pair()
    ra, ax, ay, sa = fit(RW, A); nb, bx, by, sb = fit(NW, B)
    return ra, ax, ay, sa, nb, bx, by, sb


def s_match(t, d):
    fr = bg(t)
    title(fr, "第 2 步", "换角度、换光线，照样对齐", t)
    ra, ax, ay, sa, nb, bx, by, sb = pair_layout()
    paste(fr, ra, ax, ay, 1.0, px(22)); paste(fr, nb, bx, by, 1.0, px(22))
    pill(fr, "上个月", ax + px(90), ay + px(44), 28, (20, 28, 38), GREY); pill(fr, "今天", bx + px(70), by + px(44), 28, (20, 28, 38), AMBER)
    lay = np.zeros((H, W, 3), np.uint8); pts = np.zeros((H, W, 3), np.uint8)
    n = len(M_REF); shown = 0
    for i in range(n):
        ti = 0.35 + 2.7 * i / n
        if t < ti:
            break
        shown += 1
        p_r = (M_REF[i] * sa + [ax, ay]); p_n = (M_NEW[i] * sb + [bx, by])
        age = t - ti
        k = min(age / 0.22, 1.0)
        fade = 0.28 + 0.72 * max(0, 1 - age / 0.8)
        col = tuple(int(v * fade) for v in CYAN)
        e = p_n + (p_r - p_n) * k
        cv2.line(lay, (int(p_n[0]), int(p_n[1])), (int(e[0]), int(e[1])), col, 1, cv2.LINE_AA)
        r = max(int((2.5 + 5 * max(0, 1 - age / 0.3)) * U), 1)
        cv2.circle(pts, (int(p_n[0]), int(p_n[1])), r, AMBER, -1, cv2.LINE_AA)
        if k >= 1:
            cv2.circle(pts, (int(p_r[0]), int(p_r[1])), r, GOOD, -1, cv2.LINE_AA)
    glow(fr, lay, 4, 3, 1.0)
    glow(fr, pts, 4, 3, 1.2)
    cnt = int(N_MATCH * min(max((t - 0.35) / 2.7, 0), 1))
    cy_ = H * 0.872 if PORT else H * 0.905
    put(fr, f"特征点匹配  {cnt:,} 对", W / 2, cy_, 44, CYAN, ease(t / 0.4))
    return post(fr, t, 0.7, 0.7)


# ---- 8. 变形对齐
GRID_NEW = None


def s_warp(t, d):
    global GRID_NEW
    fr = bg(t)
    title(fr, "第 2 步", "换角度、换光线，照样对齐", 9)
    ra, ax, ay, sa, nb, bx, by, sb = pair_layout()
    paste(fr, ra, ax, ay, 1.0, px(22))
    if GRID_NEW is None:
        g = cv2.resize(NW, (1000, int(1000 * WH / WW)), interpolation=cv2.INTER_AREA)
        gl = g.copy()
        for i in range(1, 10):
            cv2.line(gl, (i * 100, 0), (i * 100, g.shape[0]), CYAN, 2, cv2.LINE_AA)
            cv2.line(gl, (0, i * g.shape[0] // 10), (1000, i * g.shape[0] // 10), CYAN, 2, cv2.LINE_AA)
        GRID_NEW = (g, gl)
    g, gl = GRID_NEW
    k = ease((t - 0.25) / 2.2)
    corners = np.array([[0, 0], [WW, 0], [WW, WH], [0, WH]], np.float32)
    q0 = corners * sb + [bx, by]
    q1 = cv2.perspectiveTransform(corners[None], HNEW2REF)[0] * sa + [ax, ay]
    q = (q0 * (1 - k) + q1 * k).astype(np.float32)
    src = np.array([[0, 0], [g.shape[1], 0], [g.shape[1], g.shape[0]], [0, g.shape[0]]], np.float32)
    M = cv2.getPerspectiveTransform(src, q)
    ga = 1 - ease((t - 2.2) / 0.5)
    srcimg = cv2.addWeighted(g, 1 - 0.6 * ga, gl, 0.6 * ga, 0)
    wimg = cv2.warpPerspective(srcimg, M, (W, H), flags=cv2.INTER_LINEAR)
    wm = cv2.warpPerspective(np.full(g.shape[:2], 255, np.uint8), M, (W, H))
    if k > 0.6:                                              # 只保留落在参考照片框内的部分
        clip = np.zeros((H, W), np.uint8); clip[ay:ay + ra.shape[0], ax:ax + ra.shape[1]] = 255
        kk = ease((k - 0.6) / 0.4)
        wm = cv2.addWeighted(wm, 1 - kk, cv2.bitwise_and(wm, clip), kk, 0)
    a = (wm.astype(np.float32) / 255.0 * (0.92 - 0.12 * k))[..., None]
    fr[:] = (fr * (1 - a) + wimg * a).astype(np.uint8)
    lay = np.zeros((H, W, 3), np.uint8)
    of = 1 - ease((t - 2.3) / 0.4)
    cv2.polylines(lay, [q.astype(np.int32)], True, tuple(int(v * of) for v in CYAN), max(px(3), 2), cv2.LINE_AA)
    glow(fr, lay, 4, 4, 1.2 * of)
    if t > 2.55:
        aa = eout((t - 2.55) / 0.3)
        pill(fr, "已对齐", ax + ra.shape[1] / 2, ay + ra.shape[0] / 2, 54, GOOD, WHITE, aa * (1 - ease((t - 3.2) / 0.3)), 1.6 - 0.6 * aa)
        flash(fr, max(0, 0.5 - (t - 2.55) * 2.2))
    cy_ = H * 0.872 if PORT else H * 0.905
    put(fr, "逐像素变形，把今天这张“摆正”到上个月的视角", W / 2, cy_, 34 if PORT else 38, GREY, ease(t / 0.4), bold=False)
    return post(fr, t, 0.7, 0.7)


# ---- 9. 滑动对比
def s_wipe(t, d):
    fr = bg(t)
    title(fr, "第 2 步", "同一位置，同一视角", t)
    box = box_main()
    a_im, x, y, s = fit(RW, box); b_im = cv2.resize(AL, (a_im.shape[1], a_im.shape[0]), interpolation=cv2.INTER_AREA)
    sp = 0.5 + 0.40 * np.sin(t * 2.6 - 0.6)
    cut = int(a_im.shape[1] * sp)
    mix = a_im.copy(); mix[:, cut:] = b_im[:, cut:]
    paste(fr, mix, x, y, 1.0, px(22))
    lay = np.zeros((H, W, 3), np.uint8)
    cv2.line(lay, (x + cut, y), (x + cut, y + a_im.shape[0]), AMBER, max(px(5), 2), cv2.LINE_AA)
    glow(fr, lay, 4, 4, 1.4)
    pill(fr, "上个月", x + px(90), y + px(44), 28, (20, 28, 38), GREY); pill(fr, "今天", x + a_im.shape[1] - px(70), y + px(44), 28, (20, 28, 38), AMBER)
    cy_ = H * 0.872 if PORT else H * 0.905
    put(fr, "样板实测：靶标与图像特征偏差 0.2 像素", W / 2, cy_, 34 if PORT else 38, GOOD, ease(t / 0.4))
    return post(fr, t, 0.7, 0.7)


# ---- 10. 裂缝：放大 → 标黄 → 标红 + 读数
ROI_CACHE = {}


def s_crack(t, d):
    fr = bg(t)
    title(fr, "第 3 步", "新长出来的裂缝，自动标红", t)
    box = (W * 0.06, H * 0.20, W * 0.88, H * 0.47) if PORT else (W * 0.05, H * 0.235, W * 0.44, H * 0.70)
    bx, by, bw, bh = [int(v) for v in box]
    y0, x0 = ALLP.min(0); y1, x1 = ALLP.max(0)
    asp = bw / bh
    rh = (y1 - y0) * 1.16; rw_ = max(rh * asp, (x1 - x0) * 1.5); rh = rw_ / asp
    rcx, rcy = (x0 + x1) / 2, (y0 + y1) / 2
    fh = min(WH, WW / asp); fw = fh * asp                                 # 起始：铺满图像的最大同比例矩形
    k = ease(t / 2.3)
    cw, chh = fw + (rw_ - fw) * k, fh + (rh - fh) * k
    ccx, ccy = WW / 2 + (rcx - WW / 2) * k, WH / 2 + (rcy - WH / 2) * k
    ccx = min(max(ccx, cw / 2), WW - cw / 2); ccy = min(max(ccy, chh / 2), WH - chh / 2)
    s = bw / cw
    key = "roi" if k >= 1 else None
    if key and key in ROI_CACHE:
        view = ROI_CACHE[key]
    else:
        M = np.array([[s, 0, -(ccx - cw / 2) * s], [0, s, -(ccy - chh / 2) * s]], np.float32)
        view = cv2.warpAffine(AL, M, (bw, bh), flags=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        if key:
            ROI_CACHE[key] = view
    paste(fr, view, bx, by, 1.0, px(22))
    to_scr = lambda pts: np.stack([(pts[:, 1] - (ccx - cw / 2)) * s + bx, (pts[:, 0] - (ccy - chh / 2)) * s + by], 1)
    lay = np.zeros((H, W, 3), np.uint8)
    if 1.3 < t < 2.6:                                                     # 扫描光束
        sy = by + int(ease((t - 1.3) / 1.2) * bh)
        cv2.line(lay, (bx, sy), (bx + bw, sy), CYAN, max(px(4), 2), cv2.LINE_AA)
    a = ease((t - 2.5) / 2.3); b = ease((t - 5.0) / 3.3)
    th = max(px(5), 3)
    if a > 0:
        p = to_scr(OLD[:max(int(len(OLD) * a), 2)]).astype(np.int32)
        cv2.polylines(lay, [np.ascontiguousarray(p[::3])], False, YEL, th, cv2.LINE_AA)
    tip = None
    if b > 0:
        p = to_scr(GRW[:max(int(len(GRW) * b), 2)]).astype(np.int32)
        cv2.polylines(lay, [np.ascontiguousarray(p[::3])], False, RED, th + 1, cv2.LINE_AA)
        tip = p[-1]
    glow(fr, lay, 4, 5, 1.5)
    if tip is not None and b < 1:                                         # 生长尖端的火花
        sp = np.zeros((H, W, 3), np.uint8)
        for j in range(70):
            tb = t - (j * 0.011)
            r = np.random.default_rng(int(tb * 90) * 131 + j)
            age = (j * 0.011) + r.random() * 0.25
            v = r.normal(0, 1, 2) * 260 * U
            pos = tip + v * age + np.array([0, 420 * U]) * age * age
            c = max(0.0, 1 - age / 0.6)
            cv2.circle(sp, (int(pos[0]), int(pos[1])), max(int(3 * U * c + 1), 1), (int(255 * c), int(170 * c), int(60 * c)), -1, cv2.LINE_AA)
        cv2.circle(sp, (int(tip[0]), int(tip[1])), px(9), (255, 240, 220), -1, cv2.LINE_AA)
        glow(fr, sp, 4, 4, 1.8)
    hit = t - 8.5
    if 0 <= hit < 0.8 and tip is not None:
        lay2 = np.zeros((H, W, 3), np.uint8)
        ring(lay2, tip[0], tip[1], hit * 900 * U, tuple(int(v * (1 - hit / 0.8)) for v in RED), 6)
        glow(fr, lay2, 4, 5, 1.5)
        flash(fr, max(0, 0.45 - hit * 1.6))
    # 读数面板
    if PORT:
        pxl, py = W * 0.08, H * 0.70; cxn = W / 2
    else:
        pxl, py = W * 0.545, H * 0.27; cxn = W * 0.755
    put(fr, "■ 原有裂缝", pxl, py, 32, YEL, ease((t - 2.5) / 0.4), anchor="l")
    put(fr, f"上个月  {fs.L_REF * a:.1f} mm", pxl + px(230), py, 34, WHITE, ease((t - 2.5) / 0.4), bold=False, anchor="l")
    put(fr, "■ 新增", pxl, py + px(62), 32, RED, ease((t - 5.0) / 0.4), anchor="l")
    if b > 0:
        big = 190 if PORT else 210
        sc = 1.0 + 0.18 * max(0, 1 - hit / 0.3) if hit >= 0 else 1.0
        put(fr, f"+{fs.GROW * b:.1f}", cxn - px(70), py + px(210 if PORT else 250), big, RED, 1.0, sc)
        put(fr, "mm", cxn + px(250 if PORT else 290), py + px(240 if PORT else 285), 70, RED, 1.0)
        put(fr, f"总长  {fs.L_REF:.1f}  →  {fs.L_REF + (fs.L_NEW - fs.L_REF) * b:.1f} mm", cxn, py + px(350 if PORT else 420), 40, WHITE, 1.0, bold=False)
    if hit > 0.2:
        aa = eout((hit - 0.2) / 0.3)
        pill(fr, f"一个月 +{fs.PCT}%", cxn - px(150), py + px(425 if PORT else 520), 34, RED, WHITE, aa, 1.4 - 0.4 * aa)
        pill(fr, "建议重点关注", cxn + px(150), py + px(425 if PORT else 520), 34, AMBER, INK, eout((hit - 0.45) / 0.3))
    return post(fr, t, 0.7, 0.7)


# ---- 11. App 结论页（Blender）：三部手机展示 App 真实截图
def s_report(t, d):
    fr = bl("phones", t, bloom=0.35, shift=0.0 if PORT else 0.09)
    scrim(fr, 0, H * 0.26, 0.6)
    title(fr, "结论", "报告，自动写好", t, "长度  ·  宽度  ·  新增多少  ·  要不要关注")
    tags = [("延长 %.1f mm" % fs.GROW, RED, WHITE), ("+%d%%" % fs.PCT, RED, WHITE), ("可分享 · 可导出表格", GOOD, WHITE)]
    for i, (s_, bgc, fg) in enumerate(tags):
        a = eout((t - 1.8 - 0.35 * i) / 0.5)
        cx_ = W * (0.22 + 0.28 * i) if PORT else W * (0.30 + 0.20 * i)
        cy_ = H * 0.885 if PORT else H * 0.87
        pill(fr, s_, cx_, cy_, 28 if PORT else 30, bgc, fg, a)
    return post(fr, t, 0.8, 0.7)


# ---- 12. 卖点卡片
FEATS = [("一个二维码", "既是测点编号，又是比例尺", "不用靠尺，像素直接换算成毫米"),
         ("离线可用", "不联网、不传云", "隧道里照样跑"),
         ("自动归档", "每个测点一份“成长档案”", "裂缝总长的走势，一眼看出是否在发展"),
         ("现场把关", "反光、模糊、偏暗", "当场提醒重拍，不把问题带回办公室")]


def s_feats(t, d):
    i = min(int(t / 1.25), 3); tl = t - i * 1.25
    fr = bg(t)
    k = eout(tl / 0.25)
    put(fr, f"0{i + 1}", W / 2, H * (0.27 if PORT else 0.24), 40, AMBER, k, bold=False, track=0.3)
    put(fr, FEATS[i][0], W / 2, H * 0.42 + (1 - k) * px(20), 140 if PORT else 150, WHITE, k, bold=False, track=0.08)
    put(fr, FEATS[i][1], W / 2, H * (0.53 if PORT else 0.58), 46 if PORT else 50, AMBER, eout((tl - 0.15) / 0.35), bold=False, track=0.06)
    put(fr, FEATS[i][2], W / 2, H * (0.585 if PORT else 0.665), 32 if PORT else 36, GREY, eout((tl - 0.3) / 0.35), bold=False, track=0.06)
    for j in range(4):
        cv2.circle(fr, (int(W / 2 + (j - 1.5) * px(46)), int(H * (0.72 if PORT else 0.82))), px(5), AMBER if j == i else (70, 86, 102), -1, cv2.LINE_AA)
    return post(fr, t, 0.9, 0.7)


# ---- 13. 片尾（Blender）：从裂缝前缓缓后退，回到隧道中线
def s_end(t, d):
    fr = bl("end", t)
    k = ease((t - 1.6) / 1.2)
    fr = (fr * (1 - 0.45 * k)).astype(np.uint8)
    a = ease((t - 1.8) / 1.0)
    cy = H * (0.42 if PORT else 0.43)
    put(fr, "复拍对齐相机", W / 2, cy, 120 if PORT else 128, WHITE, a, bold=False, track=0.16)
    ln = int(px(260) * ease((t - 2.3) / 0.8))
    if ln > 0:
        yl = int(cy + px(96)); m = int(W / 2 + ln * 0.3)
        cv2.line(fr, (int(W / 2 - ln), yl), (m, yl), tuple(int(v * 0.8) for v in YEL), max(px(2), 1), cv2.LINE_AA)
        cv2.line(fr, (m, yl), (int(W / 2 + ln), yl), tuple(int(v * 0.9) for v in RED), max(px(2), 1), cv2.LINE_AA)
    put(fr, "让每一条裂缝，都有据可查", W / 2, cy + px(170), 50 if PORT else 52, AMBER, ease((t - 2.6) / 0.8), bold=False, track=0.12)
    put(fr, "隧道  ·  桥梁  ·  铁路  ·  水运工程", W / 2, cy + px(250), 30 if PORT else 32, GREY, ease((t - 3.0) / 0.8), bold=False, track=0.2)
    put(fr, "演示照片为测试样板，读数为该样板在 App 中的实测结果；隧道画面为三维示意动画。", W / 2, H * 0.955, 20 if PORT else 22,
        (120, 134, 148), ease((t - 3.2) / 0.6), bold=False)
    fr = post(fr, t, 0.9, 0.7)
    flash(fr, ease((t - (d - 1.2)) / 1.2), (0, 0, 0))
    return fr


SCENE_FN = dict(tunnel=s_tunnel, album=s_album, pair=s_pair, chaos=s_chaos, slogan=s_slogan, scan=s_scan, match=s_match,
                warp=s_warp, wipe=s_wipe, crack=s_crack, report=s_report, feats=s_feats, end=s_end)
NO_SUB = {"slogan", "end", "feats", "chaos"}
XFADE = {"album": 0.35, "pair": 0.3, "match": 0.3, "report": 0.4, "end": 0.5}


def subtitle(fr, T, scene):
    if scene in NO_SUB:
        return
    for s in SUBS:
        if s["start"] <= T < s["start"] + s["dur"] + 0.35:
            a = ease((T - s["start"]) / 0.15) * (1 - ease((T - s["start"] - s["dur"] - 0.15) / 0.2))
            spr = tsprite(s["text"], px(34 if PORT else 34), (236, 240, 244), False, 2, 0.04)
            cy = H * (0.962 if PORT else 0.962)
            blit(fr, spr, W / 2, cy, a)
            return


def frame_at(T):
    for j, (name, a, b) in enumerate(fs.SCENES):
        if a <= T < b:
            fr = SCENE_FN[name](T - a, b - a)
            xf = XFADE.get(name, 0)
            if j and T - a < xf:
                pn, pa, pb = fs.SCENES[j - 1]
                prev = SCENE_FN[pn](T - pa, pb - pa)
                k = ease((T - a) / xf)
                fr = cv2.addWeighted(prev, 1 - k, fr, k, 0)
            subtitle(fr, T, name)
            return fr
    return np.zeros((H, W, 3), np.uint8)


def main():
    if PREVIEW:
        times = [1.2, 3.2, 5.0, 7.0, 9.2, 11.5, 14.0, 15.4, 17.3, 19.0, 21.2, 24.5, 28.0, 29.6, 31.5, 33.2, 34.6, 37.2, 40.0, 43.0, 45.3,
                 47.5, 50.5, 52.6, 55.2, 58.6, 61.0]
        sw = 480 if not PORT else 270; sh = int(sw * H / W)
        cols = 5 if not PORT else 9
        tiles = []
        for T in times:
            f = cv2.resize(frame_at(T), (sw, sh), interpolation=cv2.INTER_AREA)
            cv2.putText(f, f"{T:.1f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            tiles.append(f)
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        out = os.path.join(SHOTS, f"预览_{'竖' if PORT else '横'}.jpg")
        cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("预览图：", out)
        return
    import subprocess
    FF = os.path.expanduser("~/Desktop/robot/教材制作/工具/录屏/bin/ffmpeg")
    out = os.path.join(HERE, f"宣传片_{'竖' if PORT else '横'}版.mp4")
    cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", os.path.join(SHOTS, "音轨.wav"), "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(fs.TOTAL * FPS)
    keep = {"tunnel": 5.0, "chaos": 14.6, "slogan": 18.6, "scan": 24.8, "match": 29.3, "warp": 31.6, "wipe": 34.6, "crack": 45.3, "report": 50.5, "end": 60.0}
    for i in range(n):
        T = i / FPS
        fr = frame_at(T)
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        for kname, kt in keep.items():
            if abs(T - kt) < 0.5 / FPS:
                cv2.imwrite(os.path.join(SHOTS, f"{'竖' if PORT else '横'}_{kname}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if i % 150 == 0:
            print(f"  {T:5.1f}s / {fs.TOTAL:.0f}s", flush=True)
    proc.stdin.close(); proc.wait()
    print("已生成", out)


if __name__ == "__main__":
    main()

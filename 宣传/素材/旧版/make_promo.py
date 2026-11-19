"""
生成宣传片（竖屏 1080×1920，30fps，无声）和宣传配图。

  ../.venv/bin/python make_promo.py

画面全部来自真实流程：samples/measure/w1.0mm 这一对测试样板照片经对齐、裂缝对比得到的结果，
以及模拟器上 App 的真实截图（素材/截图_*.png）。换成现场照片：改 REF / NEW 两个路径即可。
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import fieldalign as fa  # noqa: E402
import measure as ms  # noqa: E402

REF = os.path.join(HERE, "..", "samples", "measure", "w1.0mm_ref.jpg")
NEW = os.path.join(HERE, "..", "samples", "measure", "w1.0mm_new.jpg")
SHOTS = os.path.join(HERE, "素材")
APP_NUMBERS = (126.4, 193.6, 64.3)      # App 截图中的 上次总长、这次总长、新增（mm）；换素材后改这里或设为 None
W, H, FPS = 1080, 1920, 30
INK = (22, 32, 42); AMBER = (255, 176, 32); GOOD = (34, 160, 107); RED = (255, 64, 64); WHITE = (255, 255, 255); GREY = (170, 182, 195)
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"


def font(size, bold=True):
    return ImageFont.truetype(FONT, size, index=2 if bold else 0)   # W6 / W3


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def text(img, s, xy, size, fill=WHITE, bold=True, anchor="mm", alpha=1.0, stroke=0):
    """在 numpy RGB 图上写字（支持中文）。alpha<1 时淡入。"""
    if alpha <= 0:
        return img
    pil = Image.fromarray(img)
    if alpha >= 1:
        ImageDraw.Draw(pil).text(xy, s, font=font(size, bold), fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=(0, 0, 0))
        return np.asarray(pil).copy()
    lay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ImageDraw.Draw(lay).text(xy, s, font=font(size, bold), fill=fill + (int(255 * alpha),), anchor=anchor,
                             stroke_width=stroke, stroke_fill=(0, 0, 0, int(255 * alpha)))
    return np.asarray(Image.alpha_composite(pil.convert("RGBA"), lay).convert("RGB")).copy()


def pill(img, s, cxy, size, bg, fg=WHITE, pad=(34, 18)):
    pil = Image.fromarray(img); d = ImageDraw.Draw(pil)
    f = font(size); l, t, r, b = d.textbbox((0, 0), s, font=f)
    w, h = r - l + 2 * pad[0], b - t + 2 * pad[1]
    x0, y0 = cxy[0] - w // 2, cxy[1] - h // 2
    d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=h // 2, fill=bg)
    d.text(cxy, s, font=f, fill=fg, anchor="mm")
    return np.asarray(pil).copy()


def bg_frame():
    """深色渐变背景。"""
    y = np.linspace(0, 1, H)[:, None, None]
    top = np.array(INK, np.float32); bot = np.array((43, 58, 72), np.float32)
    return np.broadcast_to(top * (1 - y) + bot * y, (H, W, 3)).astype(np.uint8).copy()


def paste(canvas, img, x, y, alpha=1.0):
    h, w = img.shape[:2]
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return canvas
    src = img[y0 - y:y1 - y, x0 - x:x1 - x]
    if alpha >= 1:
        canvas[y0:y1, x0:x1] = src
    else:
        canvas[y0:y1, x0:x1] = (canvas[y0:y1, x0:x1] * (1 - alpha) + src * alpha).astype(np.uint8)
    return canvas


def rounded(img, r=36):
    h, w = img.shape[:2]
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w, h), radius=r, fill=255)
    return np.asarray(m)[..., None] / 255.0


def paste_round(canvas, img, x, y, r=36, alpha=1.0):
    h, w = img.shape[:2]
    if y >= H or x >= W or y + h <= 0:
        return canvas
    m = rounded(img, r) * alpha
    y0, y1 = max(y, 0), min(y + h, H)
    reg = canvas[y0:y1, x:x + w].astype(np.float32)
    canvas[y0:y1, x:x + w] = (reg * (1 - m[y0 - y:y1 - y]) + img[y0 - y:y1 - y] * m[y0 - y:y1 - y]).astype(np.uint8)
    return canvas


def fit_w(img, w):
    return cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)


def main():
    # ---------------- 真实处理结果
    ref_img, new_img = fa.load_image(REF), fa.load_image(NEW)
    rm = fa.detect_markers(ref_img); dots = fa.select_reference_dots(rm)
    res, rw, nw = fa.align(ref_img, rm, dots, new_img)
    al, valid = fa.warp_to_ref(nw, res, rm.work_size)
    mask = fa.comparison_mask(rm.work_size, res, rm.qr_corners, dots) & (valid > 0)
    ch = ms.compare_cracks(fa.gray_of(rw), fa.gray_of(al), mask.astype(np.uint8))
    sc = ms.scale_from_qr(rm.qr_corners, ms.qr_size_mm(rm.qr_text))
    ys, xs = np.nonzero(ch.new.skel)
    mmpp = sc.mm_per_px((xs.mean(), ys.mean()))
    L_new, L_ref, G = ch.new.total_px * mmpp, ch.ref.total_px * mmpp, ch.grow_px * mmpp
    # 与 App 截图里的数字保持一致（手机端与电脑端结果相差约 2%，以 App 显示为准）
    if APP_NUMBERS:
        L_ref, L_new, G = APP_NUMBERS
    print(f"结果：上次 {L_ref:.1f} mm → 这次 {L_new:.1f} mm，新增 {G:.1f} mm，最大宽 {ch.new.max_w_px * mmpp:.2f} mm")
    rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    rw, al, nw = rgb(rw), rgb(al), rgb(nw)
    IW = 1000; s_img = IW / rw.shape[1]
    ref_s, al_s, new_s = fit_w(rw, IW), fit_w(al, IW), fit_w(nw, IW)
    IH = ref_s.shape[0]; IX = (W - IW) // 2; IY = 520
    qr = np.asarray(rm.qr_corners).reshape(4, 2) * s_img
    # 裂缝放大区域（竖幅）
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ch_h = (y1 - y0) * 1.12; ch_w = ch_h * 1000 / 1040
    X0 = int(max(cx - ch_w / 2, 0)); Y0 = int(max(cy - ch_h / 2, 0))
    crop = al[Y0:Y0 + int(ch_h), X0:X0 + int(ch_w)]
    zs = 1000 / crop.shape[1]
    crop_z = cv2.resize(crop, (1000, int(crop.shape[0] * zs)), interpolation=cv2.INTER_CUBIC)
    ZH = crop_z.shape[0]
    old_pts = np.array(sorted([(y, x) for y, x in zip(*np.nonzero((ch.new.skel > 0) & (ch.grow_skel == 0)))]))
    new_pts = np.array(sorted([(y, x) for y, x in zip(*np.nonzero(ch.grow_skel))]))

    def draw_skel(img, pts, frac, color, r=5):
        n = int(len(pts) * frac)
        for y, x in pts[:n:2]:
            cv2.circle(img, (int((x - X0) * zs), int((y - Y0) * zs)), r, color, -1, cv2.LINE_AA)
        return img

    shot_cmp = cv2.cvtColor(cv2.imread(os.path.join(SHOTS, "截图_对比结论.png")), cv2.COLOR_BGR2RGB)
    shot_zone = cv2.cvtColor(cv2.imread(os.path.join(SHOTS, "截图_系列页.png")), cv2.COLOR_BGR2RGB)
    card = fit_w(shot_cmp[290:2050], 900)          # 结论卡 + 结果区
    zone = fit_w(shot_zone[290:1500], 900)

    out_path = os.path.join(HERE, "宣传片.mp4")
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    if not vw.isOpened():
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    stills = {}

    def emit(frame, keep=None):
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if keep:
            stills[keep] = frame.copy()

    def scene(sec, fn):
        n = int(sec * FPS)
        for i in range(n):
            emit(*fn(i / max(n - 1, 1), i / FPS))

    def header(f, step, title, sub=None):
        f = pill(f, step, (W // 2, 210), 40, AMBER, INK)
        f = text(f, title, (W // 2, 330), 76)
        if sub:
            f = text(f, sub, (W // 2, 425), 40, GREY, bold=False)
        return f

    # ---------------- 1. 钩子
    def s1(p, t):
        f = np.zeros((H, W, 3), np.uint8)
        f = text(f, "这条裂缝", (W // 2, 760), 110, alpha=ease(t / 0.5))
        f = text(f, "一个月里", (W // 2, 930), 110, alpha=ease((t - 0.7) / 0.5))
        f = text(f, "长了多少？", (W // 2, 1110), 130, AMBER, alpha=ease((t - 1.3) / 0.5))
        return f, ("hook" if p == 1 else None)
    scene(3.2, s1)

    # ---------------- 2. 痛点
    pains = ["靠肉眼看？", "靠翻相册比？", "靠记忆猜？"]

    def s2(p, t):
        f = np.zeros((H, W, 3), np.uint8)
        for i, s in enumerate(pains):
            a = ease((t - i * 0.7) / 0.4)
            y = 700 + i * 200
            f = text(f, s, (W // 2, y), 96, GREY, alpha=a)
            k = ease((t - 2.3 - i * 0.12) / 0.3)
            if k > 0:
                cv2.line(f, (200, y), (int(200 + 680 * k), y), RED, 10, cv2.LINE_AA)
        return f, None
    scene(3.6, s2)

    # ---------------- 3. 口号
    def s3(p, t):
        f = bg_frame()
        k = ease(t / 0.35)
        f = text(f, "一码", (W // 2, 700), int(150 * (1.5 - 0.5 * k)), alpha=k)
        k2 = ease((t - 0.45) / 0.35)
        f = text(f, "一拍", (W // 2, 930), int(150 * (1.5 - 0.5 * k2)), alpha=k2)
        k3 = ease((t - 0.9) / 0.35)
        f = text(f, "出结果", (W // 2, 1170), int(170 * (1.5 - 0.5 * k3)), AMBER, alpha=k3)
        f = text(f, "手机拍照 · 自动对齐 · 裂缝变化直接读数", (W // 2, 1400), 40, GREY, bold=False, alpha=ease((t - 1.5) / 0.4))
        return f, ("slogan" if p == 1 else None)
    scene(2.6, s3)

    # ---------------- 4. 步骤一：扫码
    def s4(p, t):
        f = header(bg_frame(), "第 1 步", "贴一个二维码，扫一下", "测点编号就此锁定，照片自动归档")
        f = paste_round(f, ref_s, IX, IY + 60)
        k = ease((t - 0.4) / 1.0)
        big = np.array([[60, 60], [IW - 60, 60], [IW - 60, IH - 60], [60, IH - 60]], np.float32)
        q = big * (1 - k) + qr * k
        col = GOOD if k >= 1 else AMBER
        pts = (q + [IX, IY + 60]).astype(np.int32)
        for i in range(4):
            a, b = pts[i], pts[(i + 1) % 4]
            cv2.line(f, tuple(int(v) for v in a), tuple(int(v) for v in a + (b - a) // 4), col, 8, cv2.LINE_AA)
            cv2.line(f, tuple(int(v) for v in b), tuple(int(v) for v in b - (b - a) // 4), col, 8, cv2.LINE_AA)
        if t > 1.6:
            f = pill(f, "已锁定测点  B01-Z023", (W // 2, IY + 60 + IH + 110), 46, GOOD)
        if t > 2.3:
            f = text(f, "不贴码也行：按位置 + 朝向 + 画面特征自动识别", (W // 2, IY + 60 + IH + 220), 34, GREY, bold=False,
                     alpha=ease((t - 2.3) / 0.4))
        return f, ("step1" if p == 1 else None)
    scene(4.2, s4)

    # ---------------- 5. 步骤二：复拍对齐
    def s5(p, t):
        f = header(bg_frame(), "第 2 步", "下次来，随手再拍一张", "角度偏了、远近不同，都自动对齐")
        y = IY + 60
        if t < 1.5:
            f = paste_round(f, new_s, IX, y)
            f = pill(f, "这次拍的（角度不一样）", (W // 2, y + IH + 110), 42, (60, 76, 92))
        elif t < 2.5:
            k = ease((t - 1.5) / 1.0)
            f = paste_round(f, new_s, IX, y); f = paste_round(f, al_s, IX, y, alpha=k)
            f = pill(f, "自动对齐中…", (W // 2, y + IH + 110), 42, AMBER, INK)
        else:
            sp = 0.5 + 0.38 * np.sin((t - 2.5) * 2.2)
            mix = ref_s.copy(); cut = int(IW * sp); mix[:, cut:] = al_s[:, cut:]
            f = paste_round(f, mix, IX, y)
            cv2.line(f, (IX + cut, y), (IX + cut, y + IH), AMBER, 6, cv2.LINE_AA)
            f = text(f, "上次", (IX + 90, y + 50), 40, stroke=3); f = text(f, "这次", (IX + IW - 90, y + 50), 40, stroke=3)
            f = pill(f, "已对齐 · 对齐误差 < 1 像素", (W // 2, y + IH + 110), 42, GOOD)
        return f, ("step2" if abs(t - 4.0) < 0.02 else None)
    scene(6.0, s5)

    # ---------------- 6. 步骤三：裂缝标红 + 读数
    def s6(p, t):
        f = header(bg_frame(), "第 3 步", "新增的裂缝，自动标红")
        img = crop_z.copy()
        a = ease(t / 1.6); b = ease((t - 1.9) / 2.0)
        img = draw_skel(img, old_pts, a, (255, 220, 0))
        img = draw_skel(img, new_pts, b, RED, 6)
        zy = 430
        f = paste_round(f, img, IX, zy)
        yb = zy + ZH + 30
        f = text(f, "■ 原有裂缝", (IX + 20, yb + 30), 36, (255, 220, 0), anchor="lm", alpha=ease(t / 0.5))
        f = text(f, "■ 新增", (IX + 300, yb + 30), 36, RED, anchor="lm", alpha=ease((t - 1.9) / 0.5))
        if b > 0:
            f = text(f, f"+{G * b:.1f} mm", (W // 2, yb + 150), 150, RED)
            f = text(f, f"总长 {L_ref:.1f} → {L_ref + (L_new - L_ref) * b:.1f} mm", (W // 2, yb + 262), 46, WHITE, bold=False)
        return f, ("step3" if p == 1 else None)
    scene(6.2, s6)

    # ---------------- 7. 结论卡（App 真实截图）
    def s7(p, t):
        f = header(bg_frame(), "结论", "报告，自动写好", "长度、宽度、新增多少、要不要关注")
        k = ease(t / 0.7)
        y = int(H - (H - 520) * k)
        f = paste_round(f, card, (W - 900) // 2, y, r=40)
        return f, ("report" if p == 1 else None)
    scene(4.2, s7)

    # ---------------- 8. 走势（App 真实截图）+ 卖点
    feats = ["只贴一个二维码，像素直接换算成毫米", "不联网、不用训练，手机离线就能跑", "每次复拍自动归档，走势一目了然",
             "反光、模糊、偏暗，当场提醒重拍"]

    def s8(p, t):
        f = bg_frame()
        f = text(f, "不止是拍照", (W // 2, 260), 96)
        for i, s in enumerate(feats):
            a = ease((t - 0.3 - i * 0.45) / 0.4)
            y = 480 + i * 150
            if a > 0:
                cv2.circle(f, (130, y), 26, AMBER, -1, cv2.LINE_AA)
                cv2.polylines(f, [np.array([[118, y + 1], [127, y + 10], [143, y - 9]], np.int32)], False, INK, 5, cv2.LINE_AA)
                f = text(f, s, (190, y), 42, WHITE, bold=False, anchor="lm", alpha=a)
        k = ease((t - 2.2) / 0.7)
        if k > 0:
            f = paste_round(f, zone[:760], (W - 900) // 2, int(H - (H - 1110) * k), r=40)
        return f, ("features" if p == 1 else None)
    scene(5.0, s8)

    # ---------------- 9. 片尾
    def s9(p, t):
        f = bg_frame()
        f = text(f, "复拍对齐相机", (W // 2, 800), 130, alpha=ease(t / 0.5))
        f = text(f, "让每一条裂缝，都有据可查", (W // 2, 980), 58, AMBER, alpha=ease((t - 0.5) / 0.5))
        f = text(f, "隧道 · 桥梁 · 铁路 · 水运工程 病害复拍对比", (W // 2, 1110), 38, GREY, bold=False, alpha=ease((t - 0.9) / 0.5))
        f = text(f, "画面为测试样板演示，数值来自该样板的实测结果", (W // 2, 1780), 28, (120, 132, 145), bold=False, alpha=ease((t - 1.2) / 0.5))
        return f, ("end" if p == 1 else None)
    scene(3.6, s9)

    vw.release()
    for k, im in stills.items():
        cv2.imwrite(os.path.join(HERE, "素材", f"画面_{k}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("已生成", out_path, f"{os.path.getsize(out_path) / 1e6:.1f} MB；配图 {len(stills)} 张")


if __name__ == "__main__":
    main()

"""
生成可打印靶标（A4，300dpi）。

  python make_target.py --zone B01-Z023              # 输出 targets/B01-Z023.pdf/.png
  python make_target.py --zone B01-Z023 B01-Z024     # 一次生成多个测区

每页包含：
  * 一个二维码（码区 40mm，内容=测区编号，纠错等级 H）
  * 6 个直径 16mm 的黑色圆点贴纸（剪下后自由贴，选 3 个即可，多的备用）
  * 一个“平板试验布局”示例：码和 3 个圆点已排好，整页贴在硬板上就能直接试验

尺寸说明：照片之间的对齐不依赖物理尺寸；但圆点直径/码边长 = 0.4 是算法里的
预设比例（fieldalign.DOT_TO_QR_RATIO），改尺寸时两者要一起改。
"""
import argparse
import os

import qrcode
from PIL import Image, ImageDraw, ImageFont

DPI = 300
MM = DPI / 25.4
A4 = (round(210 * MM), round(297 * MM))
QR_MM = 40.0
DOT_MM = 16.0


def font(size_px):
    for p in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
              "C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size_px)
    return ImageFont.load_default()


def qr_image(text: str) -> Image.Image:
    q = qrcode.QRCode(version=2, error_correction=qrcode.constants.ERROR_CORRECT_H,
                      box_size=20, border=0)
    q.add_data(text)
    q.make(fit=True)
    img = q.make_image(fill_color="black", back_color="white").convert("L")
    side = round(QR_MM * MM)
    return img.resize((side, side), Image.NEAREST)


def draw_dot(d: ImageDraw.ImageDraw, cx, cy, with_cut=True):
    r = DOT_MM * MM / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)
    if with_cut:  # 剪裁虚线框，留 6mm 白边
        m = r + 6 * MM
        for x0 in range(int(cx - m), int(cx + m), 24):
            d.line([x0, cy - m, x0 + 12, cy - m], fill=150, width=2)
            d.line([x0, cy + m, x0 + 12, cy + m], fill=150, width=2)
        for y0 in range(int(cy - m), int(cy + m), 24):
            d.line([cx - m, y0, cx - m, y0 + 12], fill=150, width=2)
            d.line([cx + m, y0, cx + m, y0 + 12], fill=150, width=2)


def make_page(zone: str) -> Image.Image:
    page = Image.new("L", A4, 255)
    d = ImageDraw.Draw(page)
    f_big, f_small = font(round(5 * MM)), font(round(3.2 * MM))
    d.text((15 * MM, 10 * MM), f"测区靶标  {zone}", fill=0, font=f_big)
    d.text((15 * MM, 18 * MM), "上半页：剪下二维码和圆点自由粘贴（圆点选3个，分散、不共线、不遮挡病害）。"
           "下半页：整体贴在硬板上用于试验。", fill=90, font=f_small)

    # ---- 上半页：剪贴用
    qr = qr_image(zone)
    qx, qy = round(20 * MM), round(32 * MM)
    page.paste(qr, (qx, qy))
    d.text((qx, qy + qr.height + 3 * MM), zone, fill=0, font=f_small)
    m = 8 * MM  # 二维码裁切框（含静区）
    d.rectangle([qx - m, qy - m, qx + qr.width + m, qy + qr.height + m + 6 * MM], outline=150, width=2)
    for i in range(6):
        cx = (95 + (i % 3) * 36) * MM
        cy = (45 + (i // 3) * 36) * MM
        draw_dot(d, cx, cy)

    # ---- 下半页：试验布局（带仿混凝土纹理，图像特征才有东西可匹配）
    top = 135 * MM
    d.line([10 * MM, top - 6 * MM, 200 * MM, top - 6 * MM], fill=150, width=2)
    d.text((15 * MM, top - 4 * MM), "试验布局（不要剪开；可用笔在纹理上画“裂缝”，复拍时测试变化检测）", fill=90, font=f_small)
    texture(d, 12 * MM, top + 2 * MM, 198 * MM, 290 * MM, seed=sum(map(ord, zone)))
    qx2, qy2 = round(85 * MM), round(top + 50 * MM)
    m2 = round(6 * MM)
    d.rectangle([qx2 - m2, qy2 - m2, qx2 + qr.width + m2, qy2 + qr.height + m2], fill=255)
    page.paste(qr, (qx2, qy2))
    for cx, cy in ((30, top / MM + 22), (178, top / MM + 40), (55, top / MM + 140)):
        r = (DOT_MM / 2 + 6) * MM
        d.ellipse([cx * MM - r, cy * MM - r, cx * MM + r, cy * MM + r], fill=255)   # 白色贴纸底
        draw_dot(d, cx * MM, cy * MM, with_cut=False)
    return page


def texture(d: ImageDraw.ImageDraw, x0, y0, x1, y1, seed=0):
    """浅色随机斑点和细线。斑点直径 ≤ 2.5mm，远小于 16mm 圆点，不会被误认成靶标。"""
    import random
    rnd = random.Random(seed)
    area_mm2 = (x1 - x0) * (y1 - y0) / (MM * MM)
    for _ in range(int(area_mm2 * 0.12)):
        x, y = rnd.uniform(x0, x1), rnd.uniform(y0, y1)
        r = rnd.uniform(0.25, 1.25) * MM
        g = rnd.randint(110, 205)
        if rnd.random() < 0.7:
            d.ellipse([x - r, y - r * rnd.uniform(0.5, 1), x + r, y + r], fill=g)
        else:
            d.polygon([(x + rnd.uniform(-r, r) * 1.6, y + rnd.uniform(-r, r) * 1.6) for _ in range(4)], fill=g)
    for _ in range(int(area_mm2 * 0.004)):
        x, y = rnd.uniform(x0, x1), rnd.uniform(y0, y1)
        L = rnd.uniform(3, 9) * MM
        import math
        a = rnd.uniform(0, math.pi)
        d.line([x, y, x + L * math.cos(a), y + L * math.sin(a)], fill=rnd.randint(130, 190), width=rnd.randint(2, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", nargs="+", default=["B01-Z023"], help="测区编号（会写进二维码）")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "targets"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pages = [make_page(z) for z in a.zone]
    for z, p in zip(a.zone, pages):
        p.save(os.path.join(a.out, f"{z}.png"), dpi=(DPI, DPI))
    pdf = os.path.join(a.out, f"targets_{a.zone[0]}{'_etc' if len(a.zone) > 1 else ''}.pdf")
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=DPI)
    print("已生成：", os.path.abspath(pdf))
    print("打印时选“实际大小/100%”，不要选“适合页面”。")


if __name__ == "__main__":
    main()

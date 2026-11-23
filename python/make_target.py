"""
生成可打印靶标（A4，300dpi）。

  python make_target.py --zone B01-Z023              # 输出 targets/B01-Z023.pdf/.png
  python make_target.py --zone B01-Z023 B01-Z024     # 一次生成多个测区
  python make_target.py --sheet T 1 24               # 批量编号贴纸：T0001…T0024，每页 6 个 + 编号对应表 CSV

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


CJK_FONTS = ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc", "C:/Windows/Fonts/msyh.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
HAS_CJK = any(os.path.exists(p) for p in CJK_FONTS)


def font(size_px):
    for p in CJK_FONTS + ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if os.path.exists(p):
            return ImageFont.truetype(p, size_px)
    try:
        return ImageFont.load_default(size_px)
    except TypeError:
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


# ---------------------------------------------------------------- 批量编号贴纸（和手机 App「生成靶标」同一版式）
CELL_W, CELL_H = 95.0, 92.0       # 每个标签（mm），A4 上 2 列 × 3 行
SHEET_COLS, SHEET_ROWS = 2, 3


def codes_for(prefix: str, start: int, count: int, digits: int = 4):
    return [f"{prefix}{n:0{digits}d}" for n in range(start, start + count)]


def _dashed_rect(d, x0, y0, x1, y1, fill=170, dash=24, w=2):
    for x in range(int(x0), int(x1), dash):
        d.line([x, y0, min(x + dash // 2, x1), y0], fill=fill, width=w)
        d.line([x, y1, min(x + dash // 2, x1), y1], fill=fill, width=w)
    for y in range(int(y0), int(y1), dash):
        d.line([x0, y, x0, min(y + dash // 2, y1)], fill=fill, width=w)
        d.line([x1, y, x1, min(y + dash // 2, y1)], fill=fill, width=w)


def draw_label(page, d, x0, y0, code, f_code, f_tip):
    """一个标签：二维码 40mm（内容 = 编号）+ 编号文字 + 3 个 16mm 圆点贴纸，虚线为裁切线。"""
    _dashed_rect(d, x0, y0, x0 + CELL_W * MM, y0 + CELL_H * MM)
    qr = qr_image(code)
    qx = round(x0 + (CELL_W - QR_MM) / 2 * MM)
    qy = round(y0 + 7 * MM)
    page.paste(qr, (qx, qy))
    tw = d.textlength(code, font=f_code)
    d.text((x0 + CELL_W / 2 * MM - tw / 2, qy + qr.height + 2 * MM), code, fill=0, font=f_code)
    tip = "复拍对齐靶标 · 二维码和圆点分开剪下" if HAS_CJK else "Cut out the QR code and the 3 dots separately"
    d.text((x0 + CELL_W / 2 * MM - d.textlength(tip, font=f_tip) / 2, qy + qr.height + 9.5 * MM), tip, fill=110, font=f_tip)
    for i in range(3):
        cx = x0 + (CELL_W / 2 + (i - 1) * 27) * MM
        cy = y0 + 76 * MM
        r = DOT_MM / 2 * MM
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)
        m = r + 4.5 * MM
        _dashed_rect(d, cx - m, cy - m, cx + m, cy + m, fill=200, dash=16)


def make_sheet(codes):
    """返回 PIL 页面列表（A4，300dpi），每页 6 个标签。"""
    f_code, f_tip, f_head = font(round(6 * MM)), font(round(2.6 * MM)), font(round(3 * MM))
    per = SHEET_COLS * SHEET_ROWS
    pages = []
    for p0 in range(0, len(codes), per):
        page = Image.new("L", A4, 255)
        d = ImageDraw.Draw(page)
        chunk = codes[p0:p0 + per]
        head = (f"复拍对齐靶标  {chunk[0]} – {chunk[-1]}    按 100%（实际大小）打印，二维码边长应为 40 mm" if HAS_CJK else
                f"Targets {chunk[0]} - {chunk[-1]}    Print at 100% (actual size): QR side must be 40 mm")
        d.text((10 * MM, 4.5 * MM), head, fill=90, font=f_head)
        for i, c in enumerate(chunk):
            x0 = (210 - SHEET_COLS * CELL_W) / 2 * MM + (i % SHEET_COLS) * CELL_W * MM
            y0 = 11 * MM + (i // SHEET_COLS) * CELL_H * MM
            draw_label(page, d, x0, y0, c, f_code, f_tip)
        pages.append(page)
    return pages


def sheet_pdf(codes, path):
    pages = make_sheet(codes)
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=DPI)
    return path


def mapping_csv(codes) -> str:
    """编号对应表：贴好后填上项目编号/名称，再导入手机，测点会自动改名。"""
    return "\ufeff编号,项目编号,名称,备注\n" + "".join(f"{c},,,\n" for c in codes)


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
    ap.add_argument("--sheet", nargs=3, metavar=("前缀", "起始号", "数量"), help="批量编号贴纸，如 --sheet T 1 24")
    ap.add_argument("--digits", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.sheet:
        codes = codes_for(a.sheet[0], int(a.sheet[1]), int(a.sheet[2]), a.digits)
        base = os.path.join(a.out, f"靶标_{codes[0]}-{codes[-1]}")
        sheet_pdf(codes, base + ".pdf")
        with open(base + "_编号对应表.csv", "w", encoding="utf-8") as f:
            f.write(mapping_csv(codes))
        print("已生成：", os.path.abspath(base + ".pdf"), "及编号对应表 CSV")
        print("打印时选“实际大小/100%”，不要选“适合页面”。")
        return
    pages = [make_page(z) for z in a.zone]
    for z, p in zip(a.zone, pages):
        p.save(os.path.join(a.out, f"{z}.png"), dpi=(DPI, DPI))
    pdf = os.path.join(a.out, f"targets_{a.zone[0]}{'_etc' if len(a.zone) > 1 else ''}.pdf")
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=DPI)
    print("已生成：", os.path.abspath(pdf))
    print("打印时选“实际大小/100%”，不要选“适合页面”。")


if __name__ == "__main__":
    main()

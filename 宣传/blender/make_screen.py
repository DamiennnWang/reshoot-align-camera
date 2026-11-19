"""手机屏幕贴图：App 拍照界面（扫码锁定状态）示意。画面是样板墙面，配 App 的配色和锁定条。"""
import os, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
F = "/System/Library/Fonts/Hiragino Sans GB.ttc"
W, H = 1080, 2400
wall = cv2.imread(os.path.join(HERE, "tex", "wall_ref.png"))
# 竖屏取景：墙面中部（裂缝 + 二维码），按屏幕比例裁切
CX0, CY0, CX1, CY1 = 700, 75, 1950, 2125
crop = wall[CY0:CY1, CX0:CX1]
view = cv2.resize(crop, (W, int(W * crop.shape[0] / crop.shape[1])))
canvas = np.full((H, W, 3), 12, np.uint8)
y0 = 260
canvas[y0:y0 + view.shape[0]] = view[:H - y0]
img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
d = ImageDraw.Draw(img)
f = lambda s: ImageFont.truetype(F, s, index=2)
AMBER, GOOD, INK = (255, 176, 32), (34, 200, 120), (22, 32, 42)
# 顶部锁定条
d.rounded_rectangle((40, 90, W - 40, 230), 36, fill=INK)
d.ellipse((80, 135, 130, 185), fill=GOOD)
d.line((92, 160, 103, 172, 120, 148), fill=(255, 255, 255), width=7)
d.text((160, 118), "已锁定 B01-Z023", font=f(54), fill=(255, 255, 255))
d.text((W - 210, 130), "解锁", font=f(44), fill=AMBER)
# 二维码框（二维码在裁切图中的位置）
k = W / (CX1 - CX0)
qx0, qy0, qx1, qy1 = int((1455 - CX0) * k), int((895 - CY0) * k) + y0, int((1905 - CX0) * k), int((1345 - CY0) * k) + y0
L = 70
for (x, y, dx, dy) in ((qx0, qy0, 1, 1), (qx1, qy0, -1, 1), (qx0, qy1, 1, -1), (qx1, qy1, -1, -1)):
    d.line((x, y, x + dx * L, y), fill=GOOD, width=12)
    d.line((x, y, x, y + dy * L), fill=GOOD, width=12)
d.rounded_rectangle((qx0 - 20, qy1 + 30, qx0 + 330, qy1 + 110), 30, fill=GOOD)
d.text((qx0 + 10, qy1 + 44), "比例尺 40 mm", font=f(44), fill=(255, 255, 255))
# 底部快门
d.rectangle((0, H - 380, W, H), fill=(12, 12, 12))
d.ellipse((W // 2 - 110, H - 300, W // 2 + 110, H - 80), outline=(255, 255, 255), width=12)
d.ellipse((W // 2 - 88, H - 278, W // 2 + 88, H - 102), fill=(255, 255, 255))
d.text((80, H - 230), "复拍", font=f(46), fill=AMBER)
img.save(os.path.join(HERE, "tex", "scan_screen.png"))
print("ok", (qx0, qy0, qx1, qy1))

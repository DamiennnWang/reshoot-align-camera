"""导出 Blender 用贴图：样板墙面（与 samples/promo 同一块墙，400×275 mm，8 px/mm）、原有/新增裂缝掩膜。"""
import os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
import synth_test as st, test_measure as tm

out = os.path.join(HERE, "tex"); os.makedirs(out, exist_ok=True)
cv2.imwrite(os.path.join(out, "wall_ref.png"), tm.plane(8, 5))
cv2.imwrite(os.path.join(out, "wall_new.png"), tm.plane(8, 8))
pts = tm.crack_pts()
for name, a, b in (("crack_old", 0, 5), ("crack_new", 5, 8)):
    m = np.zeros(st.PLANE[::-1], np.uint8)
    for i in range(a, b):
        cv2.line(m, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), 255, 10, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out, name + ".png"), m)
print(st.PLANE, st.PX_PER_MM, pts.round(0).tolist())

# 贴花版：边缘羽化的 RGBA，贴到隧道衬砌上不露方框
for name in ("wall_ref", "wall_new"):
    g = cv2.imread(os.path.join(out, name + ".png"), 0)
    h, w = g.shape
    fx = np.clip(np.minimum(np.arange(w), w - 1 - np.arange(w)) / (0.06 * w), 0, 1)
    fy = np.clip(np.minimum(np.arange(h), h - 1 - np.arange(h)) / (0.06 * h), 0, 1)
    a = (np.outer(fy, fx) ** 0.8 * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out, name + "_decal.png"), np.dstack([g, g, g, a]))

# 只保留标记（二维码、三个圆点）和裂缝的贴花：背景透明，露出隧道衬砌
for name in ("wall_ref", "wall_new"):
    g = cv2.imread(os.path.join(out, name + ".png"), 0)
    white = cv2.morphologyEx((g >= 238).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mk = np.zeros_like(g)
    for c in cs:
        if cv2.contourArea(c) > 3000:
            cv2.drawContours(mk, [cv2.convexHull(c)], -1, 255, -1)
    mk = cv2.dilate(mk, np.ones((5, 5), np.uint8))
    upto = 5 if name == "wall_ref" else 8
    cr = np.zeros_like(g)
    for i in range(upto):
        cv2.line(cr, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), 255, 14, cv2.LINE_AA)
    a = cv2.GaussianBlur(np.maximum(mk, cr), (0, 0), 2.0)
    rgb = g.copy()
    rgb[cr > 0] = np.minimum(g[cr > 0], 70)          # 裂缝两侧略暗，像真实开裂的阴影
    rgb[(cr > 0) & (mk == 0)] = np.where(g[(cr > 0) & (mk == 0)] < 60, g[(cr > 0) & (mk == 0)], 70)
    cv2.imwrite(os.path.join(out, name + "_marks.png"), np.dstack([rgb, rgb, rgb, a]))
    print(name, "markers", sum(cv2.contourArea(c) > 3000 for c in cs))

"""宣传用演示照片：上次（正常光线）与今天（换角度、换远近、偏暖偏暗、一侧有阴影渐变）。裂缝从 5 段长到 8 段。"""
import os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import synth_test as st, test_measure as tm, fieldalign as fa, measure as ms

out = os.path.join(HERE, "..", "samples", "promo"); os.makedirs(out, exist_ok=True)
H_ref = st.camera_h(100, tilt=0.05, rot_deg=5)
ref = st.render(tm.plane(8, 5), H_ref, 100)
Hc = st.camera_h(777, tilt=0.14, rot_deg=22, zoom=0.82)
new = st.render(tm.plane(8, 8), Hc, 777, light=0.45, blur=1.0).astype(np.float32)
h, w = new.shape[:2]
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
shade = 0.62 + 0.38 / (1 + np.exp(-((xx / w) * 1.2 + (yy / h) * 0.5 - 0.75) * 7))     # 一侧暗下去
new = new * shade[..., None] * np.array([0.78, 0.92, 1.08])                              # 偏暖（BGR）
new = np.clip(new * 0.9, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(out, "last_month.jpg"), ref, [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite(os.path.join(out, "today.jpg"), new, [cv2.IMWRITE_JPEG_QUALITY, 92])
rm = fa.detect_markers(ref); dots = fa.select_reference_dots(rm)
res, rw, nw = fa.align(ref, rm, dots, new)
print("对齐：", res.status, res.message, "| 内点", res.feature_inliers, "| 偏差", res.consistency_px)
al, valid = fa.warp_to_ref(nw, res, rm.work_size)
excl = fa.comparison_mask(rm.work_size, res, rm.qr_corners, dots)
c, mask, _ = fa.compare(rw, al, valid, excl, res_ratio=fa.resolution_ratio(res))
cm = ((excl > 0) & (valid > 0) & (mask != 64)).astype(np.uint8)
ch = ms.compare_cracks(fa.gray_of(rw), fa.gray_of(al), cm)
sc = ms.scale_from_qr(rm.qr_corners, 40.0); ys, xs = np.nonzero(ch.new.skel)
mm = sc.mm_per_px((xs.mean(), ys.mean()))
print("结论：", ms.conclude(ch, mm, len(c.light_regions), c.light_ratio))

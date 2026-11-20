"""检测下限探查（只在电脑上跑，不进测试集）：更小的剥落、更小的张开、更小的螺母转角、更强的光照差。"""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_defect_samples as g, synth_test as st, defects as dfx, fieldalign as fa, measure as ms

OUT = os.path.join(st.OUT, "defects_limits"); os.makedirs(OUT, exist_ok=True)
H_ref = st.camera_h(100, tilt=0.05, rot_deg=5)
p0 = g.to_bgr(g.base(1))
cases = []
for r in (6, 9, 13):
    m = g.blob(300, 195, r, r * 0.7, 700 + r); newp, a = g.spall(p0, m, 700 + r)
    cases.append((f"spall_r{r}", p0, newp, dict(kind="spall", area=float((a >= 0.5 * a.max()).sum()) / 64)))
for d in (0.3, 0.5, 1.0):
    cases.append((f"joint_{d}", g.segments(), g.segments(dx_mm=d), dict(kind="joint", disp=d)))
for deg in (5, 10, 15):
    cases.append((f"bolt_{deg}", g.bolts(), g.bolts(rot={1: deg}), dict(kind="bolt", deg=deg)))
cases.append(("harsh_light_nochange", p0, p0, dict(kind="none")))
for i, (name, a_, b_, t) in enumerate(cases):
    harsh = name.startswith("harsh")
    Hn = st.camera_h(800 + i, tilt=0.13 if harsh else 0.11, rot_deg=20 if harsh else 16, zoom=0.85)
    ref = g.render(a_, H_ref, 100)
    new = g.render(b_, Hn, 800 + i, light=0.7 if harsh else 0.35, blur=1.2, tint=(0.8, 1.0, 1.25) if harsh else (0.9, 1.0, 1.12), gain=0.7 if harsh else 0.9)
    rm = fa.detect_markers(ref); dots = fa.select_reference_dots(rm)
    res, rw, nw = fa.align(ref, rm, dots, new)
    al, valid = fa.warp_to_ref(nw, res, rm.work_size)
    excl = fa.comparison_mask(rm.work_size, res, rm.qr_corners, dots)
    c, change, _ = fa.compare(rw, al, valid, excl, res_ratio=fa.resolution_ratio(res))
    mask = ((excl > 0) & (valid > 0)).astype(np.uint8) * 255
    mfull = ((fa.marker_mask(rm.work_size, rm.qr_corners, dots) > 0) & (valid > 0)).astype(np.uint8) * 255
    mmpp = ms.scale_from_qr(rm.qr_corners, 40.0).mm_per_px
    D = dfx.displacement(fa.gray_of(rw), fa.gray_of(al), mfull if t["kind"] == "joint" else mask, mmpp)
    A = dfx.areas(rw, dfx.compensate(al, D), mask, change, mmpp)
    M = dfx.marks(rw, al, mask) if t["kind"] == "bolt" else []
    out = []
    if A: out.append("面状 " + "，".join(f"{x.cls} {x.mm2:.0f}mm²" for x in A))
    if D and (D.mm or 0) >= 0.5: out.append(f"位移 {D.mm:.2f}mm 张开 {D.opening:.2f}")
    elif D: out.append(f"(位移 {D.mm:.2f}mm 低于报告门槛)")
    if M: out.append("标记 " + "，".join(f"{x.status}{x.dang:.0f}°" for x in M))
    print(f"{name:<22} 真值 {t}  →  {'；'.join(out) or '无发现'}  [{res.status}, 1px={mmpp((1000,750)):.2f}mm]")

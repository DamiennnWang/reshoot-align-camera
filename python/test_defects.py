"""小场景病害合成样例评估：先 make_defect_samples.py 生成样例，再运行本脚本。"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import defects as dfx  # noqa: E402
import fieldalign as fa  # noqa: E402
import measure as ms  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples", "defects")


def run(name):
    ref = fa.load_image(os.path.join(D, f"{name}_ref.jpg")); new = fa.load_image(os.path.join(D, f"{name}_new.jpg"))
    rm = fa.detect_markers(ref); dots = fa.select_reference_dots(rm)
    res, rw, nw = fa.align(ref, rm, dots, new)
    al, valid = fa.warp_to_ref(nw, res, rm.work_size)
    excl = fa.comparison_mask(rm.work_size, res, rm.qr_corners, dots)
    c, change, _ = fa.compare(rw, al, valid, excl, res_ratio=fa.resolution_ratio(res))
    mask = ((excl > 0) & (valid > 0)).astype(np.uint8) * 255
    mfull = ((fa.marker_mask(rm.work_size, rm.qr_corners, dots) > 0) & (valid > 0)).astype(np.uint8) * 255
    sc = ms.scale_from_qr(rm.qr_corners, 40.0)
    return dict(res=res, rw=rw, al=al, mask=mask, mfull=mfull, change=change, mmpp=sc.mm_per_px)


def main():
    truth = json.load(open(os.path.join(D, "truth.json"), encoding="utf-8"))
    fails = 0
    for name, t in truth.items():
        r = run(name)
        dd = {}
        Dp = dfx.displacement(fa.gray_of(r["rw"]), fa.gray_of(r["al"]), r["mfull"], r["mmpp"], debug=dd)   # 位移类测点：不限于一致区
        al = dfx.compensate(r["al"], Dp)
        dbg = {}
        A = dfx.areas(r["rw"], al, r["mask"], r["change"], r["mmpp"], debug=dbg)
        Mk = dfx.marks(r["rw"], r["al"], r["mask"])
        Pt = dfx.parts(fa.gray_of(r["rw"]), fa.gray_of(r["al"]), r["change"], r["mask"])
        k = t["kind"]
        ok = True
        big = [a for a in A if (a.mm2 or 0) >= 100]
        if k in ("spall", "rust", "seep", "effl"):
            want = {"spall": "剥落/掉块", "rust": "锈蚀", "seep": "渗水/湿痕", "effl": "泛碱/白化"}[k]
            hit = [a for a in big if a.cls == want]
            ok = len(hit) == 1 and abs(hit[0].mm2 / t["area_mm2"] - 1) < 0.35 and len(big) == 1
        elif k in ("offset", "joint"):
            ok = Dp is not None and abs(Dp.mm - t["disp_mm"]) < 0.5 and not big
        elif k == "bolt_small":
            st = [m.status for m in Mk]
            ok = st.count("疑似松动") == 1 and st.count("正常") == 3
        elif k == "bolt":
            st = [m.status for m in Mk]
            ok = st.count("疑似松动") == 1 and st.count("正常") == 2 and len(Mk) == 4
        else:
            ok = not big and Dp is None and all(m.status == "正常" for m in Mk)
        fails += not ok
        print(f"{'✓' if ok else '✗'} {name:<18} {r['res'].status:<10} 真值 {json.dumps({x: y for x, y in t.items() if x not in ('bolts_mm',)}, ensure_ascii=False)}")
        for a in A:
            print(f"     面状 {a.cls:<6} {a.mm2 or 0:8.0f} mm²  dL {a.dL:+.0f}  box {a.box}")
        if Dp:
            print(f"     位移 {Dp.mm:.2f} mm  (dx {Dp.dx:+.1f} dy {Dp.dy:+.1f} px) 张开 {Dp.opening} 错动 {Dp.shear} 块 {Dp.blocks}")
        for m in Mk:
            print(f"     标记 ({m.cx:.0f},{m.cy:.0f}) {m.status} 转角 {m.dang:.1f}° 覆盖 {m.coverage:.2f}")
        for p in Pt:
            print(f"     部件 {p.box} {p.status} ncc {p.ncc:.2f} E {p.e_ratio:.2f} d {p.d_mean:+.0f}")
        print(f"     调试 面状 {dbg}  位移 {dd}")
    print(f"\n{len(truth) - fails}/{len(truth)} 通过")
    return fails


if __name__ == "__main__":
    sys.exit(main())

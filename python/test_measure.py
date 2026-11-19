"""
裂缝测量精度测试（合成照片，有真值）：
  * 三种宽度 0.25 / 0.5 / 1.0 mm，参考照片裂缝 5 段，复拍延长到 8 段；
  * 比例尺只用参考照片里 40mm 二维码；
  * 无裂缝墙面检查误报。
输出长度、宽度、新增长度与真值的误差，并把标注图写到 samples/measure/。

  python test_measure.py
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldalign as fa  # noqa: E402
import measure as ms  # noqa: E402
import synth_test as st  # noqa: E402


CRACK_SHIFT = np.array([-700.0, 0.0])     # 裂缝放在二维码左侧，不穿过靶标（靶标区不参与比较）


def crack_pts():
    return st.crack_points(7) + CRACK_SHIFT


def plane(width_px, upto, tex=1, strength=1.0):
    p = st.texture(tex, strength)
    pts = crack_pts()
    for i in range(min(upto, len(pts) - 1)):
        cv2.line(p, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), 45, width_px, cv2.LINE_AA)
    st.paste_target(p, "B01-Z023", st.DOTS_MM)
    return p


def seg_len_mm(a, b):
    pts = crack_pts()
    return float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(a, b))) / st.PX_PER_MM


def true_mmpp(H_cam, s_work, plane_xy):
    """真值：参考工作图在某平面点处 1 像素对应的毫米数。"""
    A = np.diag([s_work, s_work, 1.0]) @ H_cam
    p = np.array([plane_xy[0], plane_xy[1], 1.0])

    def m(q):
        v = A @ q
        return v[:2] / v[2]
    a = m(p + [1, 0, 0]) - m(p)
    b = m(p + [0, 1, 0]) - m(p)
    px_per_planepx = np.sqrt(abs(a[0] * b[1] - a[1] * b[0]))
    return 1.0 / (px_per_planepx * st.PX_PER_MM)


def main():
    out = os.path.join(st.OUT, "measure"); os.makedirs(out, exist_ok=True)
    H_ref = st.camera_h(100, tilt=0.05, rot_deg=5)
    ok_all = True
    truth = {}
    L_ref, L_new = seg_len_mm(0, 5), seg_len_mm(0, 8)
    centre = crack_pts()[:6].mean(0)
    print(f"真值：参考长 {L_ref:.1f} mm，复拍长 {L_new:.1f} mm，新增 {L_new - L_ref:.1f} mm\n")
    print(f"{'用例':<16}{'比例 mm/px 测/真':<20}{'长度 mm 测/真':<20}{'最大宽 mm 测/真':<20}{'平均宽':<10}{'新增 mm 测/真':<18}{'结论'}")
    print("-" * 120)
    for name, wpx, seed, strength in (("w0.25mm", 2, 301, 1.0), ("w0.5mm", 4, 302, 1.0), ("w1.0mm", 8, 303, 1.0),
                                      ("w0.5mm_rough", 4, 305, 1.4), ("no_crack", 0, 304, 1.0), ("no_crack_rough", 0, 306, 1.4)):
        ref_img = st.render(plane(wpx, 5 if wpx else 0, strength=strength), H_ref, 100)
        Hc = st.camera_h(seed, tilt=0.08)
        new_img = st.render(plane(wpx, 8 if wpx else 0, strength=strength), Hc, seed)
        rm = fa.detect_markers(ref_img); dots = fa.select_reference_dots(rm)
        res, rw, nw = fa.align(ref_img, rm, dots, new_img)
        al, valid = fa.warp_to_ref(nw, res, rm.work_size)
        mask = fa.comparison_mask(rm.work_size, res, rm.qr_corners, dots) & (valid > 0)
        ch = ms.compare_cracks(fa.gray_of(rw), fa.gray_of(al), mask.astype(np.uint8))
        sc = ms.scale_from_qr(rm.qr_corners, ms.qr_size_mm(rm.qr_text))
        # 比例取裂缝中心处
        c_work = fa._ph(np.diag([rm.work_scale, rm.work_scale, 1]) @ H_ref, [centre])[0]
        mmpp = sc.mm_per_px(c_work)
        mmpp_true = true_mmpp(H_ref, rm.work_scale, centre)
        level, text = ms.conclude(ch, mmpp)
        ok = True
        if wpx:
            Ln = ch.new.total_px * mmpp; Wn = ch.new.max_w_px * mmpp; Wm = (ch.new.cracks[0].mean_w_px * mmpp) if ch.new.cracks else 0
            G = ch.grow_px * mmpp
            w_true = wpx / st.PX_PER_MM
            ok &= abs(mmpp / mmpp_true - 1) < 0.05
            ok &= abs(Ln / L_new - 1) < 0.10
            ok &= abs(G / (L_new - L_ref) - 1) < 0.25
            if w_true / mmpp < ms.RES_LIMIT_PX:                  # 细于分辨率：要求标出“测不准”，且不夸大
                ok &= ch.new.cracks[0].limited and ch.new.max_w_px <= 2.8
            else:
                ok &= abs(Wn - w_true) < max(0.3 * w_true, 1.25 * mmpp)  # 宽度：误差 < 30% 或 1.25 像素
            ok &= level.startswith("延长")
            print(f"{name:<16}{mmpp:.3f}/{mmpp_true:.3f}{'':<8}{Ln:6.1f}/{L_new:6.1f}{'':<7}{Wn:5.2f}/{w_true:5.2f}{'':<9}"
                  f"{Wm:5.2f}{'':<5}{G:6.1f}/{L_new - L_ref:5.1f}{'':<6}{level}{'' if ok else '  ✗'}")
        else:
            ok &= len(ch.new.cracks) == 0 and len(ch.ref.cracks) == 0
            print(f"{name:<16}{mmpp:.3f}/{mmpp_true:.3f}{'':<8}{'裂缝 %d/%d 条' % (len(ch.new.cracks), len(ch.ref.cracks)):<20}"
                  f"{'':<40}{level}{'' if ok else '  ✗'}")
        ok_all &= ok
        cv2.imwrite(os.path.join(out, f"{name}_ref.jpg"), ref_img)
        cv2.imwrite(os.path.join(out, f"{name}_new.jpg"), new_img)
        cv2.imwrite(os.path.join(out, f"{name}_cracks.jpg"), ms.draw_cracks(al, ch, mmpp))
        truth[name] = {"width_mm": wpx / st.PX_PER_MM, "len_ref_mm": L_ref if wpx else 0, "len_new_mm": L_new if wpx else 0,
                       "mmpp_true": mmpp_true}
        if name == "w0.5mm":
            print("\n  结论示例：" + text + "\n")
    with open(os.path.join(out, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    print("\n全部符合预期" if ok_all else "\n有用例不符合预期")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

"""
镜头畸变测试：给合成照片加桶形畸变（k<0，手机广角常见），检验
  * 能否估计出畸变并正确对齐（真值误差）
  * 状态判断不被畸变干扰（不能误判“圆点移动”）
  * 无变化时假变化少、裂缝延长仍能检出
同时测试“两张照片来自不同手机（畸变不同）”。

  python test_distortion.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldalign as fa  # noqa: E402
import synth_test as st  # noqa: E402


def distort_img(img, k):
    """把理想图像变成带径向畸变 k 的照片：输出像素 p_d 取理想图像中 undistort(p_d) 处的值。"""
    if k == 0:
        return img
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    u = fa.undistort_pts(np.c_[gx.ravel(), gy.ravel()], k, (w, h))
    return cv2.remap(img, u[:, 0].reshape(h, w).astype(np.float32), u[:, 1].reshape(h, w).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def main():
    ref_plane = st.make_plane(1, 5, st.DOTS_MM)
    H_ref = st.camera_h(100, tilt=0.05, rot_deg=5)
    print(f"{'畸变 k(参考/新)':<18}{'用例':<10}{'状态':<14}{'估计k(新/参考)':<20}{'误差均/最大px':<18}{'变化':<16}")
    print("-" * 100)
    ok_all = True
    out = os.path.join(st.OUT, "distortion"); os.makedirs(out, exist_ok=True)
    truth = {}
    for k_ref, k_new in ((-0.12, -0.12), (-0.20, -0.20), (-0.05, -0.18)):
        ref = distort_img(st.render(ref_plane, H_ref, 100), k_ref)
        tag = f"{k_ref:+.2f}_{k_new:+.2f}"
        cv2.imwrite(os.path.join(out, f"ref_{tag}.jpg"), ref)
        rm = fa.detect_markers(ref)
        dots = fa.select_reference_dots(rm)
        for name, plane, seed, expect_change in (("无变化", st.make_plane(1, 5, st.DOTS_MM), 211, False),
                                                 ("裂缝延长", st.make_plane(1, 8, st.DOTS_MM), 212, True)):
            Hc = st.camera_h(seed, tilt=0.12, rot_deg=15)
            new = distort_img(st.render(plane, Hc, seed), k_new)
            res, rw, nw = fa.align(ref, rm, dots, new)
            # 真值：新照片像素 → 去畸变(k_new) → 理想新图 → 理想参考图 → 加畸变(k_ref)
            Ht = st.true_new_to_ref(H_ref, Hc, rm.work_scale, fa.to_work(new)[1])
            nname = f"new_{tag}_{'nochange' if not expect_change else 'crack'}.jpg"
            cv2.imwrite(os.path.join(out, nname), new)
            truth[nname] = {"ref": f"ref_{tag}.jpg", "H_true": Ht.tolist(), "k_ref": k_ref, "k_new": k_new,
                            "expect_change": expect_change}
            grid = fa._roi_grid(rm.qr_corners, dots, rm.work_size)          # 参考（畸变）坐标
            g_ref_u = fa.undistort_pts(grid, k_ref, rm.work_size)
            g_new = fa.distort_pts(fa._ph(np.linalg.inv(Ht), g_ref_u), k_new, rm.work_size)
            e = np.linalg.norm(fa.map_new_to_ref(res, g_new) - grid, axis=1)
            al, v = fa.warp_to_ref(nw, res, rm.work_size)
            c, _, _ = fa.compare(rw, al, v, fa.comparison_mask(rm.work_size, res, rm.qr_corners, dots), res_ratio=fa.resolution_ratio(res))
            ok = res.status == "confirmed" and e.max() < 3.0 and ((len(c.regions) > 0) == expect_change or
                                                                 (not expect_change and c.changed_ratio < 0.001))
            ok_all &= ok
            print(f"{f'{k_ref:+.2f}/{k_new:+.2f}':<18}{name:<10}{res.status:<14}"
                  f"{f'{res.k_new:+.3f}/{res.k_ref:+.3f}':<20}{f'{e.mean():.2f}/{e.max():.2f}':<18}"
                  f"{f'{c.changed_ratio:.3%} {len(c.regions)}处':<16}{'' if ok else '  ✗'}")
    import json
    with open(os.path.join(out, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    print("\n全部符合预期" if ok_all else "\n有用例不符合预期")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

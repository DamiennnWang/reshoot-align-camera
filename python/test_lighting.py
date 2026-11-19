"""
光影测试：复拍时阴影位置、光照不同是现场常态。
  * 软 / 硬阴影：不能报“结构变化”，应归为“明暗变化”
  * 阴影下的裂缝延长：仍要报结构变化
  * 无变化：两类都不报

  python test_lighting.py
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldalign as fa  # noqa: E402
import synth_test as st  # noqa: E402


def shadowed(plane, strength=0.55, soft=9):
    """在平面上投一块阴影（乘性变暗，边缘柔和程度 soft）。"""
    p = plane.astype(np.float32)
    m = np.zeros(p.shape, np.float32)
    cv2.fillPoly(m, [np.int32([[0, 1400], [2000, 600], [3200, 900], [3200, 2200], [0, 2200]])], 1.0)
    m = cv2.GaussianBlur(m, (0, 0), soft)
    return np.clip(p * (1 - strength * m), 0, 255).astype(np.uint8)


def main():
    out = os.path.join(st.OUT, "lighting"); os.makedirs(out, exist_ok=True)
    ref = st.render(st.make_plane(1, 5, st.DOTS_MM), st.camera_h(100, tilt=0.05, rot_deg=5), 100)
    cv2.imwrite(os.path.join(out, "ref.jpg"), ref)
    rm = fa.detect_markers(ref)
    dots = fa.select_reference_dots(rm)
    cases = [
        ("soft_shadow", shadowed(st.make_plane(1, 5, st.DOTS_MM)), False, True),
        ("hard_shadow", shadowed(st.make_plane(1, 5, st.DOTS_MM), soft=2), False, True),
        ("hard_shadow_crack", shadowed(st.make_plane(1, 8, st.DOTS_MM), soft=2), True, True),
        ("crack", st.make_plane(1, 8, st.DOTS_MM), True, False),
        ("nochange", st.make_plane(1, 5, st.DOTS_MM), False, False),
    ]
    truth = {}
    ok_all = True
    print(f"{'用例':<20}{'结构变化':<22}{'明暗变化':<16}")
    print("-" * 60)
    for name, plane, exp_struct, exp_light in cases:
        new = st.render(plane, st.camera_h(220, tilt=0.1), 220)
        cv2.imwrite(os.path.join(out, f"{name}.jpg"), new)
        truth[f"{name}.jpg"] = {"expect_structural": exp_struct, "expect_light": exp_light}
        res, rw, nw = fa.align(ref, rm, dots, new)
        al, v = fa.warp_to_ref(nw, res, rm.work_size)
        c, _, _ = fa.compare(rw, al, v, fa.comparison_mask(rm.work_size, res, rm.qr_corners, dots), res_ratio=fa.resolution_ratio(res))
        ok = (len(c.regions) > 0) == exp_struct and (c.light_ratio > 0.001) == exp_light
        ok_all &= ok
        print(f"{name:<20}{f'{len(c.regions)}处 {c.changed_ratio:.3%}':<22}"
              f"{f'{len(c.light_regions)}处 {c.light_ratio:.3%}':<16}{'' if ok else '  ✗'}")
    with open(os.path.join(out, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    print("\n全部符合预期" if ok_all else "\n有用例不符合预期")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

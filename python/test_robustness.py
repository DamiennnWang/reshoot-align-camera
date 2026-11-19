"""
真实手机拍摄的退化情况压力测试：噪声 / 低质量 JPEG / 欠曝 / 过曝 / 运动模糊 / 偏色 /
镜面反光 / 竖拍 / 低分辨率。
每个用例检查：对齐状态、对齐误差、是否报结构变化（与期望一致）。

  python test_robustness.py
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldalign as fa  # noqa: E402
import synth_test as st  # noqa: E402

MAX_ERR_PX = 1.0


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def noise(img, sigma=9, q=60):
    rng = np.random.default_rng(1)
    f = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return jpeg(np.clip(f, 0, 255).astype(np.uint8), q)


def exposure(img, gain):
    return jpeg(np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8), 90)


def motion_blur(img, length=11, ang=30):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    R = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), ang, 1)
    k = cv2.warpAffine(k, R, (length, length)); k /= k.sum()
    return jpeg(cv2.filter2D(img, -1, k), 90)


def color_cast(img):
    return jpeg(np.clip(img.astype(np.float32) * np.array([1.35, 1.0, 0.7]), 0, 255).astype(np.uint8), 90)


def glare(img, center=(0.62, 0.62), radius=0.07):
    """镜面反光：一块饱和的亮斑（中心过曝成纯白，边缘渐变）。"""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - center[0] * w, yy - center[1] * h) / (radius * w)
    a = np.clip(1.6 - r, 0, 1)[..., None]
    f = img.astype(np.float32) * (1 - a) + 255 * a
    return jpeg(np.clip(f, 0, 255).astype(np.uint8), 90)


def portrait(img):
    return img  # 由相机矩阵决定，见下


def lowres(img, side=1600):
    s = side / max(img.shape[:2])
    return jpeg(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 85)


def main():
    out = os.path.join(st.OUT, "robustness"); os.makedirs(out, exist_ok=True)
    ref_plane = st.make_plane(1, 5, st.DOTS_MM)
    H_ref = st.camera_h(100, tilt=0.05, rot_deg=5)
    ref = st.render(ref_plane, H_ref, 100)
    rm = fa.detect_markers(ref)
    dots = fa.select_reference_dots(rm)
    same = st.make_plane(1, 5, st.DOTS_MM)
    crack = st.make_plane(1, 8, st.DOTS_MM)
    cases = [
        # 名称, 平面, 相机种子, 退化函数, 期望结构变化, 相机额外参数
        ("noise_jpeg60", same, 501, noise, False, {}),
        ("noise_crack", crack, 502, noise, True, {}),
        ("under_exposed", same, 503, lambda i: exposure(i, 0.35), False, {}),
        ("over_exposed", same, 504, lambda i: exposure(i, 1.7), False, {}),
        ("motion_blur", same, 505, motion_blur, False, {}),
        ("motion_blur_crack", crack, 506, motion_blur, True, {}),
        ("color_cast", same, 507, color_cast, False, {}),
        ("glare", same, 508, glare, False, {}),
        ("glare_crack", crack, 509, glare, True, {}),
        ("rotated_90", same, 510, lambda i: i, False, {"rot90": True}),
        ("lowres_1600", same, 511, lowres, False, {}),
        ("lowres_crack", crack, 512, lowres, True, {}),
    ]
    ok_all = True
    truth = {}
    print(f"{'用例':<20}{'状态':<14}{'误差 均/最大px':<18}{'结构变化':<16}{'明暗变化':<12}{'耗时':<8}")
    print("-" * 92)
    for name, plane, seed, fn, exp_struct, extra in cases:
        Hc = st.camera_h(seed, tilt=0.08)
        img = st.render(plane, Hc, seed)
        s_post = 1.0
        if extra.get("rot90"):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            W, H = st.PHOTO
            R = np.array([[0, -1, H - 1], [1, 0, 0], [0, 0, 1]], np.float64)
            Hc = R @ Hc
        new = fn(img)
        if new.shape[:2] != img.shape[:2]:
            s_post = new.shape[1] / img.shape[1]
            Hc = np.diag([s_post, s_post, 1.0]) @ Hc
        cv2.imwrite(os.path.join(out, f"{name}.jpg"), new)
        truth[f"{name}.jpg"] = {"expect_structural": exp_struct,
                                "H_true": st.true_new_to_ref(H_ref, Hc, rm.work_scale, fa.to_work(new)[1]).tolist()}
        t0 = time.time()
        res, rw, nw = fa.align(ref, rm, dots, new)
        err = "—"; struct = "—"; light = "—"; ok = res.status == "confirmed"
        if res.H is not None:
            Ht = st.true_new_to_ref(H_ref, Hc, rm.work_scale, fa.to_work(new)[1])
            grid = fa._roi_grid(rm.qr_corners, dots, rm.work_size)
            e = np.linalg.norm(fa.map_new_to_ref(res, fa._ph(np.linalg.inv(Ht), grid)) - grid, axis=1)
            err = f"{e.mean():.2f} / {e.max():.2f}"
            ok &= e.max() < MAX_ERR_PX
            al, v = fa.warp_to_ref(nw, res, rm.work_size)
            c, mask, _ = fa.compare(rw, al, v, fa.comparison_mask(rm.work_size, res, rm.qr_corners, dots),
                                    res_ratio=fa.resolution_ratio(res))
            struct = f"{len(c.regions)}处 {c.changed_ratio:.3%}"
            light = f"{c.light_ratio:.2%} 反光{c.glare_ratio:.1%} σ{c.blur_sigma:+.1f}"
            ok &= (len(c.regions) > 0) == exp_struct
            vis = fa.make_views(rw, al, v, mask, c.regions, c.light_regions)
            cv2.imwrite(os.path.join(out, f"{name}_changes.jpg"), vis["changes"])
        dt = time.time() - t0
        ok_all &= ok
        print(f"{name:<20}{res.status:<14}{err:<18}{struct:<16}{light:<12}{dt:<8.1f}{'' if ok else '  ✗'}")
    import json
    with open(os.path.join(out, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    print("\n全部符合预期" if ok_all else "\n有用例不符合预期")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

"""
合成测试：在“混凝土”纹理平面上贴靶标、画裂缝，模拟手机从不同角度复拍，
用已知的真实变换检验 fieldalign 的对齐精度和异常判断。

  python synth_test.py            # 生成 samples/ 下的测试照片并输出评估表
"""
import os
import sys
import zlib

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldalign as fa  # noqa: E402
from make_target import qr_image, QR_MM, DOT_MM, MM as PRINT_MM  # noqa: E402

PX_PER_MM = 8.0                  # 合成平面分辨率
PLANE = (3200, 2200)             # 400mm x 275mm
PHOTO = (4000, 3000)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples")


def texture(seed, strength=1.0):
    rng = np.random.default_rng(seed)
    w, h = PLANE
    acc = np.zeros((h, w), np.float32)
    for k, amp in ((4, 30), (16, 18), (64, 10), (256, 6)):
        n = rng.normal(0, 1, (h // k + 2, w // k + 2)).astype(np.float32)
        acc += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)[:h, :w] * amp
    img = 150 + acc * strength
    # 骨料斑点
    for _ in range(int(2500 * strength)):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 9)), float(rng.uniform(90, 200)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    return np.clip(img, 0, 255).astype(np.uint8)


def crack_points(seed, n=9):
    rng = np.random.default_rng(seed)
    x, y = 1500.0, 300.0
    pts = [(x, y)]
    for _ in range(n):
        x += rng.uniform(-60, 90)
        y += rng.uniform(140, 200)
        pts.append((x, y))
    return np.array(pts)


def draw_crack(img, pts, upto):
    for i in range(min(upto, len(pts) - 1)):
        cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), 45, 4, cv2.LINE_AA)


DOTS_MM = [(40, 40), (360, 60), (80, 230)]
QR_POS_MM = (190, 120)


def paste_target(img, text, dots_mm):
    qr = qr_image(text)
    side = int(QR_MM * PX_PER_MM)
    qr = cv2.resize(np.asarray(qr), (side, side), interpolation=cv2.INTER_NEAREST)
    qx, qy = int(QR_POS_MM[0] * PX_PER_MM), int(QR_POS_MM[1] * PX_PER_MM)
    m = int(8 * PX_PER_MM)
    cv2.rectangle(img, (qx - m, qy - m), (qx + side + m, qy + side + m), 245, -1)
    img[qy:qy + side, qx:qx + side] = qr
    for x, y in dots_mm:
        c = (int(x * PX_PER_MM), int(y * PX_PER_MM))
        cv2.circle(img, c, int(14 * PX_PER_MM), 245, -1, cv2.LINE_AA)
        cv2.circle(img, c, int(DOT_MM / 2 * PX_PER_MM), 15, -1, cv2.LINE_AA)


def camera_h(seed, tilt=0.10, rot_deg=12, flip=False, zoom=1.0):
    """平面 → 照片 的真实单应矩阵（随机透视）。"""
    rng = np.random.default_rng(seed)
    w, h = PLANE
    W, H = PHOTO
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    s = min(W / w, H / h) * rng.uniform(0.80, 0.92) * zoom
    c = np.array([W / 2 + rng.uniform(-150, 150), H / 2 + rng.uniform(-150, 150)])
    a = np.deg2rad(rng.uniform(-rot_deg, rot_deg) + (180 if flip else 0))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = (src - [w / 2, h / 2]) * s @ R.T + c
    dst += rng.uniform(-tilt, tilt, dst.shape) * np.array([W, H])
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32)).astype(np.float64)


def render(plane, Hc, seed, light=0.25, blur=0.8):
    rng = np.random.default_rng(seed)
    W, H = PHOTO
    bg = np.full((H, W), 70, np.uint8)
    img = cv2.warpPerspective(plane, Hc, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_TRANSPARENT, dst=bg.copy())
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 1 + light * ((xx / W - 0.5) * rng.uniform(-1, 1) + (yy / H - 0.5) * rng.uniform(-1, 1))
    gamma = rng.uniform(0.8, 1.25)
    f = 255 * ((img.astype(np.float32) / 255) ** gamma) * g * rng.uniform(0.85, 1.1)
    f = cv2.GaussianBlur(f, (0, 0), blur) + rng.normal(0, 3, f.shape)
    color = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    color = (color * np.array([1.0, 1.02, 1.05])).clip(0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def make_plane(tex_seed, crack_upto, dots_mm, text="B01-Z023", strength=1.0):
    p = texture(tex_seed, strength)
    draw_crack(p, crack_points(7), crack_upto)
    paste_target(p, text, dots_mm)
    return p


def true_new_to_ref(H_ref_cam, H_new_cam, s_ref, s_new):
    S = lambda s: np.diag([s, s, 1.0])
    return S(s_ref) @ H_ref_cam @ np.linalg.inv(H_new_cam) @ np.linalg.inv(S(s_new))


def main():
    os.makedirs(OUT, exist_ok=True)
    ref_plane = make_plane(1, 5, DOTS_MM)
    H_ref = camera_h(100, tilt=0.05, rot_deg=5)
    ref_img = render(ref_plane, H_ref, 100)
    cv2.imwrite(os.path.join(OUT, "00_reference.jpg"), ref_img)

    moved = list(DOTS_MM)
    moved[1] = (moved[1][0] - 20, moved[1][1] + 12)  # 圆点2 移动约 23mm
    cases = [
        ("01_normal", make_plane(1, 5, DOTS_MM), dict(seed=201), "confirmed"),
        ("02_crack_grew", make_plane(1, 8, DOTS_MM), dict(seed=202), "confirmed"),
        ("03_oblique_lighting", make_plane(1, 5, DOTS_MM), dict(seed=203, tilt=0.18, rot_deg=25), "confirmed"),
        ("04_upside_down", make_plane(1, 5, DOTS_MM), dict(seed=204, flip=True), "confirmed"),
        ("05_dot_moved", make_plane(1, 5, moved), dict(seed=205), "dot_moved"),
        ("06_target_on_other_wall", make_plane(99, 0, DOTS_MM), dict(seed=206), "conflict"),
        ("08_zoomed_in_dot_outside", make_plane(1, 5, DOTS_MM), dict(seed=208, zoom=1.35), "confirmed"),
        # 远、斜、模糊：圆点小且落在纹理“洞”里（曾因只找外轮廓而漏检）
        ("09_far_oblique_blurry", make_plane(1, 5, DOTS_MM), dict(seed=209, tilt=0.15, rot_deg=30, zoom=0.6, blur=1.6), "confirmed"),
        ("07_smooth_surface_ref", None, None, None),
    ]

    rm = fa.detect_markers(ref_img)
    ref_dots = fa.select_reference_dots(rm, 3)
    print(f"参考：二维码 {rm.qr_text!r}，圆点 {len(ref_dots)} 个，码边长 {rm.qr_side():.0f}px\n")
    hdr = f"{'用例':<26}{'期望':<14}{'结果':<14}{'对齐误差(均/最大px)':<22}{'特征内点':<10}{'变化':<8}"
    print(hdr)
    print("-" * len(hdr) * 1)
    all_ok = True
    truth = {}   # 给安卓端测试用：真实的 新→参考 变换（工作坐标）和期望状态
    for name, plane, cam, expect in cases:
        if plane is None:
            continue
        blur = cam.pop("blur", 0.8) if cam else 0.8
        Hc = camera_h(cam.pop("seed"), **cam) if cam else None
        new_img = render(plane, Hc, zlib.crc32(name.encode()) % 1000, blur=blur)
        cv2.imwrite(os.path.join(OUT, f"{name}.jpg"), new_img)
        res, ref_w, new_w = fa.align(ref_img, rm, ref_dots, new_img)
        err = "—"
        if res.H is not None:
            Ht = true_new_to_ref(H_ref, Hc, rm.work_scale, res_scale(new_img))
            grid = fa._roi_grid(rm.qr_corners, ref_dots, rm.work_size)
            g_new = fa._ph(np.linalg.inv(Ht), grid)
            e = np.linalg.norm(fa.map_new_to_ref(res, g_new) - grid, axis=1)
            err = f"{e.mean():.2f} / {e.max():.2f}"
        chg = "—"
        if res.H is not None:
            aligned, valid = fa.warp_to_ref(new_w, res, rm.work_size)
            c, mask, _ = fa.compare(ref_w, aligned, valid, fa.comparison_mask(rm.work_size, res, rm.qr_corners, ref_dots), res_ratio=fa.resolution_ratio(res))
            chg = f"{c.changed_ratio:.2%} 相似{c.similarity:.2f}"
            out = os.path.join(OUT, "results", name)
            os.makedirs(out, exist_ok=True)
            for k, v in fa.make_views(ref_w, aligned, valid, mask, c.regions, c.light_regions).items():
                cv2.imwrite(os.path.join(out, f"{k}.jpg"), v)
        ok = res.status == expect
        all_ok &= ok
        Ht_w = true_new_to_ref(H_ref, Hc, rm.work_scale, res_scale(new_img))
        truth[name] = {"ref": "00_reference.jpg", "new": f"{name}.jpg", "H_true": Ht_w.tolist(),
                       "expect": expect, "python_status": res.status}
        print(f"{name:<26}{expect:<14}{res.status:<14}{err:<22}{res.feature_inliers:<10}{chg:<8}{'' if ok else '  ✗'}")

    # 光滑表面：参考和新照片都在低纹理墙面上
    sm_ref_plane = make_plane(5, 0, DOTS_MM, strength=0.0)
    sm_ref = render(sm_ref_plane, H_ref, 300)
    Hs = camera_h(301)
    sm_new = render(make_plane(5, 0, DOTS_MM, strength=0.0), Hs, 301)
    cv2.imwrite(os.path.join(OUT, "07_smooth_ref.jpg"), sm_ref)
    cv2.imwrite(os.path.join(OUT, "07_smooth_new.jpg"), sm_new)
    srm = fa.detect_markers(sm_ref)
    sdots = fa.select_reference_dots(srm, 3)
    res, _, _ = fa.align(sm_ref, srm, sdots, sm_new)
    Ht = true_new_to_ref(H_ref, Hs, srm.work_scale, res_scale(sm_new))
    grid = fa._roi_grid(srm.qr_corners, sdots, srm.work_size)
    e = np.linalg.norm(fa.map_new_to_ref(res, fa._ph(np.linalg.inv(Ht), grid)) - grid, axis=1)
    ok = res.status == "markers_only"
    all_ok &= ok
    print(f"{'07_smooth_surface':<26}{'markers_only':<14}{res.status:<14}{f'{e.mean():.2f} / {e.max():.2f}':<22}"
          f"{res.feature_inliers:<10}{'—':<8}{'' if ok else '  ✗'}")
    # 码读不出（远 + 重模糊）：按特征摆正放大后应能读出，并完成靶标+特征对齐
    hard = render(make_plane(1, 5, DOTS_MM), camera_h(410, tilt=0.12, rot_deg=20, zoom=0.55), 410, blur=3.0)
    direct = fa.detect_qr(fa.gray_of(fa.to_work(hard)[0]))[0]
    rec = fa.recover_qr(ref_img, rm, hard)
    st_ = fa.align(ref_img, rm, ref_dots, hard, rec)[0].status if rec is not None else "failed"
    ok = direct is None and rec is not None and st_ == "confirmed"
    all_ok &= ok
    print(f"{'10_qr_unreadable':<26}{'confirmed':<14}{st_:<14}{'直接读码失败→特征定位后读出' if rec else '未能恢复':<22}"
          f"{'':<10}{'':<8}{'' if ok else '  ✗'}")
    truth["07_smooth_surface"] = {"ref": "07_smooth_ref.jpg", "new": "07_smooth_new.jpg", "H_true": Ht.tolist(),
                                  "expect": "markers_only", "python_status": res.status}
    import json
    with open(os.path.join(OUT, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    print("\n全部符合预期" if all_ok else "\n有用例不符合预期")
    print(f"测试照片和结果图：{os.path.abspath(OUT)}")
    return 0 if all_ok else 1


def res_scale(img):
    return fa.to_work(img)[1]


if __name__ == "__main__":
    sys.exit(main())

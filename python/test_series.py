"""
无二维码的系列识别测试：

  * 系列 B、C：两面纹理不同的墙，无靶标 → 新照片应按特征唯一归入
  * 系列 D：参考照片和 B 拍的是同一面墙、只错开一点（模拟“背景相同”）
            → 拍 B/D 重叠区的新照片应判为“有歧义”，要求贴码或手选
  * 一面从没登记过的墙 → 应判为“未匹配”
  * 带二维码的 A 系列不参与（有码时直接按码归类）

  python test_series.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldalign as fa  # noqa: E402
import synth_test as st  # noqa: E402

OUT = os.path.join(st.OUT, "series")


def wall(seed):
    p = st.texture(seed)
    st.draw_crack(p, st.crack_points(seed), 6)
    return p


def shifted_cam(seed, dx_frac):
    """同一面墙，取景整体平移 dx_frac 倍画面宽度。"""
    H = st.camera_h(seed, tilt=0.04, rot_deg=4)
    T = np.array([[1, 0, -dx_frac * st.PHOTO[0]], [0, 1, 0], [0, 0, 1.0]])
    return T @ H


def main():
    os.makedirs(OUT, exist_ok=True)
    B, C, X = wall(50), wall(60), wall(70)
    refs_imgs = {
        "B": st.render(B, st.camera_h(500, tilt=0.04, rot_deg=4), 500),
        "C": st.render(C, st.camera_h(600, tilt=0.04, rot_deg=4), 600),
        "D": st.render(B, shifted_cam(500, 0.12), 700),     # 和 B 几乎同一处
    }
    for k, v in refs_imgs.items():
        cv2.imwrite(os.path.join(OUT, f"ref_{k}.jpg"), v)
    refs = [(k, fa.signature(v)) for k, v in refs_imgs.items()]

    cases = [
        ("C 系列复拍", st.render(C, st.camera_h(601, tilt=0.12, rot_deg=15), 601), ["B", "C", "D"], "unique", "C"),
        ("B 系列复拍（与D重叠）", st.render(B, st.camera_h(501, tilt=0.10), 501), ["B", "C", "D"], "ambiguous", None),
        ("B 系列复拍（只登记了B、C）", st.render(B, st.camera_h(501, tilt=0.10), 501), ["B", "C"], "unique", "B"),
        ("未登记的墙", st.render(X, st.camera_h(701), 701), ["B", "C", "D"], "no_match", None),
    ]
    ok_all = True
    print(f"{'用例':<26}{'期望':<12}{'结果':<12}{'说明'}")
    print("-" * 90)
    for name, img, use, exp, exp_key in cases:
        sig = fa.signature(img)
        r = fa.identify(sig, [(k, s) for k, s in refs if k in use])
        ok = r.decision == exp and (exp_key is None or r.best_key == exp_key)
        ok_all &= ok
        rank = " ".join(f"{k}:{n}" for k, n, _ in r.ranking)
        print(f"{name:<26}{exp:<12}{r.decision:<12}{r.message}  [{rank}]{'' if ok else '  ✗'}")

    # 纯特征对齐精度：C 系列复拍
    Hr = st.camera_h(600, tilt=0.04, rot_deg=4)
    Hn = st.camera_h(601, tilt=0.12, rot_deg=15)
    new = st.render(C, Hn, 601)
    res, ref_w, new_w = fa.align_features_only(refs_imgs["C"], new)
    Ht = st.true_new_to_ref(Hr, Hn, 0.5, 0.5)
    xs, ys = np.meshgrid(np.linspace(300, 1700, 8), np.linspace(250, 1250, 6))
    grid = np.c_[xs.ravel(), ys.ravel()]
    e = np.linalg.norm(fa.map_new_to_ref(res, fa._ph(np.linalg.inv(Ht), grid)) - grid, axis=1)
    print(f"\n纯特征对齐（C 系列）：{res.message}；误差 均值 {e.mean():.2f}px / 最大 {e.max():.2f}px")
    ok_all &= res.status == "features_only" and e.max() < 3
    print("\n全部符合预期" if ok_all else "\n有用例不符合预期")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

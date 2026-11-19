"""
真实纹理上的“前后对比”测试（DeepCrack 真实裂缝照片 + 人工标注）：
  上次 = 把真实裂缝沿走向的后 40% 修补掉（图像修复），裂缝较短；
  这次 = 原始照片，再加随机透视、亮度/对比度变化、轻微模糊和噪声（模拟换了位置、换了光照重拍）；
  流程 = 纯图像特征对齐 → 裂缝对比 → 新增长度。真值 = 被修补掉那段标注骨架的长度。
另外做“无变化”对照：上次、这次都是完整裂缝，新增应接近 0。

  python test_real_pairs.py [--n 60]
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldalign as fa  # noqa: E402
import measure as ms  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "real_data", "deepcrack")


def retake(img, rng):
    """模拟重拍：透视 + 亮度/对比度 + 模糊 + 噪声。返回 (新照片, H 原→新)。"""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + rng.uniform(-0.05, 0.05, (4, 2)) * [w, h]).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # 镜像填充出来的边缘不是真实内容：记下真实内容的范围，比较时排除
    real = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    f = out.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gain = rng.uniform(0.8, 1.2) * (1 + 0.2 * ((xx / w - 0.5) * rng.uniform(-1, 1) + (yy / h - 0.5) * rng.uniform(-1, 1)))
    f = (f - 128) * rng.uniform(0.85, 1.15) + 128
    f = f * gain[..., None]
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 0.8)) + rng.normal(0, 2.0, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8), real


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(ROOT, "test_img", "*.jpg")))
    if not files:
        print("没有找到数据集（real_data/deepcrack/），跳过"); return 0
    out = os.path.join(ROOT, "pairs"); os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []; skipped = 0
    for f in files:
        if len(rows) >= a.n:
            break
        name = os.path.splitext(os.path.basename(f))[0]
        img = cv2.imread(f); gt = cv2.imread(os.path.join(ROOT, "test_lab", name + ".png"), cv2.IMREAD_GRAYSCALE)
        gt_b = (gt > 127).astype(np.uint8)
        sk = ms.prune(ms.thin(gt_b), 8)
        ys, xs = np.nonzero(sk)
        w_gt = gt_b.sum() / max(ms.link_length(sk), 1)
        # 只用适合的样本：单条为主、宽度 ≤ 12px（更宽的裂缝修补后痕迹太假）
        if len(xs) < 150 or w_gt > 12:
            skipped += 1; continue
        # 沿主方向排序，修补掉后 40%
        pts = np.stack([xs, ys], 1).astype(np.float64)
        d = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))[1][:, 1]
        t = (pts - pts.mean(0)) @ d
        cut = np.percentile(t, 60)
        yy, xx = np.mgrid[0:gt.shape[0], 0:gt.shape[1]]
        side = ((xx - pts[:, 0].mean()) * d[0] + (yy - pts[:, 1].mean()) * d[1]) > cut
        hole = cv2.dilate((gt_b & side).astype(np.uint8), np.ones((9, 9), np.uint8))
        before = cv2.inpaint(img, hole * 255, 5, cv2.INPAINT_TELEA)
        grow_true = ms.link_length(sk & side.astype(np.uint8))
        if grow_true < 40:
            skipped += 1; continue
        for kind, ref in (("grow", before), ("same", img)):
            new, real = retake(img, rng)
            res, rw, nw = fa.align_features_only(ref, new)
            row = dict(name=name, kind=kind, status=res.status, grow_true=grow_true if kind == "grow" else 0.0)
            if res.H is not None:
                al, valid = fa.warp_to_ref(nw, res, (rw.shape[1], rw.shape[0]))
                real_al, _ = fa.warp_to_ref(cv2.cvtColor(real, cv2.COLOR_GRAY2BGR), res, (rw.shape[1], rw.shape[0]))
                mask = ((valid > 0) & (cv2.erode(real_al[..., 0], np.ones((9, 9), np.uint8)) > 250)).astype(np.uint8)
                ch = ms.compare_cracks(fa.gray_of(rw), fa.gray_of(al), mask)
                level, _ = ms.conclude(ch, None)
                row.update(grow=ch.grow_px, level=level, L_ref=ch.ref.total_px, L_new=ch.new.total_px)
                if kind == "same" and level not in ("稳定", "无裂缝"):
                    cv2.imwrite(os.path.join(out, f"FALSE_{name}.jpg"), np.hstack([ref, ms.draw_cracks(al, ch, None)]))
                if len(rows) < 16:
                    cv2.imwrite(os.path.join(out, f"{name}_{kind}.jpg"), np.hstack([ref, ms.draw_cracks(al, ch, None)]))
            rows.append(row)
    g = [r for r in rows if r["kind"] == "grow"]; s = [r for r in rows if r["kind"] == "same"]
    ok_align = [r for r in rows if r.get("grow") is not None]
    print(f"样本 {len(g)} 对“延长” + {len(s)} 对“无变化”（跳过不适合的 {skipped} 张）；对齐成功 {len(ok_align)}/{len(rows)}")
    gg = [r for r in g if "grow" in r]
    ratio = np.array([r["grow"] / r["grow_true"] for r in gg])
    hit = sum(1 for r in gg if r["level"] in ("延长", "延长且变宽", "新增"))
    print(f"延长：判定为“延长/新增” {hit}/{len(gg)}（{hit / max(len(gg), 1):.0%}）；新增长度 测/真 中位数 {np.median(ratio):.2f}"
          f"（25%~75%：{np.percentile(ratio, 25):.2f}~{np.percentile(ratio, 75):.2f}）")
    ss = [r for r in s if "grow" in r]
    fa_n = sum(1 for r in ss if r["level"] in ("延长", "延长且变宽", "新增", "变宽"))
    gs = np.array([r["grow"] for r in ss])
    print(f"无变化：误判为有发展 {fa_n}/{len(ss)}（{fa_n / max(len(ss), 1):.0%}）；误报的新增长度中位数 {np.median(gs):.0f} px，90 分位 {np.percentile(gs, 90):.0f} px")
    from collections import Counter
    print("  无变化各判定：", dict(Counter(r["level"] for r in ss)), "；误判样本的新增长度：",
          sorted(int(r["grow"]) for r in ss if r["level"] not in ("稳定", "无裂缝")))
    # 阈值敏感性：把“延长”的最小新增长度从 10px 提高，会怎样
    for t in (10, 15, 20, 30):
        h2 = sum(1 for r in gg if r["grow"] >= max(t, 0.05 * r["L_ref"]))
        f2 = sum(1 for r in ss if r["grow"] >= max(t, 0.05 * r["L_ref"]))
        print(f"  最小新增 {t:>2}px：延长检出 {h2}/{len(gg)}，无变化误报 {f2}/{len(ss)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

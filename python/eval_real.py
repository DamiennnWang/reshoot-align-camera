"""
用真实裂缝照片评估裂缝检测（DeepCrack 数据集，人工标注掩膜为真值）。

  python eval_real.py [--n 237] [--split test] [--save 12]

指标（按骨架、容差 5 像素）：
  精确率 P = 检出骨架中落在标注裂缝附近的比例（低 = 误检多）
  召回率 R = 标注骨架中被检出区域覆盖的比例（低 = 漏检多）
  长度比 = 检出总长 / 标注骨架总长

数据集仅限非商业研究使用，放在 real_data/deepcrack/，不随 App 分发。
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure as ms  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "real_data", "deepcrack")
TOL = 5


def evaluate(img, gt, detect=None):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.ones(g.shape, np.uint8)
    cm = (detect or ms.detect_cracks)(g, mask)
    gt_b = (gt > 127).astype(np.uint8)
    gt_sk = ms.prune(ms.thin(gt_b), 8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOL + 1, 2 * TOL + 1))
    near_gt = cv2.dilate(gt_b, k) > 0
    near_det = cv2.dilate(cm.mask, k) > 0
    det_n = int(cm.skel.sum()); gt_n = int(gt_sk.sum())
    tp_p = int(((cm.skel > 0) & near_gt).sum())
    tp_r = int(((gt_sk > 0) & near_det).sum())
    L_det = ms.link_length(cm.skel); L_gt = ms.link_length(gt_sk)
    # 标注宽度：面积 / 骨架长度
    w_gt = float(gt_b.sum()) / max(L_gt, 1.0)
    return dict(det_n=det_n, gt_n=gt_n, tp_p=tp_p, tp_r=tp_r, L_det=L_det, L_gt=L_gt, w_gt=w_gt,
                w_det=(np.mean([c.mean_w_px for c in cm.cracks]) if cm.cracks else 0.0)), cm, gt_sk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=237)
    ap.add_argument("--split", default="test")
    ap.add_argument("--save", type=int, default=12, help="保存前几张的标注图到 real_data/deepcrack/out/")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(ROOT, f"{a.split}_img", "*.jpg")))[:a.n]
    if not files:
        print("没有找到数据集，请先下载到 real_data/deepcrack/"); return 1
    out = os.path.join(ROOT, "out"); os.makedirs(out, exist_ok=True)
    rows = []
    for i, f in enumerate(files):
        name = os.path.splitext(os.path.basename(f))[0]
        img = cv2.imread(f); gt = cv2.imread(os.path.join(ROOT, f"{a.split}_lab", name + ".png"), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        r, cm, gt_sk = evaluate(img, gt)
        r["name"] = name; rows.append(r)
        if i < a.save:
            vis = img.copy()
            vis[cv2.dilate(gt_sk, np.ones((3, 3), np.uint8)) > 0] = (0, 200, 0)         # 标注：绿
            vis[cv2.dilate(cm.skel, np.ones((3, 3), np.uint8)) > 0] = (0, 0, 255)       # 检出：红
            cv2.imwrite(os.path.join(out, name + ".jpg"), vis)
    P = sum(r["tp_p"] for r in rows) / max(sum(r["det_n"] for r in rows), 1)
    R = sum(r["tp_r"] for r in rows) / max(sum(r["gt_n"] for r in rows), 1)
    pi = np.array([r["tp_p"] / r["det_n"] if r["det_n"] else np.nan for r in rows])
    ri = np.array([r["tp_r"] / r["gt_n"] if r["gt_n"] else np.nan for r in rows])
    lr = np.array([r["L_det"] / r["L_gt"] for r in rows if r["L_gt"] > 20])
    wg = np.array([r["w_gt"] for r in rows])
    print(f"{len(rows)} 张 | 精确率 {P:.1%} | 召回率 {R:.1%} | F1 {2 * P * R / max(P + R, 1e-9):.1%}")
    print(f"逐张中位数：精确率 {np.nanmedian(pi):.1%}，召回率 {np.nanmedian(ri):.1%}，长度比 {np.median(lr):.2f}"
          f"（25%~75%：{np.percentile(lr, 25):.2f}~{np.percentile(lr, 75):.2f}）")
    print(f"完全漏检（有标注但没检出）：{int(sum(1 for r in rows if r['gt_n'] > 20 and r['det_n'] == 0))} 张")
    for lo, hi in ((0, 3), (3, 6), (6, 12), (12, 99)):
        sel = [r for r in rows if lo <= r["w_gt"] < hi]
        if sel:
            p = sum(r["tp_p"] for r in sel) / max(sum(r["det_n"] for r in sel), 1)
            rr = sum(r["tp_r"] for r in sel) / max(sum(r["gt_n"] for r in sel), 1)
            print(f"  标注宽 {lo:>2}~{hi:<2}px：{len(sel):>3} 张  精确率 {p:.1%}  召回率 {rr:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

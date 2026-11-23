"""
把样例照片里的二维码换成另一个编号（网站展示用）。

测试样例生成时都用同一个码 B01-Z023；放到网站上给人用手机翻拍时，不同样例会被 App 当成同一个测点、
互相对比。这里按“测点”给每组样例换一个独立编号：找到原二维码的四个角，把新码按透视贴进同一位置，
黑白电平和模糊程度按原码区域估计，看起来和原图一致。算法测试仍用 samples/ 里的原图。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import fieldalign as fa  # noqa: E402
import make_target as mt  # noqa: E402


def restamp(img: np.ndarray, code: str) -> np.ndarray | None:
    m = fa.detect_markers(img)
    if m.qr_corners is None:
        return None
    quad = (np.asarray(m.qr_corners, np.float32).reshape(4, 2) / m.work_scale).astype(np.float32)
    q = np.array(mt.qr_image(code).convert("L"))
    n = q.shape[0]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [n, 0], [n, n], [0, n]]), quad)
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(q, H, (w, h), flags=cv2.INTER_AREA, borderValue=255).astype(np.float32) / 255
    inside = cv2.warpPerspective(np.full((n, n), 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
    region = inside > 0
    if region.sum() < 100:
        return None
    side = float(np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]))
    out = img.astype(np.float32)
    for c in range(3):
        v = out[:, :, c][region]
        lo, hi = np.percentile(v, 4), np.percentile(v, 96)
        layer = lo + (hi - lo) * warped
        out[:, :, c][region] = layer[region]
    # 原码边缘的模糊程度（约 1/120 码边长），让新码和周围一致
    blur = max(0.6, side / 160)
    soft = cv2.GaussianBlur(out, (0, 0), blur)
    grow = cv2.dilate(inside, np.ones((5, 5), np.uint8)) > 0
    out[grow] = soft[grow]
    noise = np.random.default_rng(len(code)).normal(0, 2.0, out.shape).astype(np.float32)
    out[grow] += noise[grow]
    return np.clip(out, 0, 255).astype(np.uint8)


def restamp_file(src: str, dst: str, code: str) -> bool:
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    r = restamp(img, code) if img is not None else None
    if r is None:
        return False
    cv2.imwrite(dst, r, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


if __name__ == "__main__":
    a, b, c = sys.argv[1:4]
    print(restamp_file(a, b, c))

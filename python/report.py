"""
电脑端后处理：读取手机导出的 ZIP（单个测点或「导出全部」），在电脑上重新分析裂缝和面状病害，生成 HTML 报告。

  ../.venv/bin/python report.py 导出包.zip [更多.zip] [-o 输出目录] [--grow-mm 10]

输出目录（默认 <第一个 ZIP 名>_报告/）：
  index.html     总表（按严重程度排序）+ 每个测点的参考/本次/标注图、手机结论与电脑复核结论对照
  img/           标注图（电脑复核结果）

和手机用的是同一套算法（measure.py / defects.py 与 App 逐步对应），区别是：电脑上可以调阈值（--grow-mm 等），
可以一次处理很多测点，结论写进一份可打印的报告。位移、螺栓标记线沿用手机结果。
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import sys
import zipfile
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import defects as dfx  # noqa: E402
import fieldalign as fa  # noqa: E402
import measure as ms  # noqa: E402

RANK = {"建议重点关注": 5, "新增": 4, "延长且变宽": 4, "明显": 3, "延长": 3, "变宽": 3, "轻微": 2, "稳定": 1, "无裂缝": 1, "无变化": 1}
COLOR = {"无变化": "#22a06b", "稳定": "#22a06b", "无裂缝": "#22a06b", "轻微": "#2f80ed", "明显": "#e08a00", "延长": "#e08a00", "变宽": "#e08a00"}
LINE = {"crack"}
AREA = {"spall", "seep", "effl", "rust"}


def imread(z: zipfile.ZipFile, name: str):
    try:
        return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
    except KeyError:
        return None


def zones_in(z: zipfile.ZipFile):
    for n in z.namelist():
        if n == "zone.json" or n.endswith("/zone.json"):
            yield n[:-len("zone.json")]


def analyze(ref_full, aligned, kinds: set, grow_mm: float | None, cmp_mask=None, change=None):
    """ref_full：参考原图；aligned：已对齐到参考工作坐标的本次照片；cmp_mask：可选的比较区域（对齐时得到的一致区）。
    返回 (等级, 结论, 标注图, 明细)。"""
    h, w = aligned.shape[:2]
    ref = cv2.resize(ref_full, (w, h), interpolation=cv2.INTER_AREA)
    valid = cv2.erode((fa.gray_of(aligned) > 3).astype(np.uint8) * 255, np.ones((15, 15), np.uint8))
    rm = fa.detect_markers(ref)
    mask = valid.copy()
    scale = None
    if rm.qr_corners is not None:
        mask &= fa.marker_mask((w, h), rm.qr_corners, fa.select_reference_dots(rm))
        scale = ms.scale_from_qr(rm.qr_corners, ms.qr_size_mm(rm.qr_text))
    mmpp_at = scale.mm_per_px if scale else None
    mmpp = mmpp_at((w / 2, h / 2)) if scale else None
    if cmp_mask is not None:
        mask &= cmp_mask
    general = not kinds
    change = np.zeros((h, w), np.uint8) if change is None else change
    areas = dfx.areas(ref, aligned, mask, change, mmpp_at) if (general or kinds & AREA) else []
    real = [a for a in areas if a.cls != dfx.SHADOW]
    parts, items, sev = [], [], 0
    out = aligned.copy()
    level = None
    if general or kinds & LINE:
        rules = ms.Rules(grow_min_px=(grow_mm / mmpp) if (grow_mm and mmpp) else ms.Rules().grow_min_px)
        ch = ms.compare_cracks(fa.gray_of(ref), fa.gray_of(aligned), mask, ignore=dfx.area_ignore(real, (h, w)))
        level, txt = ms.conclude(ch, mmpp, rules=rules)
        parts.append(("裂缝：" if general or len(kinds) > 1 else "") + txt)
        sev = max(sev, {"无裂缝": 0, "稳定": 0, "延长": 2, "变宽": 2}.get(level, 3))
        out = ms.draw_cracks(out, ch, mmpp)
        for i, c in enumerate(ch.new.cracks):
            items.append(f"C{i + 1} 长 {ms.fmt_len(c.length_px, mmpp)} 宽 {ms.fmt_w(c.max_w_px, mmpp)}" + ("（新增）" if c.is_new else ""))
    for i, a in enumerate(real):
        mm2 = a.mm2 or 0
        heavy = a.cls in ("剥落/掉块", "锈蚀")
        sev = max(sev, 3 if mm2 >= (3000 if heavy else 10000) else (2 if mm2 >= 1000 else 1))
        items.append(f"A{i + 1} {a.cls} 约 {mm2 / 100:.1f} cm²")
        col = {"锈蚀": (0, 120, 255), "剥落/掉块": (40, 40, 255), "渗水/湿痕": (255, 140, 40), "泛碱/白化": (255, 120, 190)}.get(a.cls, (0, 200, 255))
        cv2.polylines(out, [a.contour.astype(np.int32)], True, col, max(4, w // 400), cv2.LINE_AA)
        cv2.putText(out, f"A{i + 1}", (int(a.box[0]), max(30, int(a.box[1]) - 10)), cv2.FONT_HERSHEY_SIMPLEX, max(1.2, w / 900), col, 3, cv2.LINE_AA)
    if real:
        by = {}
        for a in real:
            by.setdefault(a.cls, []).append(a.mm2 or 0)
        # 有面状病害、裂缝没有发展：面状结论放第一行
        parts.insert(0 if level in ("无裂缝", "稳定") else len(parts), "面状：" + "；".join(f"{k} {len(v)} 处，共约 {sum(v) / 100:.1f} cm²" for k, v in by.items()))
    if len(real) < len(areas):
        parts.append(f"另有 {len(areas) - len(real)} 处柔和的明暗变化，判断为阴影/光照，未计入")
    if not (kinds == {"crack"} and level):
        level = ["无变化", "轻微", "明显", "建议重点关注"][sev]
    return level, "\n".join(parts) or "未发现变化", out, items


KIND_CODES = LINE | AREA | {"joint", "offset", "bolt", "part"}


def pair(ref_img, new_img, kinds: set = frozenset(), grow_mm: float | None = None):
    """两张照片直接对比（网页“在线处理”用）：对齐 → 裂缝 + 面状病害。
    返回 dict(ok, how, level, text, items, warnings) 和图像 dict(aligned, marked, changes)。"""
    rm = fa.detect_markers(ref_img)
    nm = fa.detect_markers(new_img)
    dots = fa.select_reference_dots(rm) if rm.qr_corners is not None else []
    if rm.qr_corners is not None and nm.qr_corners is not None:
        res, ref_w, new_w = fa.align(ref_img, rm, dots, new_img, nm)
        how = "二维码靶标 + 图像特征"
    else:
        res, ref_w, new_w = fa.align_features_only(ref_img, new_img)
        how = "图像特征（没有找到两张都有的二维码）"
    info = dict(how=how, status=getattr(res, "status", ""), message=getattr(res, "message", ""))
    if res.H is None:
        return dict(ok=False, level="无法对齐", text="两张照片对不上：请确认是同一位置，重叠足够多。", items=[], warnings=[], **info), {}
    size = (ref_w.shape[1], ref_w.shape[0])
    aligned, valid = fa.warp_to_ref(new_w, res, size)
    excl = fa.comparison_mask(size, res, rm.qr_corners if rm.qr_corners is not None else None, dots)
    cmp_res, change, _ = fa.compare(ref_w, aligned, valid, excl, res_ratio=fa.resolution_ratio(res))
    cm = ((excl > 0) & (valid > 0) & (change != 64)).astype(np.uint8) * 255
    level, text, marked, items = analyze(ref_img, aligned, set(kinds) & (LINE | AREA), grow_mm, cm, change)
    views = fa.make_views(ref_w, aligned, valid, change, cmp_res.regions, cmp_res.light_regions)
    imgs = dict(ref=ref_w, aligned=aligned, marked=marked)
    imgs.update({k: v for k, v in views.items() if k in ("changes", "checker")})
    warn = [str(w) for w in (fa.quality_warnings(cmp_res, new_w) or [])]
    if set(kinds) - LINE - AREA:
        warn.append("位移、螺栓、部件需要在手机上分析（电脑端暂不支持），这里只做了裂缝和面状病害。")
    return dict(ok=True, level=level, text=text, items=items, warnings=warn, **info), imgs


def build_report(zips, out: str, grow_mm: float | None = None, log=print) -> str:
    """把一个或多个导出 ZIP 生成 HTML 报告，返回 index.html 路径。"""
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    rows = []
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for d in zones_in(z):
                meta = json.loads(z.read(d + "zone.json").decode("utf-8"))
                kinds = set() if meta.get("kinds") in (None, "general") else set(meta["kinds"].split(","))
                photos = list(csv.DictReader(io.StringIO(z.read(d + "report.csv").decode("utf-8-sig"))))
                ref_row = next((p for p in photos if p["是否参考"] == "是"), None)
                if ref_row is None:
                    continue
                ref_full = imread(z, d + ref_row["原图"])
                tag = f"{len(rows):03d}"
                cv2.imwrite(os.path.join(out, "img", f"{tag}_ref.jpg"), cv2.resize(ref_full, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
                last = [p for p in photos if p["是否参考"] != "是" and p["对齐图"]]
                entry = dict(name=meta.get("name"), kinds=meta.get("kinds_label", "通用"), qr=meta.get("qr"), tag=tag,
                             ref_time=ref_row["拍摄时间"], src=os.path.basename(zp), lat=meta.get("lat"), lon=meta.get("lon"))
                if last:
                    p = last[-1]
                    al = imread(z, d + p["对齐图"])
                    lv, txt, img, items = analyze(ref_full, al, kinds, grow_mm)
                    if kinds - LINE - AREA:          # 位移、螺栓、部件：电脑端暂不重算，沿用手机结果（取两者中较重的等级）
                        img = imread(z, d + p["标注图"]) if p["标注图"] else img
                        if RANK.get(p["判定等级"], 0) > RANK.get(lv, 0) or not (kinds & (LINE | AREA)):
                            lv = p["判定等级"]
                        txt = ("" if not (kinds & (LINE | AREA)) else txt + "\n") + "位移/螺栓/部件：沿用手机结果——" + p["文字结论"]
                    cv2.imwrite(os.path.join(out, "img", f"{tag}_pc.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    entry.update(time=p["拍摄时间"], app_level=p["判定等级"], app_text=p["文字结论"], level=lv, text=txt, items=items)
                    log(f"{meta.get('name')}：手机【{p['判定等级']}】 电脑【{lv}】")
                else:
                    log(f"{meta.get('name')}：只有参考照片")
                rows.append(entry)
    rows.sort(key=lambda r: -RANK.get(r.get("level"), 0))
    e = html.escape
    def badge(lv):
        return f'<span class="lv" style="background:{COLOR.get(lv, "#d64545" if lv else "#6b7785")}">{e(lv or "待复拍")}</span>'
    body = [f"<h1>复拍对比报告</h1><p class='m'>{datetime.now():%Y-%m-%d %H:%M} · {len(rows)} 个测点 · 来源 {e(', '.join(map(os.path.basename, zips)))}</p>",
            "<table><tr><th>测点</th><th>类型</th><th>本次</th><th>电脑复核</th><th>手机结论</th></tr>"]
    for r in rows:
        body.append(f"<tr><td><a href='#z{r['tag']}'>{e(r['name'])}</a></td><td>{e(r['kinds'])}</td><td>{e(r.get('time', '—'))}</td>"
                    f"<td>{badge(r.get('level'))}</td><td>{badge(r.get('app_level')) if r.get('app_level') else '—'}</td></tr>")
    body.append("</table>")
    for r in rows:
        body.append(f"<section id='z{r['tag']}'><h2>{e(r['name'])} {badge(r.get('level'))}</h2>"
                    f"<p class='m'>编号 {e(r['qr'] or '无码')} · 类型 {e(r['kinds'])} · 参考 {e(r['ref_time'])} → 本次 {e(r.get('time', '—'))}"
                    + (f" · 位置 {r['lat']:.6f}, {r['lon']:.6f}" if r.get('lat') is not None else "") + "</p>")
        if r.get("level"):
            body.append(f"<div class='g'><figure><img src='img/{r['tag']}_ref.jpg'><figcaption>参考</figcaption></figure>"
                        f"<figure><img src='img/{r['tag']}_pc.jpg'><figcaption>本次（电脑复核标注）</figcaption></figure></div>"
                        f"<div class='c'><b>电脑复核：</b>{e(r['text'])}</div>"
                        + (f"<ul>{''.join(f'<li>{e(i)}</li>' for i in r['items'])}</ul>" if r["items"] else "")
                        + f"<div class='c m'><b>手机结论（{e(r['app_level'])}）：</b>{e(r['app_text'])}</div>")
        else:
            body.append(f"<div class='g'><figure><img src='img/{r['tag']}_ref.jpg'><figcaption>参考（待复拍）</figcaption></figure></div>")
        body.append("</section>")
    css = """body{font:14px/1.6 -apple-system,'PingFang SC','Microsoft YaHei',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#16202a}
    table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #e3e7ec;padding:6px 8px;text-align:left}th{color:#6b7785;font-weight:500}
    .lv{color:#fff;border-radius:99px;padding:1px 8px;font-size:12px}.m{color:#6b7785;font-size:12px}section{page-break-inside:avoid;border-top:1px solid #e3e7ec;margin-top:18px}
    .g{display:flex;gap:10px}figure{margin:0;flex:1}img{width:100%;border-radius:8px}figcaption{color:#6b7785;font-size:12px}
    .c{white-space:pre-wrap;background:#f2f4f7;border-radius:8px;padding:8px 10px;margin-top:8px}a{color:#2f80ed}"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                f"<title>复拍对比报告</title><style>{css}</style><body>{''.join(body)}</body></html>")
    return os.path.join(out, "index.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zips", nargs="+")
    ap.add_argument("-o", "--out")
    ap.add_argument("--grow-mm", type=float, default=None, help="裂缝新增至少多少毫米才算延长/新增（默认约 40 像素）")
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.zips[0])[0] + "_报告"
    print(f"\n报告：{build_report(a.zips, out, a.grow_mm)}")


if __name__ == "__main__":
    main()

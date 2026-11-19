"""
电脑端“测区归档”工具 —— 与手机 App 同样的流程：扫码 → 查库 → 对齐 → 对比 → 存档。
用来在开发手机端之前，先拿真实照片验证算法。

  # 直接比较两张照片（第一张作参考）
  python photo_archive.py compare 参考.jpg 新照片.jpg -o out/

  # 归档模式：照片逐张加入，同一二维码自动归到一个测区，第一张自动成为参考；
  # 没有二维码的照片按图像特征识别属于哪个系列（唯一且足够吻合才自动归入）
  python photo_archive.py add 照片1.jpg 照片2.jpg ...
  python photo_archive.py add --new-series 天桥东侧 无码照片.jpg   # 用无码照片新建系列
  python photo_archive.py list
  python photo_archive.py set-ref <照片编号>

归档目录默认 ../archive（SQLite 数据库 archive.db + 每张照片的结果图）。
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sqlite3
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldalign as fa
import measure as ms  # noqa: E402

DEFAULT_ARCHIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "archive")

SCHEMA = """
CREATE TABLE IF NOT EXISTS zone(
  id INTEGER PRIMARY KEY, qr_text TEXT UNIQUE, name TEXT,
  created_at TEXT, ref_photo_id INTEGER);
CREATE TABLE IF NOT EXISTS photo(
  id INTEGER PRIMARY KEY, zone_id INTEGER, path TEXT, taken_at TEXT,
  markers_json TEXT, align_json TEXT, status TEXT, message TEXT,
  changed_ratio REAL, result_dir TEXT);
"""


def run_pair(ref_img, ref_m, ref_dots, new_img, out_dir, new_m=None):
    """对齐+对比一对照片，写出结果图，返回 (AlignResult, CompareResult|None)。
    两张都有同一个二维码 → 靶标+特征；否则 → 纯特征。"""
    os.makedirs(out_dir, exist_ok=True)
    nm = new_m or fa.detect_markers(new_img)
    use_markers = ref_m is not None and ref_m.qr_corners is not None and nm.qr_corners is not None
    if use_markers:
        res, ref_w, new_w = fa.align(ref_img, ref_m, ref_dots, new_img, nm)
        cv2.imwrite(os.path.join(out_dir, "markers_ref.jpg"), fa.draw_markers(ref_w, ref_m.qr_corners, ref_dots))
        cv2.imwrite(os.path.join(out_dir, "markers_new.jpg"),
                    fa.draw_markers(new_w, nm.qr_corners,
                                    [(*d, ref_dots[i][2]) if d else None for i, d in enumerate(res.dots_matched)]
                                    if res.dots_matched else [], color=(0, 160, 255)))
    else:
        res, ref_w, new_w = fa.align_features_only(ref_img, new_img)
    cmp_res = None
    if res.H is not None:
        ref_size = (ref_w.shape[1], ref_w.shape[0])
        aligned, valid = fa.warp_to_ref(new_w, res, ref_size)
        excl = fa.comparison_mask(ref_size, res, ref_m.qr_corners if use_markers else None, ref_dots)
        cmp_res, mask, _ = fa.compare(ref_w, aligned, valid, excl, res_ratio=fa.resolution_ratio(res))
        views = fa.make_views(ref_w, aligned, valid, mask, cmp_res.regions, cmp_res.light_regions)
        cv2.imwrite(os.path.join(out_dir, "aligned.jpg"), aligned)
        for k, v in views.items():
            cv2.imwrite(os.path.join(out_dir, f"{k}.jpg"), v)
        # 裂缝参数 + 文字结论（比例尺来自参考照片里的二维码）
        cmask = ((excl > 0) & (valid > 0) & (mask != 64)).astype(np.uint8)
        ch = ms.compare_cracks(fa.gray_of(ref_w), fa.gray_of(aligned), cmask)
        sc = ms.scale_from_qr(ref_m.qr_corners, ms.qr_size_mm(ref_m.qr_text))
        mmpp = None
        if sc is not None:
            ys, xs = np.nonzero(ch.new.skel)
            at = (xs.mean(), ys.mean()) if xs.size else np.asarray(ref_m.qr_corners).reshape(4, 2).mean(0)
            mmpp = sc.mm_per_px(at)
        level, text = ms.conclude(ch, mmpp, len(cmp_res.light_regions), cmp_res.light_ratio)
        cv2.imwrite(os.path.join(out_dir, "cracks.jpg"), ms.draw_cracks(aligned, ch, mmpp))
        cmp_res.crack = {"level": level, "conclusion": text, "mm_per_px": mmpp,
                         "total_px": ch.new.total_px, "max_w_px": ch.new.max_w_px, "grow_px": ch.grow_px,
                         "ref_total_px": ch.ref.total_px, "ref_max_w_px": ch.ref.max_w_px,
                         "cracks": [dict(c.__dict__) for c in ch.new.cracks]}
    report = {"align": res.to_json(),
              "compare": None if cmp_res is None else cmp_res.__dict__}
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
    return res, cmp_res


def print_result(res, cmp_res, out_dir):
    print(f"  状态：{res.message}")
    if res.H is not None:
        cons = "—" if res.consistency_px is None else f"{res.consistency_px:.1f}px"
        rms = "—" if res.marker_rms is None else f"{res.marker_rms:.2f}px"
        print(f"  靶标残差 RMS {rms} | 特征内点 {res.feature_inliers} "
              f"覆盖 {res.feature_coverage:.0%} | 靶标/特征偏差 {cons}")
    if cmp_res is not None:
        print(f"  结构变化 {len(cmp_res.regions)} 处，占对比区 {cmp_res.changed_ratio:.2%}；"
              f"明暗变化（阴影/水渍等）{len(cmp_res.light_regions)} 处 {cmp_res.light_ratio:.2%}；"
              f"整体相似度 {cmp_res.similarity:.2f}")
        for w in fa.quality_warnings(cmp_res):
            print(f"  ⚠ {w}")
        cr = getattr(cmp_res, "crack", None)
        if cr:
            print(f"  裂缝结论【{cr['level']}】{cr['conclusion']}")
    print(f"  结果图：{os.path.abspath(out_dir)}")


# ---------------------------------------------------------------- compare 子命令
def cmd_compare(a):
    ref_img, new_img = fa.load_image(a.ref), fa.load_image(a.new)
    rm = fa.detect_markers(ref_img)
    ref_dots = fa.select_reference_dots(rm, a.dots) if rm.qr_corners is not None else []
    if rm.qr_corners is None:
        print("参考照片没有二维码 → 纯图像特征对齐")
    else:
        print(f"二维码：{rm.qr_text}；参考照片选中圆点 {len(ref_dots)} 个")
    if rm.qr_corners is not None and len(ref_dots) < 3:
        print("  警告：圆点少于 3 个，只能靠二维码和图像特征对齐")
    res, cmp_res = run_pair(ref_img, rm, ref_dots, new_img, a.out)
    print_result(res, cmp_res, a.out)


# ---------------------------------------------------------------- 归档子命令
def open_db(root):
    os.makedirs(root, exist_ok=True)
    con = sqlite3.connect(os.path.join(root, "archive.db"))
    con.executescript(SCHEMA)
    return con


def _zone_ref(con, root, zid):
    ref_id, = con.execute("SELECT ref_photo_id FROM zone WHERE id=?", (zid,)).fetchone()
    ref_path, ref_mj = con.execute("SELECT path, markers_json FROM photo WHERE id=?", (ref_id,)).fetchone()
    ref_m, ref_dots = fa.markers_from_json(json.loads(ref_mj))
    return ref_id, fa.load_image(os.path.join(root, ref_path)), ref_m, ref_dots


def _new_zone(con, root, path, img, m, name, dots_n):
    now = dt.datetime.now().isoformat(timespec="seconds")
    dots = fa.select_reference_dots(m, dots_n) if m.qr_corners is not None else []
    cur = con.execute("INSERT INTO zone(qr_text,name,created_at) VALUES(?,?,?)", (m.qr_text, name, now))
    zid = cur.lastrowid
    pid = _store_photo(con, root, zid, name, path, now, fa.markers_to_json(m, dots), "reference", "参考照片")
    con.execute("UPDATE zone SET ref_photo_id=? WHERE id=?", (pid, zid))
    con.commit()
    return zid, pid, dots


def cmd_add(a):
    con = open_db(a.archive)
    sig_cache = {}
    for path in a.photos:
        print(f"\n{os.path.basename(path)}")
        img = fa.load_image(path)
        m = fa.detect_markers(img)
        now = dt.datetime.now().isoformat(timespec="seconds")
        if a.new_series:
            zid, pid, _ = _new_zone(con, a.archive, path, img, m, a.new_series, a.dots)
            print(f"  新建系列「{a.new_series}」，设为参考照片（编号 {pid}）")
            a.new_series = None
            continue
        if m.qr_corners is not None:
            row = con.execute("SELECT id, name FROM zone WHERE qr_text=?", (m.qr_text,)).fetchone()
            if row is None:
                zid, pid, dots = _new_zone(con, a.archive, path, img, m, m.qr_text, a.dots)
                print(f"  新测区 {m.qr_text}，设为参考照片（编号 {pid}），选中圆点 {len(dots)} 个")
                continue
            zid, zname = row
            how = f"二维码 {m.qr_text}"
        else:
            # 无码：与所有系列的参考照片比特征
            refs = []
            for zid_, zname_ in con.execute("SELECT id, name FROM zone ORDER BY id").fetchall():
                if zid_ not in sig_cache:
                    sig_cache[zid_] = fa.signature(_zone_ref(con, a.archive, zid_)[1])
                refs.append(((zid_, zname_), sig_cache[zid_]))
            sig = fa.signature(img)
            idr = fa.identify(sig, refs)
            top = "，".join(f"{k[1]}:{n}" for k, n, _ in idr.ranking[:3])
            if idr.decision != "unique":
                print(f"  未自动归类：{idr.message}（候选 {top or '无'}）")
                print("  可用 --new-series 名称 把它建成新系列")
                continue
            zid, zname = idr.best_key
            how = idr.message
        ref_id, ref_img, ref_m, ref_dots = _zone_ref(con, a.archive, zid)
        if m.qr_corners is None and ref_m.qr_corners is not None:
            # 参考有码、新照片没读出：按特征摆正放大后再解码，读出就走“靶标+特征”流程
            rec = fa.recover_qr(ref_img, ref_m, img)
            if rec is not None:
                m = rec
                how += "；二维码由特征定位后读出"
        pid = _store_photo(con, a.archive, zid, zname, path, now, fa.markers_to_json(m), "pending", "")
        out_dir = os.path.join(a.archive, "zones", zname, f"{pid}_vs_{ref_id}")
        res, cmp_res = run_pair(ref_img, ref_m, ref_dots, img, out_dir, m)
        con.execute("UPDATE photo SET align_json=?, status=?, message=?, changed_ratio=?, result_dir=? WHERE id=?",
                    (json.dumps(res.to_json(), ensure_ascii=False), res.status, res.message,
                     None if cmp_res is None else cmp_res.changed_ratio,
                     os.path.relpath(out_dir, a.archive), pid))
        con.commit()
        print(f"  归入「{zname}」（{how}；照片编号 {pid}，对比参考 {ref_id}）")
        print_result(res, cmp_res, out_dir)


def _store_photo(con, root, zid, zone_name, src, now, markers, status, msg):
    d = os.path.join(root, "zones", zone_name)
    os.makedirs(d, exist_ok=True)
    cur = con.execute("INSERT INTO photo(zone_id,taken_at,markers_json,status,message) VALUES(?,?,?,?,?)",
                      (zid, now, json.dumps(markers, ensure_ascii=False), status, msg))
    pid = cur.lastrowid
    dst = os.path.join(d, f"{pid}{os.path.splitext(src)[1].lower()}")
    shutil.copy2(src, dst)
    con.execute("UPDATE photo SET path=? WHERE id=?", (os.path.relpath(dst, root), pid))
    return pid


def cmd_list(a):
    con = open_db(a.archive)
    for zid, q, name, ref in con.execute("SELECT id, qr_text, name, ref_photo_id FROM zone ORDER BY id"):
        print(f"系列「{name}」{'二维码 ' + q if q else '（无码）'}（参考照片 {ref}）")
        for pid, t, st, msg, r in con.execute(
                "SELECT id, taken_at, status, message, changed_ratio FROM photo WHERE zone_id=? ORDER BY id", (zid,)):
            rr = "" if r is None else f"  变化 {r:.2%}"
            print(f"   #{pid} {t} [{st}] {msg}{rr}")


def cmd_set_ref(a):
    con = open_db(a.archive)
    row = con.execute("SELECT zone_id, markers_json, path FROM photo WHERE id=?", (a.photo_id,)).fetchone()
    if not row:
        sys.exit("没有这张照片")
    zid, mj, path = row
    img = fa.load_image(os.path.join(a.archive, path))
    m = fa.detect_markers(img)
    dots = fa.select_reference_dots(m, a.dots) if m.qr_corners is not None else []
    con.execute("UPDATE photo SET markers_json=?, status='reference' WHERE id=?",
                (json.dumps(fa.markers_to_json(m, dots), ensure_ascii=False), a.photo_id))
    con.execute("UPDATE zone SET ref_photo_id=? WHERE id=?", (a.photo_id, zid))
    con.commit()
    print(f"已把照片 {a.photo_id} 设为参考，圆点 {len(dots)} 个")


def main():
    ap = argparse.ArgumentParser(description="二维码+圆点+图像特征 复拍照片对齐对比")
    ap.add_argument("--archive", default=DEFAULT_ARCHIVE)
    ap.add_argument("--dots", type=int, default=3, help="参考照片中使用的圆点数（3~6）")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("compare"); p.add_argument("ref"); p.add_argument("new"); p.add_argument("-o", "--out", default="out")
    p.set_defaults(fn=cmd_compare)
    p = sp.add_parser("add"); p.add_argument("photos", nargs="+")
    p.add_argument("--new-series", help="把第一张照片建成一个新系列（无码照片用）")
    p.set_defaults(fn=cmd_add)
    p = sp.add_parser("list"); p.set_defaults(fn=cmd_list)
    p = sp.add_parser("set-ref"); p.add_argument("photo_id", type=int); p.set_defaults(fn=cmd_set_ref)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

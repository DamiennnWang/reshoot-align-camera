"""
把网站需要的东西汇总到一个目录（默认 deploy/dist/），再打成 bundle.tar.gz 上传到服务器：

  dist/site/      静态页面 + 测试图片 + 查看器 + 下载文件
  dist/server/    server.py + 算法工具包（report / fieldalign / measure / defects）
  dist/requirements.txt

  python3 deploy/build_site.py [-o 输出目录]

只收录程序合成的样例（samples/）；real_data/（DeepCrack，仅限非商业研究）绝不打包。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S = os.path.join(ROOT, "samples")
VERSION = "0.4.0"
TOOLKIT = ["fieldalign.py", "measure.py", "defects.py", "report.py", "photo_archive.py", "make_target.py"]
REQ = "numpy==2.0.2\nopencv-contrib-python-headless==5.1.0.95\nqrcode==8.2\npillow==11.3.0\n"

KIND_CN = {"crack_grow": "裂缝延长", "crack_same": "裂缝无变化", "crack_widen": "裂缝变宽", "spall": "剥落/掉块", "rust": "锈蚀",
           "seep": "渗水/湿痕", "effl": "泛碱/白化", "none": "无变化", "offset": "错台", "joint": "接缝张开",
           "bolt": "螺栓松动", "bolt_small": "螺栓松动（小）", "part": "部件"}
TITLE = {
    "curve_grow": "弯曲裂缝延长", "curve_nochange": "弯曲裂缝无变化", "curve_widen": "弯曲裂缝变宽", "map_nochange": "网状裂缝无变化",
    "hairline_grow": "0.3 mm 浅色细裂缝延长（极限）", "curve_stain": "裂缝 + 水渍", "spall_light": "浅色新鲜断面剥落",
    "spall_small": "小块剥落（约 2 cm²）", "spall_rebar": "剥落露筋", "spall_shadow": "剥落 + 大片柔和阴影", "spall_on_crack": "裂缝上的剥落",
    "spall": "剥落", "rust": "锈蚀", "seep": "渗水", "effl": "泛碱", "plain_nochange": "平整墙面无变化", "offset": "错台 3 mm",
    "joint": "接缝张开 2 mm", "joint_1mm": "接缝张开 1 mm", "segments_nochange": "管片拼缝无变化", "bolts": "螺栓松动",
    "bolts_10deg": "螺栓转 10°", "bolts_nochange": "螺栓无变化",
}


def J(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def expect_text(t: dict) -> str:
    k = t.get("kind")
    s = ["真值：" + KIND_CN.get(k, k or "")]
    for key, fmt in (("grow_mm", "新增 {:.0f} mm"), ("total_mm", "总长 {:.0f} mm"), ("area_mm2", "面积 {:.0f} mm²"),
                     ("disp_mm", "位移 {:.1f} mm"), ("width_mm", "宽 {} mm")):
        if key in t:
            s.append(fmt.format(t[key]))
    if "w_ref_mm" in t:
        s.append(f"宽 {t['w_ref_mm']} → {t['w_new_mm']} mm")
    if "area_cls" in t:
        s.append(f"另有{t['area_cls']}（不算病害发展）")
    return "，".join(s)


def pairs_group(gid, title, desc, folder):
    t = J(os.path.join(S, folder, "truth.json"))
    cases = []
    for name, v in t.items():
        ref, new = f"samples/{folder}/{name}_ref.jpg", f"samples/{folder}/{name}_new.jpg"
        if not os.path.exists(os.path.join(ROOT, ref)):
            continue
        extra = []
        if os.path.exists(os.path.join(S, folder, f"{name}_cracks.jpg")):
            extra.append(["裂缝标注", f"samples/{folder}/{name}_cracks.jpg"])
        cases.append(dict(title=TITLE.get(name, name), ref=ref, new=new, expect=expect_text(v), extra=extra))
    return dict(id=gid, title=title, desc=desc, cases=cases)


def manifest():
    g = [pairs_group("hard", "弯曲裂缝与剥落（难样例）", "v0.3.0 专门做的难例：弯曲/分叉裂缝、网状裂缝、细裂缝、浅色剥落、露筋、阴影干扰。", "hard"),
         pairs_group("defects", "病害类型", "建档时可选的病害类型：剥落、锈蚀、渗水、泛碱、错台、接缝、螺栓标记线等（位移、螺栓类需在手机上分析）。", "defects")]
    # 宽度测量
    t = J(os.path.join(S, "measure", "truth.json"))
    cs = []
    for name, v in t.items():
        if not os.path.exists(os.path.join(S, "measure", f"{name}_ref.jpg")):
            continue
        e = f"真值：宽 {v['width_mm']} mm，长 {v['len_ref_mm']:.0f} → {v['len_new_mm']:.0f} mm" if "width_mm" in v else "真值：无裂缝"
        ex = [["裂缝标注", f"samples/measure/{name}_cracks.jpg"]] if os.path.exists(os.path.join(S, "measure", f"{name}_cracks.jpg")) else []
        cs.append(dict(title=name.replace("_rough", "（粗糙表面）").replace("no_crack", "无裂缝").replace("w", "宽 ", 1) if name.startswith("w") else
                       name.replace("no_crack_rough", "无裂缝（粗糙表面）").replace("no_crack", "无裂缝"),
                       ref=f"samples/measure/{name}_ref.jpg", new=f"samples/measure/{name}_new.jpg", expect=e, extra=ex))
    g.append(dict(id="measure", title="裂缝宽度与长度测量", desc="靶标二维码提供比例尺（约 0.23 mm/像素），测宽度和延长量。", cases=cs))
    # 对齐鲁棒性
    t = J(os.path.join(S, "truth.json"))
    exp = {"confirmed": "应对齐成功", "dot_moved": "应发现靶标圆点被移动", "conflict": "应提示靶标被贴到了别的墙上", "markers_only": "光滑墙面，只能靠靶标对齐"}
    names = {"01_normal": "正常复拍", "02_crack_grew": "裂缝变长", "03_oblique_lighting": "斜光照", "04_upside_down": "倒着拍",
             "05_dot_moved": "圆点被移动", "06_target_on_other_wall": "靶标贴到别处", "07_smooth_surface": "光滑墙面",
             "08_zoomed_in_dot_outside": "拍近了（圆点出画）", "09_far_oblique_blurry": "又远又斜又糊"}
    cs = []
    for k, v in t.items():
        ex = []
        rd = os.path.join(S, "results", k)
        for f, l in (("changes.jpg", "变化图"), ("checker.jpg", "棋盘对照"), ("cracks.jpg", "裂缝")):
            if os.path.exists(os.path.join(rd, f)):
                ex.append([l, f"samples/results/{k}/{f}"])
        cs.append(dict(title=names.get(k, k), ref=f"samples/{v['ref']}", new=f"samples/{v['new']}", expect=exp.get(v.get("expect"), ""), extra=ex))
    g.append(dict(id="align", title="对齐：角度、距离、靶标异常", desc="同一面墙不同角度、距离、光照下复拍，检验对齐和靶标异常提示。", cases=cs))
    # 光照
    t = J(os.path.join(S, "lighting", "truth.json"))
    cs = [dict(title={"soft_shadow.jpg": "柔和阴影", "hard_shadow.jpg": "硬阴影", "hard_shadow_crack.jpg": "硬阴影 + 新裂缝",
                      "crack.jpg": "新裂缝", "nochange.jpg": "无变化"}.get(k, k), ref="samples/lighting/ref.jpg", new=f"samples/lighting/{k}",
               expect=("真值：" + ("有结构变化" if v["expect_structural"] else "无结构变化") + ("，有光照变化" if v["expect_light"] else "")))
          for k, v in t.items()]
    g.append(dict(id="lighting", title="光照与阴影", desc="阴影、光照变化不应被当成病害；阴影里的新裂缝要能找到。", cases=cs))
    # 镜头畸变
    t = J(os.path.join(S, "distortion", "truth.json"))
    cs = [dict(title=k.replace("new_", "畸变 ").replace("_crack.jpg", " · 有新裂缝").replace("_nochange.jpg", " · 无变化"),
               ref=f"samples/distortion/{v['ref']}", new=f"samples/distortion/{k}",
               expect="真值：" + ("有新裂缝" if "crack" in k else "无变化")) for k, v in t.items()]
    g.append(dict(id="distortion", title="镜头畸变", desc="广角 / 不同手机镜头的桶形畸变，自动估计并校正。", cases=cs))
    # 同类测点识别
    cs = [dict(title=f"测点 {x} 建档", new=f"samples/series/ref_{x}.jpg", expect="批量建档：没有二维码，靠位置和图像") for x in "BCD"]
    cs += [dict(title=f"{x} 复拍", ref=f"samples/series/ref_{x}.jpg", new=f"samples/series/{x}_new.jpg", expect="应自动归入测点 " + x)
           for x in "BC" if os.path.exists(os.path.join(S, "series", f"{x}_new.jpg"))]
    if os.path.exists(os.path.join(S, "series", "X_new.jpg")):
        cs.append(dict(title="陌生位置", new="samples/series/X_new.jpg", expect="不属于任何测点 → 应进「待确认」"))
    g.append(dict(id="series", title="无码测点自动配对", desc="没有二维码时按图像（加位置、朝向）识别是哪个测点；B 和 D 背景相似。", cases=cs))
    # 拍摄质量
    t = J(os.path.join(S, "robustness", "truth.json"))
    cn = {"color_cast": "偏色", "glare": "反光", "glare_crack": "反光 + 裂缝", "lowres_1600": "低分辨率", "lowres_crack": "低分辨率 + 裂缝",
          "motion_blur": "手抖模糊", "motion_blur_crack": "模糊 + 裂缝", "noise_crack": "噪点 + 裂缝", "noise_jpeg60": "噪点 + 高压缩",
          "over_exposed": "过曝", "under_exposed": "欠曝", "rotated_90": "旋转 90°"}
    cs = []
    for k, v in t.items():
        b = k[:-4]
        ex = [["变化图", f"samples/robustness/{b}_changes.jpg"]] if os.path.exists(os.path.join(S, "robustness", f"{b}_changes.jpg")) else []
        cs.append(dict(title=cn.get(b, b), new=f"samples/robustness/{k}", expect="真值：" + ("有结构变化" if v["expect_structural"] else "无结构变化"), extra=ex))
    g.append(dict(id="robust", title="拍摄质量", desc="模糊、过曝、欠曝、反光、偏色、噪点、低分辨率（参考图由测试程序实时生成，这里只展示复拍图和算法输出）。", cases=cs))
    return dict(version=VERSION, built=datetime.now().strftime("%Y-%m-%d %H:%M"), groups=g)


def stamp_codes(man, site):
    """每个“测点”换一个独立二维码编号（原样例都用 B01-Z023，手机翻拍时会被当成同一个测点）。"""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "deploy"))
    import restamp_qr
    letters = {"hard": "H", "defects": "D", "measure": "M"}
    done, n_ok, n_bad = {}, 0, []
    for g in man["groups"]:
        refs = {}
        for i, c in enumerate(g["cases"]):
            gid = g["id"]
            if gid in letters:
                code = f"DEMO-{letters[gid]}{i + 1:02d}"
            elif gid == "align":
                code = "DEMO-A02" if "smooth" in (c.get("ref") or "") else "DEMO-A01"
            elif gid == "lighting":
                code = "DEMO-L01"
            elif gid == "distortion":
                code = f"DEMO-K{refs.setdefault(c['ref'], len(refs) + 1):02d}"
            elif gid == "robust":
                code = "DEMO-R01"
            else:
                continue                  # 无码测点：本来就没有二维码
            stamped = False
            for k in ("ref", "new"):
                rel = c.get(k)
                if not rel:
                    continue
                if rel in done:
                    stamped = stamped or done[rel] == code
                    continue
                pth = os.path.join(site, rel)
                ok = restamp_qr.restamp_file(pth, pth, code)
                done[rel] = code if ok else None
                stamped = stamped or ok
                if ok:
                    n_ok += 1
                else:
                    n_bad.append(rel)
            if stamped:
                c["code"] = code
    print(f"二维码换号：{n_ok} 张；未找到二维码 {len(n_bad)} 张 {n_bad[:6]}")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "deploy", "dist"))
    a = ap.parse_args()
    out = a.out
    # 换一台电脑时先检查输入齐不齐，缺什么直接说怎么补
    apk_src = next((p for p in (os.path.join(ROOT, "复拍对齐相机.apk"),
                                os.path.join(ROOT, "android", "app", "build", "outputs", "apk", "debug", "app-debug.apk"))
                    if os.path.exists(p)), None)
    missing = []
    if not os.path.isdir(os.path.join(S, "results")):
        missing.append("samples/results/ —— 先运行：cd python && ../.venv/bin/python synth_test.py")
    if not os.path.exists(os.path.join(S, "export_demo", "全部测点_demo.zip")):
        missing.append("samples/export_demo/全部测点_demo.zip —— 应随仓库提供，检查是否完整拉取")
    if not apk_src:
        missing.append("APK —— 先运行：cd android && ./gradlew assembleDebug")
    if missing:
        sys.exit("缺少以下输入：\n  " + "\n  ".join(missing))
    shutil.rmtree(out, ignore_errors=True)
    site = os.path.join(out, "site")
    shutil.copytree(os.path.join(ROOT, "deploy", "site"), site)
    os.makedirs(os.path.join(site, "viewer"), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, "web", "index.html"), os.path.join(site, "viewer", "index.html"))
    # 测试图片：只收 samples/ 下的合成数据
    for d in ("hard", "defects", "measure", "lighting", "distortion", "series", "robustness", "results", "export_demo"):
        shutil.copytree(os.path.join(S, d), os.path.join(site, "samples", d), ignore=shutil.ignore_patterns("*.py", ".*"))
    for f in os.listdir(S):
        if f.endswith((".jpg", ".png")) or f == "truth.json":
            shutil.copy2(os.path.join(S, f), os.path.join(site, "samples", f))
    man = manifest()
    stamp_codes(man, site)
    with open(os.path.join(site, "samples", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False)
    # 下载
    dl = os.path.join(site, "download")
    apk = f"fieldalign-camera-v{VERSION}.apk"
    shutil.copy2(apk_src, os.path.join(dl, apk))
    kit = f"fieldalign-toolkit-v{VERSION}.zip"
    with zipfile.ZipFile(os.path.join(dl, kit), "w", zipfile.ZIP_DEFLATED) as z:
        for f in TOOLKIT:
            z.write(os.path.join(ROOT, "python", f), f"fieldalign-toolkit/{f}")
        z.write(os.path.join(ROOT, "web", "index.html"), "fieldalign-toolkit/查看器.html")
        z.writestr("fieldalign-toolkit/requirements.txt", REQ)
        z.write(os.path.join(ROOT, "targets", "targets_B01-Z023.pdf"), "fieldalign-toolkit/靶标_B01-Z023.pdf")
        z.writestr("fieldalign-toolkit/使用说明.txt",
                   "复拍对齐相机 · 电脑端工具包 v" + VERSION + "\n\n"
                   "python3 -m venv .venv && .venv/bin/pip install -r requirements.txt\n\n"
                   "report.py 导出包.zip [--grow-mm 10]      手机导出的 ZIP → HTML 报告\n"
                   "photo_archive.py compare 参考.jpg 本次.jpg -o out   两张照片对比\n"
                   "make_target.py                           生成靶标（二维码 + 圆点）\n"
                   "查看器.html                              离线查看导出 ZIP（双击用浏览器打开）\n")
    tgt = "target-B01-Z023.pdf"
    shutil.copy2(os.path.join(ROOT, "targets", "targets_B01-Z023.pdf"), os.path.join(dl, tgt))
    files = [
        dict(id="app", title=f"手机 App（安卓）v{VERSION}", file=apk, desc="安卓 8.0 以上。下载后在手机上安装（需要允许“安装未知来源应用”）。",
             points=["批量建档 / 批量复拍，二维码、位置 + 朝向、图像自动配对，认不出的进「待确认」",
                     "裂缝延长 / 变宽、剥落、渗水、泛碱、锈蚀、位移、螺栓松动", "首页「导出全部」→ 在这个网站上查看或重新分析"]),
        dict(id="toolkit", title="电脑端工具包（Python）", file=kit, desc="和手机同一套算法：批量生成报告、调阈值、两张照片对比、离线查看器、生成靶标。",
             points=["report.py：导出 ZIP → HTML 报告", "photo_archive.py：照片对比 / 归档", "查看器.html：离线查看导出包"]),
        dict(id="target", title="靶标 PDF（B01-Z023）", file=tgt, desc="二维码 + 三个圆点，按 100% 打印，贴在测点旁边。二维码内容可写尺寸和病害类型，如 B01-Z023;40mm;裂缝。"),
    ]
    for f in files:
        p = os.path.join(dl, f["file"])
        f["size"], f["sha256"] = os.path.getsize(p), sha(p)
    with open(os.path.join(dl, "files.json"), "w", encoding="utf-8") as fp:
        json.dump(files, fp, ensure_ascii=False, indent=1)
    # 服务端
    srv = os.path.join(out, "server")
    os.makedirs(srv)
    shutil.copy2(os.path.join(ROOT, "deploy", "server.py"), srv)
    for f in ("fieldalign.py", "measure.py", "defects.py", "report.py", "make_target.py"):
        shutil.copy2(os.path.join(ROOT, "python", f), srv)
    with open(os.path.join(out, "requirements.txt"), "w") as f:
        f.write(REQ)
    shutil.copy2(os.path.join(ROOT, "deploy", "start.sh"), out)
    # 检查：不许混进真实数据集
    for r, _, fs in os.walk(out):
        assert "real_data" not in r and "DeepCrack" not in r, r
    tgz = os.path.join(ROOT, "deploy", "bundle.tar.gz")
    with tarfile.open(tgz, "w:gz") as t:
        t.add(out, arcname="app")
    n = sum(len(fs) for _, _, fs in os.walk(out))
    print(f"{out}：{n} 个文件；打包 {tgz} {os.path.getsize(tgz) / 2 ** 20:.1f} MB")


if __name__ == "__main__":
    main()

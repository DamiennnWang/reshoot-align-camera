"""
复拍对齐相机 · 网站服务（只用 Python 标准库 + 工具包本身的 numpy / OpenCV）

  python server.py            # 默认 0.0.0.0:8080

页面（SITE_DIR，由 build_site.py 生成）：首页、测试图片、导出包查看器、在线处理、下载。
接口：
  POST /api/pair       两张照片对比（multipart：ref、new 文件，或 ref_sample、new_sample 样例路径；kinds、grow_mm）
  POST /api/report     手机「导出全部」的 ZIP（请求体就是 ZIP 文件；?grow_mm=）→ 生成 HTML 报告
  GET  /api/jobs/<id>  任务状态
  GET  /jobs/<id>/...  任务结果（保留 JOB_TTL_H 小时后自动删除）

环境变量：PORT、SITE_DIR、DATA_DIR、TOOLKIT_DIR、WORKERS（同时处理几个任务，默认 1）、
         QUEUE_MAX（排队上限，默认 6）、JOB_TTL_H（默认 24）、MAX_ZIP_MB（默认 600）、
         PROCESS_KEY（设置后，处理接口需要带 ?key= 或请求头 X-Key）。
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(120_000_000))   # 防止超大图片把内存吃满

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.abspath(os.environ.get("SITE_DIR", os.path.join(HERE, "site")))
DATA = os.path.abspath(os.environ.get("DATA_DIR", os.path.join(HERE, "data")))
TOOLKIT = os.path.abspath(os.environ.get("TOOLKIT_DIR", os.path.join(HERE, "..", "python")))
JOBS = os.path.join(DATA, "jobs")
WORKERS = int(os.environ.get("WORKERS", "1"))
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "6"))
JOB_TTL = float(os.environ.get("JOB_TTL_H", "24")) * 3600
MAX_ZIP = int(os.environ.get("MAX_ZIP_MB", "600")) * 2 ** 20
MAX_PHOTOS = 80 * 2 ** 20
KEY = os.environ.get("PROCESS_KEY", "")
KINDS = {"crack", "spall", "seep", "effl", "rust"}

sys.path.insert(0, TOOLKIT)
os.makedirs(JOBS, exist_ok=True)
pool = ThreadPoolExecutor(max_workers=WORKERS)
jobs: dict[str, dict] = {}
lock = threading.Lock()


def _toolkit():
    import cv2  # noqa: F401  延迟导入：首页、样例图不依赖 OpenCV
    import fieldalign as fa
    import report as rp
    return cv2, fa, rp


def _save(job):
    with open(os.path.join(JOBS, job["id"], "job.json"), "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)


def _busy():
    with lock:
        return sum(j["state"] in ("queued", "running") for j in jobs.values())


def _new_job(kind, **extra):
    jid = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(JOBS, jid))
    job = dict(id=jid, kind=kind, state="queued", created=time.time(), log=[], **extra)
    with lock:
        jobs[jid] = job
    _save(job)
    return job


def _run(job, fn):
    job["state"] = "running"
    job["started"] = time.time()
    _save(job)
    try:
        fn(job)
        job["state"] = "done"
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job["state"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
    job["finished"] = time.time()
    _save(job)


def _shrink(cv2, img, side=1800):
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    return img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def do_pair(job, ref_bytes, new_bytes, kinds, grow_mm):
    import numpy as np
    cv2, fa, rp = _toolkit()
    d = os.path.join(JOBS, job["id"])
    imgs = []
    for b in (ref_bytes, new_bytes):
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("照片无法读取（请上传 JPG / PNG）")
        imgs.append(im)
    job["log"].append("对齐、分析中…")
    info, out = rp.pair(imgs[0], imgs[1], kinds, grow_mm)
    names = {"marked": "标注", "changes": "变化", "checker": "棋盘对照", "aligned": "本次（已对齐）", "ref": "参考"}
    info["images"] = []
    for k, label in names.items():
        if k in out:
            cv2.imwrite(os.path.join(d, f"{k}.jpg"), _shrink(cv2, out[k]), [cv2.IMWRITE_JPEG_QUALITY, 86])
            info["images"].append(dict(key=k, label=label, url=f"/jobs/{job['id']}/{k}.jpg"))
    job["result"] = info


def do_report(job, zpath, grow_mm):
    _, _, rp = _toolkit()
    out = os.path.join(JOBS, job["id"], "report")
    rp.build_report([zpath], out, grow_mm, log=lambda s: job["log"].append(s))
    job["result"] = dict(report=f"/jobs/{job['id']}/report/index.html", zip=f"/jobs/{job['id']}/upload.zip",
                         zones=len(job["log"]))


def check_zip(path):
    if not zipfile.is_zipfile(path):
        raise ValueError("不是 ZIP 文件")
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if len(infos) > 50000 or sum(i.file_size for i in infos) > 4 * 2 ** 30:
            raise ValueError("ZIP 太大或文件太多")
        if not any(i.filename.endswith("zone.json") for i in infos):
            raise ValueError("ZIP 里没有 zone.json：请上传手机「导出」或「导出全部」生成的 ZIP")


def cleanup_loop():
    while True:
        now = time.time()
        for jid in os.listdir(JOBS):
            p = os.path.join(JOBS, jid)
            try:
                if now - os.path.getmtime(p) > JOB_TTL:
                    shutil.rmtree(p, ignore_errors=True)
                    with lock:
                        jobs.pop(jid, None)
            except OSError:
                pass
        time.sleep(1800)


def safe_join(root, rel):
    p = os.path.abspath(os.path.join(root, rel.lstrip("/")))
    return p if p == root or p.startswith(root + os.sep) else None


class Handler(BaseHTTPRequestHandler):
    server_version = "fieldalign"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} [{self.log_date_time_string()}] {fmt % args}\n")

    # ---------- 工具
    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _err(self, msg, code=400):
        self._json({"error": msg}, code)

    def _authed(self, q):
        return not KEY or q.get("key", [""])[0] == KEY or self.headers.get("X-Key") == KEY

    def _file(self, path, head=False):
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/json", "application/javascript"):
            ctype += "; charset=utf-8"
        size = os.path.getsize(path)
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
        if m and size:
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        rel = os.path.relpath(path, SITE)
        cache = "public, max-age=86400" if rel.startswith(("samples", "download")) else "no-cache"
        self.send_header("Cache-Control", cache)
        if path.endswith((".apk", ".zip")) and "/jobs/" not in self.path and "download" in rel:
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{os.path.basename(path)}")
        self.end_headers()
        if head:
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _targets(self, q):
        """批量编号靶标：?prefix=T&start=1&count=24&digits=4&fmt=pdf|csv"""
        try:
            prefix = q.get("prefix", ["T"])[0].strip()
            start, count = int(q.get("start", ["1"])[0]), int(q.get("count", ["12"])[0])
            digits = int(q.get("digits", ["4"])[0])
        except ValueError:
            return self._err("起始号、数量必须是整数")
        if not re.fullmatch(r"[A-Za-z0-9_\-]{0,12}", prefix):
            return self._err("前缀只能用字母、数字、- 和 _，最多 12 个字符")
        if not (1 <= count <= 300 and 0 <= start <= 10 ** 8 and 1 <= digits <= 8):
            return self._err("数量 1–300，位数 1–8")
        sys.path.insert(0, TOOLKIT)
        import make_target as mt
        codes = mt.codes_for(prefix, start, count, digits)
        name = f"targets_{codes[0]}-{codes[-1]}"
        if q.get("fmt", ["pdf"])[0] == "csv":
            b = mt.mapping_csv(codes).encode("utf-8")
            ctype, fn = "text/csv; charset=utf-8", name + "_mapping.csv"
        else:
            import io
            pages = mt.make_sheet(codes)
            buf = io.BytesIO()
            pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:], resolution=mt.DPI)
            b, ctype, fn = buf.getvalue(), "application/pdf", name + ".pdf"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Content-Disposition", f"attachment; filename={fn}")
        self.end_headers()
        self.wfile.write(b)

    # ---------- GET
    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        u = urlparse(self.path)
        path = unquote(u.path)
        if path == "/api/targets":
            return self._targets(parse_qs(u.query))
        if path == "/api/health":
            return self._json({"ok": True, "busy": _busy(), "workers": WORKERS})
        m = re.match(r"^/api/jobs/([0-9a-f]{12})$", path)
        if m:
            with lock:
                job = jobs.get(m.group(1))
            if job is None:
                p = os.path.join(JOBS, m.group(1), "job.json")
                if not os.path.exists(p):
                    return self._err("任务不存在或已过期", 404)
                job = json.load(open(p, encoding="utf-8"))
            ahead = 0
            if job["state"] == "queued":
                with lock:
                    ahead = sum(j["state"] in ("queued", "running") and j["created"] < job["created"] for j in jobs.values())
            return self._json({**job, "ahead": ahead})
        if path.startswith("/jobs/"):
            p = safe_join(JOBS, path[len("/jobs/"):])
            if p and os.path.isfile(p) and not p.endswith("job.json"):
                return self._file(p, head)
            return self._err("文件不存在或已过期", 404)
        p = safe_join(SITE, path)
        if p and os.path.isdir(p):
            if not path.endswith("/"):
                self.send_response(301)
                self.send_header("Location", u.path + "/" + (("?" + u.query) if u.query else ""))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            p = os.path.join(p, "index.html")
        if p and os.path.isfile(p):
            return self._file(p, head)
        self._err("页面不存在", 404)

    # ---------- POST
    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path not in ("/api/pair", "/api/report"):
            return self._err("接口不存在", 404)
        if not self._authed(q):
            return self._err("需要访问密码", 403)
        if _busy() >= QUEUE_MAX:
            return self._err("现在排队的任务太多，请稍后再试", 503)
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._err("缺少 Content-Length", 411)
        try:
            gm = q.get("grow_mm", [""])[0]
            grow = float(gm) if gm else None
        except ValueError:
            return self._err("grow_mm 不是数字")
        if u.path == "/api/report":
            if n <= 0 or n > MAX_ZIP:
                return self._err(f"ZIP 为空或超过 {MAX_ZIP >> 20} MB", 413)
            job = _new_job("report", name=q.get("name", ["导出包.zip"])[0][:120], grow_mm=grow)
            zp = os.path.join(JOBS, job["id"], "upload.zip")
            left = n
            with open(zp, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            try:
                if left:
                    raise ValueError("上传中断")
                check_zip(zp)
            except Exception as e:  # noqa: BLE001
                shutil.rmtree(os.path.join(JOBS, job["id"]), ignore_errors=True)
                with lock:
                    jobs.pop(job["id"], None)
                return self._err(str(e))
            pool.submit(_run, job, lambda j: do_report(j, zp, grow))
            return self._json({"id": job["id"]})
        # /api/pair
        if n <= 0 or n > MAX_PHOTOS:
            return self._err("照片为空或太大（两张合计不超过 80 MB）", 413)
        body = self.rfile.read(n)
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            return self._err("需要 multipart/form-data")
        msg = BytesParser(policy=policy.HTTP).parsebytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
        fields, files = {}, {}
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if part.get_filename() is not None:
                files[name] = part.get_payload(decode=True)
            else:
                fields[name] = (part.get_payload(decode=True) or b"").decode("utf-8", "replace")
        data = {}
        for k in ("ref", "new"):
            if files.get(k):
                data[k] = files[k]
            elif fields.get(k + "_sample"):
                sp = safe_join(os.path.join(SITE, "samples"), fields[k + "_sample"].removeprefix("samples/"))
                if not sp or not os.path.isfile(sp) or not sp.lower().endswith((".jpg", ".png")):
                    return self._err("样例图片不存在")
                data[k] = open(sp, "rb").read()
            else:
                return self._err("请选择参考照片和本次照片")
        kinds = {k for k in fields.get("kinds", "").split(",") if k in KINDS}
        job = _new_job("pair", kinds=sorted(kinds), grow_mm=grow,
                       sample=[fields.get("ref_sample"), fields.get("new_sample")] if fields.get("ref_sample") else None)
        pool.submit(_run, job, lambda j: do_pair(j, data["ref"], data["new"], kinds, grow))
        self._json({"id": job["id"]})


def main():
    for jid in os.listdir(JOBS):           # 重启后：恢复已完成的任务，丢弃没跑完的
        p = os.path.join(JOBS, jid, "job.json")
        try:
            j = json.load(open(p, encoding="utf-8"))
            if j["state"] in ("queued", "running"):
                j["state"], j["error"] = "error", "服务重启，任务中断，请重新提交"
                _save(j)
            jobs[jid] = j
        except (OSError, ValueError, KeyError):
            shutil.rmtree(os.path.join(JOBS, jid), ignore_errors=True)
    threading.Thread(target=cleanup_loop, daemon=True).start()
    port = int(os.environ.get("PORT", "8080"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    print(f"site={SITE} data={DATA} toolkit={TOOLKIT} → http://0.0.0.0:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

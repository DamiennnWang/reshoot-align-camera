package com.fieldalign.camera.align

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.features2d.SIFT
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.QRCodeDetector
import org.opencv.objdetect.QRCodeDetectorAruco
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * 复拍照片对齐与对比核心算法 —— python/fieldalign.py 的移植，两边算法要保持一致。
 *
 * 坐标约定：所有计算在“工作分辨率”（长边 [WORK_MAX_SIDE]）下进行；
 * 对齐矩阵 H 把新照片工作坐标映射到参考照片工作坐标。
 */
object AlignEngine {
    const val WORK_MAX_SIDE = 2000
    const val DOT_TO_QR_RATIO = 0.4
    const val DOT_MOVE_TOL_FRAC = 0.30
    const val FEATURE_MIN_INLIERS = 25
    const val FEATURE_MIN_COVERAGE = 0.10
    const val CONSISTENCY_TOL_PX = 6.0
    const val TEXTURE_RICH_KP = 800

    const val ID_SIDE = 1000
    const val ID_MIN_INLIERS = 40
    const val ID_MIN_COVERAGE = 0.12
    const val ID_UNIQUE_RATIO = 2.0
    const val FEAT_ALIGN_MIN_INLIERS = 40

    val STATUS_TEXT = mapOf(
        "confirmed" to "已对齐（靶标与图像特征一致）",
        "markers_only" to "对齐待确认（靶标正常，但表面纹理不足以复核）",
        "conflict" to "配准需复核（靶标与表面纹理不一致，靶标可能被移动）",
        "dot_moved" to "已对齐，但有圆点疑似移动或被遮挡",
        "features_only" to "已按图像特征对齐（无靶标）",
        "failed" to "无法对齐",
        "reference" to "参考照片",
    )

    // ------------------------------------------------------------------ 数据结构
    data class Dot(val x: Double, val y: Double, val d: Double, val score: Double = 1.0) {
        val p get() = P(x, y)
    }

    data class Markers(
        val qrText: String?,
        val qrCorners: List<P>?,            // 4 个角，按二维码自身方向
        val dotCandidates: List<Dot>,
        val workW: Int,
        val workH: Int,
    ) {
        fun qrSide(): Double {
            val c = qrCorners ?: return 0.0
            return (0..3).sumOf { c[it].dist(c[(it + 1) % 4]) } / 4
        }
    }

    data class AlignResult(
        val status: String,
        val message: String,
        val h: Homography? = null,
        val dotsMatched: List<P?> = emptyList(),
        val dotsMoved: List<Int> = emptyList(),
        val markerRms: Double? = null,
        val featureInliers: Int = 0,
        val featureCoverage: Double = 0.0,
        val consistencyPx: Double? = null,
        val roi: List<P>? = null,          // 可信对比区域（参考坐标）：与对齐平面一致的区域
        val kNew: Double = 0.0,            // 镜头径向畸变系数（新 / 参考）；h 定义在去畸变坐标下
        val kRef: Double = 0.0,
        val newW: Int = 0, val newH: Int = 0,
        val refW: Int = 0, val refH: Int = 0,
    ) {
        /** 新照片工作坐标 → 参考照片工作坐标（含畸变）。 */
        fun mapNewToRef(p: P): P = distort(h!!.apply(undistort(p, kNew, newW, newH)), kRef, refW, refH)
        /** 参考照片工作坐标 → 新照片工作坐标（含畸变）。 */
        fun mapRefToNew(p: P): P = distort(h!!.inverse().apply(undistort(p, kRef, refW, refH)), kNew, newW, newH)
    }

    data class CompareResult(
        val changedRatio: Double,
        val regions: List<IntArray>,   // x, y, w, h
        val overlapRatio: Double,
        val changeMask: Mat,
        val valid: Mat,
        val similarity: Double = 1.0,      // 对比区两图相关系数；< 0.5 表示整体差异大
        val lightRatio: Double = 0.0,      // 明暗变化（阴影、水渍、光照不均）占比，不计入 changedRatio
        val lightRegions: List<IntArray> = emptyList(),
        val glareRatio: Double = 0.0,      // 反光/过曝/死黑而未参与比较的面积占比
        val blurSigma: Double = 0.0,       // >0：参考被模糊以匹配新照片（新照片更糊）；<0：反之
    )

    // ------------------------------------------------------------------ 1. 靶标检测
    fun gray(img: Mat): Mat {
        if (img.channels() == 1) return img
        val g = Mat()
        Imgproc.cvtColor(img, g, if (img.channels() == 4) Imgproc.COLOR_RGBA2GRAY else Imgproc.COLOR_BGR2GRAY)
        return g
    }

    private fun readQuad(points: Mat): List<P>? {
        if (points.empty() || points.total() < 4) return null
        val f = Mat()
        points.convertTo(f, CvType.CV_32F)
        val arr = FloatArray((f.total() * f.channels()).toInt())
        f.get(0, 0, arr)
        if (arr.size < 8) return null
        return (0..3).map { P(arr[it * 2].toDouble(), arr[it * 2 + 1].toDouble()) }
    }

    fun detectQr(g: Mat): Pair<String, List<P>>? {
        val detectors = listOf({ QRCodeDetector() }, { QRCodeDetectorAruco() })
        for (mk in detectors) {
            for (scale in doubleArrayOf(1.0, 0.5, 1.5)) {
                val img = if (scale == 1.0) g else Mat().also {
                    Imgproc.resize(g, it, Size(), scale, scale,
                        if (scale < 1) Imgproc.INTER_AREA else Imgproc.INTER_CUBIC)
                }
                val pts = Mat()
                val text = try {
                    when (val det = mk()) {
                        is QRCodeDetector -> det.detectAndDecode(img, pts)
                        is QRCodeDetectorAruco -> det.detectAndDecode(img, pts)
                        else -> ""
                    }
                } catch (e: Exception) { "" }
                val quad = readQuad(pts)
                if (!text.isNullOrEmpty() && quad != null) return text to quad.map { it * (1.0 / scale) }
            }
        }
        return null
    }

    fun refineQrCorners(g: Mat, c: List<P>): List<P> {
        val side = (0..3).sumOf { c[it].dist(c[(it + 1) % 4]) } / 4
        val module = side / 25.0
        val win = max(2, min(8, (module * 0.6).roundToInt())).toDouble()
        val pts = c.toMat2f()
        try {
            Imgproc.cornerSubPix(g, pts, Size(win, win), Size(-1.0, -1.0),
                TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 40, 0.01))
        } catch (e: Exception) { return c }
        val r = pts.toArray()
        return c.indices.map { i ->
            val q = P(r[i].x, r[i].y)
            if (q.dist(c[i]) <= module * 0.5) q else c[i]
        }
    }

    fun detectDotCandidates(g: Mat, qr: List<P>?, qrSide: Double?): List<Dot> {
        val w = g.cols(); val h = g.rows()
        val (dMin, dMax) = if (qrSide != null && qrSide > 0) {
            val e = qrSide * DOT_TO_QR_RATIO; e * 0.35 to e * 2.5
        } else max(6.0, min(h, w) * 0.004) to min(h, w) * 0.2
        val block = max(15, ((dMax * 2).toInt() / 2) * 2 + 1)
        val blur = Mat(); Imgproc.GaussianBlur(g, blur, Size(3.0, 3.0), 0.0)
        val bw = Mat()
        Imgproc.adaptiveThreshold(blur, bw, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, block, 12.0)
        val contours = ArrayList<MatOfPoint>()
        // RETR_LIST：纹理强时，阈值后的纹理连成大片，圆点落在它的“洞”（白圈）里，外轮廓模式会漏掉
        Imgproc.findContours(bw, contours, Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE)
        val pix = ByteArray(w * h); g.get(0, 0, pix)
        val out = ArrayList<Dot>()
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < PI * (dMin / 2) * (dMin / 2) || area > PI * (dMax / 2) * (dMax / 2)) continue
            if (c.total() < 8) continue
            val c2f = MatOfPoint2f(*c.toArray())
            val per = Imgproc.arcLength(c2f, true)
            if (per <= 0) continue
            val circ = 4 * PI * area / (per * per)
            val hullIdx = org.opencv.core.MatOfInt()
            Imgproc.convexHull(c, hullIdx)
            val cp = c.toArray()
            val hullArea = convexHullArea(hullIdx.toArray().map { P(cp[it].x, cp[it].y) })
            val solidity = if (hullArea > 0) area / hullArea else 0.0
            if (circ < 0.70 || solidity < 0.92) continue
            val e = Imgproc.fitEllipse(c2f)
            val axisRatio = min(e.size.width, e.size.height) / max(e.size.width, e.size.height)
            if (axisRatio < 0.35) continue
            val mo = Imgproc.moments(c)
            if (mo.m00 == 0.0) continue
            val cx = mo.m10 / mo.m00; val cy = mo.m01 / mo.m00
            if (qr != null && pointInQuad(qr, P(cx, cy), 1.25)) continue
            val d = 2 * sqrt(area / PI)
            val r = d / 2
            // 对比度：圆内中位数 vs 外圈中位数
            val inner = ArrayList<Int>(); val ring = ArrayList<Int>()
            val x0 = max(0, (cx - 2 * r).toInt()); val x1 = min(w - 1, (cx + 2 * r).toInt())
            val y0 = max(0, (cy - 2 * r).toInt()); val y1 = min(h - 1, (cy + 2 * r).toInt())
            val step = max(1, (r / 12).toInt())
            var yy = y0
            while (yy <= y1) {
                var xx = x0
                while (xx <= x1) {
                    val rr = kotlin.math.hypot(xx - cx, yy - cy)
                    val v = pix[yy * w + xx].toInt() and 0xFF
                    if (rr < r * 0.6) inner.add(v) else if (rr > r * 1.25 && rr < r * 1.8) ring.add(v)
                    xx += step
                }
                yy += step
            }
            if (inner.isEmpty() || ring.isEmpty()) continue
            val contrast = (median(ring) - median(inner)) / 255.0
            if (contrast < 0.15) continue
            val sizeScore = if (qrSide != null && qrSide > 0) exp(-abs(ln(d / (qrSide * DOT_TO_QR_RATIO)))) else 1.0
            out.add(Dot(cx, cy, d, circ * solidity * min(1.0, contrast * 2) * sizeScore))
        }
        // 同心候选只留最小的：RETR_LIST 还会返回白圈外沿（“洞”的轮廓），与圆点同心、也“内暗外亮”，
        // 但质心受透视影响更大；真正的靶标是里面的黑圆。
        val kept = ArrayList<Dot>()
        for (c in out.sortedBy { it.d }) if (kept.none { c.p.dist(it.p) < 0.5 * it.d }) kept.add(c)
        return kept.sortedByDescending { it.score }
    }

    private fun median(v: List<Int>): Double {
        val s = v.sorted(); val n = s.size
        return if (n % 2 == 1) s[n / 2].toDouble() else (s[n / 2 - 1] + s[n / 2]) / 2.0
    }

    /** qrOverride：OpenCV 读不出时，可传入其他扫码库（ML Kit）得到的内容和四角。 */
    fun detectMarkers(work: Mat, qrOverride: Pair<String, List<P>>? = null): Markers {
        val g = gray(work)
        val qr = qrOverride ?: detectQr(g)
        var corners = qr?.second
        var side: Double? = null
        if (corners != null) {
            corners = refineQrCorners(g, corners)
            side = (0..3).sumOf { corners[it].dist(corners[(it + 1) % 4]) } / 4
        }
        return Markers(qr?.first, corners, detectDotCandidates(g, corners, side), work.cols(), work.rows())
    }

    /** 首次建档：挑得分最高、彼此分散的圆点（界面上还可以点选修改）。 */
    fun selectReferenceDots(m: Markers, maxDots: Int = 3): List<Dot> {
        val chosen = ArrayList<Dot>()
        for (c in m.dotCandidates) {
            if (c.score < 0.35) continue
            if (chosen.any { it.p.dist(c.p) < 3 * c.d }) continue
            chosen.add(c)
            if (chosen.size >= maxDots) break
        }
        return chosen
    }

    // ------------------------------------------------------------------ 2. 对齐工具
    private fun fitH(src: List<P>, dst: List<P>): Homography? {
        if (src.size < 4) return null
        return Homography.fromMat(Calib3d.findHomography(src.toMat2f(), dst.toMat2f()))
    }

    private fun perspective(src: List<P>, dst: List<P>): Homography =
        Homography.fromMat(Imgproc.getPerspectiveTransform(src.toMat2f(), dst.toMat2f()))!!

    private fun matchDots(h0: Homography, refDots: List<Dot>, cands: List<Dot>, newQrSide: Double): Pair<List<P?>, List<Double?>> {
        val inv = h0.inverse()
        val radius = max(12.0, 0.6 * newQrSide)
        val expD = newQrSide * DOT_TO_QR_RATIO
        val used = HashSet<Int>()
        val out = ArrayList<P?>(); val dists = ArrayList<Double?>()
        for (rd in refDots) {
            val pr = inv.apply(rd.p)
            var best = -1; var bd = radius
            cands.forEachIndexed { j, c ->
                if (j in used || c.d < 0.4 * expD || c.d > 2.5 * expD) return@forEachIndexed
                val dd = c.p.dist(pr)
                if (dd < bd) { best = j; bd = dd }
            }
            if (best < 0) { out.add(null); dists.add(null) } else {
                used.add(best); out.add(cands[best].p); dists.add(bd)
            }
        }
        return out to dists
    }

    fun roiGrid(refQr: List<P>, refDots: List<Dot>, w: Int, h: Int, n: Int = 9): List<P> {
        val pts = refQr + refDots.map { it.p }
        var x0 = pts.minOf { it.x }; var x1 = pts.maxOf { it.x }
        var y0 = pts.minOf { it.y }; var y1 = pts.maxOf { it.y }
        val mx = (x1 - x0) * 0.25; val my = (y1 - y0) * 0.25
        x0 = max(0.0, x0 - mx); y0 = max(0.0, y0 - my)
        x1 = min(w - 1.0, x1 + mx); y1 = min(h - 1.0, y1 + my)
        return (0 until n).flatMap { j -> (0 until n).map { i ->
            P(x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * j / (n - 1)) } }
    }

    /** 255 = 可用；0 = 靶标区域（特征和差分都排除，避免“自己验证自己”）。 */
    fun markerMask(w: Int, h: Int, qr: List<P>?, dots: List<Dot>): Mat {
        val m = Mat(h, w, CvType.CV_8U, Scalar(255.0))
        if (qr != null) {
            val cx = qr.sumOf { it.x } / 4; val cy = qr.sumOf { it.y } / 4
            val grown = qr.map { Point(cx + (it.x - cx) * 1.3, cy + (it.y - cy) * 1.3) }
            Imgproc.fillConvexPoly(m, MatOfPoint(*grown.toTypedArray()), Scalar(0.0))
        }
        for (d in dots) Imgproc.circle(m, Point(d.x, d.y), (d.d + 4).toInt(), Scalar(0.0), -1)
        return m
    }

    class FeatureMatch(val h: Homography?, val inliers: Int, val srcIn: List<P>, val dstIn: List<P>,
                       val kpRef: Int, val kpNew: Int,
                       val srcAll: List<P> = emptyList(), val dstAll: List<P> = emptyList())

    /** 参考照片的 SIFT 特征缓存：同一系列连续拍多张时，参考照片只算一次（约占单张处理时间的四分之一）。 */
    class SiftFeatures(val kp: Array<org.opencv.core.KeyPoint>, val desc: Mat)
    private val siftCache = object : LinkedHashMap<String, SiftFeatures>(8, 0.75f, true) {
        override fun removeEldestEntry(e: MutableMap.MutableEntry<String, SiftFeatures>?) = size > 6
    }

    private fun siftOf(g: Mat, mask: Mat?, n: Int, key: String?): SiftFeatures {
        val ck = key?.let { "$it|$n" }
        if (ck != null) synchronized(siftCache) { siftCache[ck] }?.let { return it }
        val k = MatOfKeyPoint(); val d = Mat()
        SIFT.create(n).detectAndCompute(g, mask ?: Mat(), k, d)
        val f = SiftFeatures(k.toArray(), d)
        if (ck != null) synchronized(siftCache) { siftCache[ck] = f }
        return f
    }

    /**
     * 对齐用 SIFT：定位精度比 ORB 高一个量级（合成测试 0.13px vs 1.15px），只慢约 30%。识别和实时引导仍用 ORB。
     * refKey：参考图像 + 掩膜的唯一标识（给了就缓存参考特征）。
     */
    fun featureHomography(refG: Mat, newG: Mat, refMask: Mat?, newMask: Mat?,
                          nFeatures: Int = 4000, ransacPx: Double = 4.0, refKey: String? = null): FeatureMatch {
        val f1 = siftOf(refG, refMask, nFeatures, refKey)
        val f2 = siftOf(newG, newMask, nFeatures, null)
        val a1 = f1.kp; val a2 = f2.kp; val d1 = f1.desc; val d2 = f2.desc
        val empty = FeatureMatch(null, 0, emptyList(), emptyList(), a1.size, a2.size)
        if (a1.size < 8 || a2.size < 8 || d1.empty() || d2.empty()) return empty
        val knn = ArrayList<MatOfDMatch>()
        BFMatcher.create(Core.NORM_L2, false).knnMatch(d2, d1, knn, 2)
        val src = ArrayList<P>(); val dst = ArrayList<P>()
        for (pair in knn) {
            val m = pair.toArray()
            if (m.size == 2 && m[0].distance < 0.8f * m[1].distance) {
                src.add(P(a2[m[0].queryIdx].pt.x, a2[m[0].queryIdx].pt.y))
                dst.add(P(a1[m[0].trainIdx].pt.x, a1[m[0].trainIdx].pt.y))
            }
        }
        if (src.size < 8) return FeatureMatch(null, src.size, emptyList(), emptyList(), a1.size, a2.size)
        return ransac(src, dst, ransacPx, a1.size, a2.size)
    }

    fun ransac(src: List<P>, dst: List<P>, px: Double, kpRef: Int = 0, kpNew: Int = 0): FeatureMatch {
        if (src.size < 8) return FeatureMatch(null, 0, emptyList(), emptyList(), kpRef, kpNew, src, dst)
        val mask = Mat()
        val h = Homography.fromMat(Calib3d.findHomography(src.toMat2f(), dst.toMat2f(), Calib3d.RANSAC, px, mask, 5000, 0.999))
            ?: return FeatureMatch(null, 0, emptyList(), emptyList(), kpRef, kpNew, src, dst)
        val inl = ByteArray(mask.total().toInt()); mask.get(0, 0, inl)
        val si = ArrayList<P>(); val di = ArrayList<P>()
        for (i in inl.indices) if (inl[i].toInt() != 0) { si.add(src[i]); di.add(dst[i]) }
        return FeatureMatch(h, si.size, si, di, kpRef, kpNew, src, dst)
    }

    // ------------------------------------------------------------------ 镜头径向畸变
    // x_d = x_u·(1 + k·r²)，画面中心为原点、长边归一化。手机镜头边缘的桶形畸变会让单应矩阵对不齐，
    // 产生大片“假变化”、甚至误判圆点移动。用特征匹配估计两张照片各自的 k，几何计算都在去畸变坐标下做。
    const val DIST_K_MIN = -0.30
    const val DIST_K_MAX = 0.20
    const val DIST_MIN_INLIERS = 80
    const val DIST_MIN_SPREAD = 0.25

    fun undistort(p: P, k: Double, w: Int, h: Int): P {
        if (k == 0.0) return p
        val f = max(w, h).toDouble(); val cx = w / 2.0; val cy = h / 2.0
        val dx = (p.x - cx) / f; val dy = (p.y - cy) / f
        var ux = dx; var uy = dy
        repeat(10) { val s = 1.0 + k * (ux * ux + uy * uy); ux = dx / s; uy = dy / s }
        return P(ux * f + cx, uy * f + cy)
    }

    fun distort(p: P, k: Double, w: Int, h: Int): P {
        if (k == 0.0) return p
        val f = max(w, h).toDouble(); val cx = w / 2.0; val cy = h / 2.0
        val ux = (p.x - cx) / f; val uy = (p.y - cy) / f
        val s = 1.0 + k * (ux * ux + uy * uy)
        return P(ux * s * f + cx, uy * s * f + cy)
    }

    /** 由特征对应点估计 (k_new, k_ref)；残差改善不明显时返回 (0, 0)。 */
    fun estimateDistortion(src: List<P>, dst: List<P>, nw: Int, nh: Int, rw: Int, rh: Int): Pair<Double, Double> {
        // 点太多时均匀抽样，估计速度与结果都稳定
        val stepN = max(1, src.size / 2000)
        val s0 = src.filterIndexed { i, _ -> i % stepN == 0 }; val d0 = dst.filterIndexed { i, _ -> i % stepN == 0 }
        fun cost(kn: Double, kr: Double): Double {
            val a = s0.map { undistort(it, kn, nw, nh) }; val b = d0.map { undistort(it, kr, rw, rh) }
            val hm = fitH(a, b) ?: return 1e9
            var acc = 0.0
            for (i in a.indices) { val e = min(6.0, hm.apply(a[i]).dist(b[i])); acc += e * e }
            return sqrt(acc / a.size) + 0.05 * (abs(kn) + abs(kr))
        }
        val c0 = cost(0.0, 0.0)
        var best = Triple(Double.MAX_VALUE, 0.0, 0.0)
        for (i in 0..10) for (j in 0..10) {
            val a = DIST_K_MIN + (DIST_K_MAX - DIST_K_MIN) * i / 10; val b = DIST_K_MIN + (DIST_K_MAX - DIST_K_MIN) * j / 10
            val c = cost(a, b); if (c < best.first) best = Triple(c, a, b)
        }
        // 局部细化（模式搜索）：当前步长下一直走到不再变好，再把步长减半
        var step = (DIST_K_MAX - DIST_K_MIN) / 20
        while (step > 2e-4) {
            val (_, kn, kr) = best
            var cand = best
            for (da in doubleArrayOf(-step, 0.0, step)) for (db in doubleArrayOf(-step, 0.0, step)) {
                if (da == 0.0 && db == 0.0) continue
                val c = cost(kn + da, kr + db); if (c < cand.first) cand = Triple(c, kn + da, kr + db)
            }
            if (cand.first < best.first) best = cand else step /= 2
        }
        return if (best.first < 0.8 * c0 && c0 - best.first > 0.15) best.second to best.third else 0.0 to 0.0
    }

    /**
     * 宽松阈值的特征匹配 → 判断能否估计畸变 → 两遍估计：
     * 第一遍用宽松内点；第二遍在去畸变坐标下用严格阈值重选内点（剔除残余误匹配）后再估计。
     */
    private fun distortionFrom(loose: FeatureMatch, nw: Int, nh: Int, rw: Int, rh: Int): Pair<Double, Double> {
        if (loose.inliers < DIST_MIN_INLIERS) return 0.0 to 0.0
        val spread = convexHullArea(loose.dstIn) / (rw.toDouble() * rh)
        if (spread < DIST_MIN_SPREAD) return 0.0 to 0.0
        val (k1n, k1r) = estimateDistortion(loose.srcIn, loose.dstIn, nw, nh, rw, rh)
        if (k1n == 0.0 && k1r == 0.0) return 0.0 to 0.0
        val strict = ransac(loose.srcAll.map { undistort(it, k1n, nw, nh) }, loose.dstAll.map { undistort(it, k1r, rw, rh) }, 3.0)
        if (strict.inliers < DIST_MIN_INLIERS) return k1n to k1r
        val keep = HashSet<P>(strict.srcIn)
        val idx = loose.srcAll.indices.filter { keep.contains(undistort(loose.srcAll[it], k1n, nw, nh)) }
        val (k2n, k2r) = estimateDistortion(idx.map { loose.srcAll[it] }, idx.map { loose.dstAll[it] }, nw, nh, rw, rh)
        return if (k2n == 0.0 && k2r == 0.0) k1n to k1r else k2n to k2r
    }

    private fun rms(h: Homography, s: List<P>, d: List<P>) =
        sqrt(s.indices.sumOf { val q = h.apply(s[it]); (q.x - d[it].x).let { x -> x * x } + (q.y - d[it].y).let { y -> y * y } } / s.size)

    // ------------------------------------------------------------------ 2a. 靶标 + 特征 对齐
    /**
     * 把新照片中参考圆点附近的区域“摆正”到参考视角（用码四角的初始矩阵），在那里找圆点。
     * 强透视下圆点被压成很扁的椭圆、全局检测会漏掉；摆正后它又是已知大小的圆。返回新照片坐标或 null。
     */
    private fun findDotRectified(newG: Mat, h0: Homography, rd: Dot): P? {
        val half = (3 * rd.d).roundToInt()
        val x0 = rd.x - half; val y0 = rd.y - half
        val t = Homography(doubleArrayOf(1.0, 0.0, -x0, 0.0, 1.0, -y0, 0.0, 0.0, 1.0))
        val patch = Mat()
        Imgproc.warpPerspective(newG, patch, (t * h0).toMat(), Size(2.0 * half, 2.0 * half),
            Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(255.0))
        val blk = max(15, ((rd.d * 2).toInt() / 2) * 2 + 1)
        val bl = Mat(); Imgproc.GaussianBlur(patch, bl, Size(3.0, 3.0), 0.0)
        val bw = Mat(); Imgproc.adaptiveThreshold(bl, bw, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, blk, 12.0)
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(bw, contours, Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE)
        val pw = patch.cols(); val ph = patch.rows()
        val pix = ByteArray(pw * ph); patch.get(0, 0, pix)
        val expArea = PI * (rd.d / 2) * (rd.d / 2)
        var best: P? = null; var bestDist = 1.5 * rd.d
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < 0.3 * expArea || area > 2.5 * expArea || c.total() < 8) continue
            val per = Imgproc.arcLength(MatOfPoint2f(*c.toArray()), true)
            if (per <= 0 || 4 * PI * area / (per * per) < 0.6) continue
            val mo = Imgproc.moments(c); if (mo.m00 == 0.0) continue
            val cx = mo.m10 / mo.m00; val cy = mo.m01 / mo.m00
            val r = sqrt(area / PI)
            val inner = ArrayList<Int>(); val ring = ArrayList<Int>()
            val xa = max(0, (cx - 2 * r).toInt()); val xb = min(pw - 1, (cx + 2 * r).toInt())
            val ya = max(0, (cy - 2 * r).toInt()); val yb = min(ph - 1, (cy + 2 * r).toInt())
            for (yy in ya..yb) for (xx in xa..xb) {
                val rr = kotlin.math.hypot(xx - cx, yy - cy)
                val v = pix[yy * pw + xx].toInt() and 0xFF
                if (rr < r * 0.6) inner.add(v) else if (rr > r * 1.25 && rr < r * 1.8) ring.add(v)
            }
            if (inner.isEmpty() || ring.isEmpty() || median(ring) - median(inner) < 40) continue
            val dist = kotlin.math.hypot(cx - half, cy - half)
            if (dist < bestDist) { best = P(cx + x0, cy + y0); bestDist = dist }
        }
        return best?.let { h0.inverse().apply(it) }
    }

    fun align(refWork: Mat, rm: Markers, refDots: List<Dot>, newWork: Mat, nm: Markers, refKey: String? = null): AlignResult {
        val newQrRaw = nm.qrCorners ?: return AlignResult("failed", "新照片中没有识别到二维码")
        val refQrRaw = rm.qrCorners ?: return AlignResult("failed", "参考照片没有二维码")
        if (nm.qrText != rm.qrText) return AlignResult("failed", "二维码不同：${nm.qrText} ≠ ${rm.qrText}")
        val sideNew = nm.qrSide(); val sideRef = rm.qrSide()
        val refG = gray(refWork); val newG = gray(newWork)

        // 码四角初始对齐：四种循环顺序都试（兼容不同扫码库的角点顺序）
        var bestKey = Pair(-1, 0.0); var newQr = newQrRaw
        var h0 = perspective(newQrRaw, refQrRaw); var matched: List<P?> = emptyList()
        for (r in 0..3) {
            val q = (0..3).map { newQrRaw[(it + r) % 4] }
            val hh = perspective(q, refQrRaw)
            val (mt, ds) = matchDots(hh, refDots, nm.dotCandidates, sideNew)
            val nOk = mt.count { it != null }
            val err = ds.filterNotNull().sum() / max(1, nOk)
            val key = Pair(nOk, -err)
            if (key.first > bestKey.first || (key.first == bestKey.first && key.second > bestKey.second)) {
                bestKey = key; newQr = q; h0 = hh; matched = mt
            }
        }
        // 全局检测没找到的圆点：摆正到参考视角再找一次（强透视下圆点很扁，全局检测会漏掉）。
        // 每找到新圆点，就用三个定位块角 + 已找到的圆点重算预测矩阵（码小时第 4 个推算角误差外推很大）。
        var hp = h0
        for (pass in 0 until 3) {
            var foundNew = false
            matched = matched.mapIndexed { i, mt ->
                mt ?: run {
                    val p = hp.inverse().apply(refDots[i].p)
                    val f = if (p.x >= 0 && p.y >= 0 && p.x < nm.workW && p.y < nm.workH) findDotRectified(newG, hp, refDots[i]) else null
                    if (f != null) foundNew = true
                    f
                }
            }
            val got = matched.indices.filter { matched[it] != null }
            if ((pass > 0 && !foundNew) || got.isEmpty() || got.size == matched.size) break
            val src = listOf(0, 1, 3).map { newQr[it] } + got.map { matched[it]!! }
            val dst = listOf(0, 1, 3).map { refQrRaw[it] } + got.map { refDots[it].p }
            hp = fitH(src, dst) ?: break
        }
        val matchedRaw = matched

        // 图像特征（遮掉靶标）：宽松阈值匹配，先用来估计镜头畸变
        val refMask = markerMask(rm.workW, rm.workH, refQrRaw, refDots)
        val newMask = markerMask(nm.workW, nm.workH, newQr, nm.dotCandidates.take(12))
        val loose = featureHomography(refG, newG, refMask, newMask, ransacPx = 10.0, refKey = refKey?.let { "$it|m" })
        val (kN, kR) = distortionFrom(loose, nm.workW, nm.workH, rm.workW, rm.workH)
        val un = { p: P -> undistort(p, kN, nm.workW, nm.workH) }
        val ur = { p: P -> undistort(p, kR, rm.workW, rm.workH) }
        // 以下几何计算都在去畸变坐标下
        newQr = newQr.map(un)
        val refQr = refQrRaw.map(ur)
        matched = matched.map { it?.let(un) }
        h0 = perspective(newQr, refQr)
        val refD = refDots.map { ur(it.p) }
        val dotD = refDots.map { it.d }

        fun ptsFor(skip: Int?, include: Set<Int>): Pair<List<P>, List<P>> {
            // 第 3 个角（右下，无定位块）是推算的，圆点够用时不参与精算
            val nDots = include.count { matched[it] != null && it != skip }
            val idx = if (nDots >= 2) listOf(0, 1, 3) else listOf(0, 1, 2, 3)
            val s = ArrayList(idx.map { newQr[it] }); val d = ArrayList(idx.map { refQr[it] })
            for (i in include.sorted()) {
                val mt = matched[i] ?: continue
                if (i == skip) continue
                s.add(mt); d.add(refD[i])
            }
            return s to d
        }

        // 每个圆点在新照片中的视直径：残差在新照片像素里量、阈值按视直径给（拍得远/斜时参考坐标误差会被放大）
        val h0Inv = h0.inverse()
        val dotDNew = refD.mapIndexed { i, p ->
            val c0 = h0Inv.apply(p); val r = dotD[i] / 2
            2 * listOf(P(p.x + r, p.y), P(p.x - r, p.y), P(p.x, p.y + r), P(p.x, p.y - r)).map { h0Inv.apply(it).dist(c0) }.average()
        }

        val allValid = matched.indices.filter { matched[it] != null }.toSet()
        val hAll = ptsFor(null, allValid).let { fitH(it.first, it.second) } ?: h0

        // 图像特征：在去畸变坐标下用严格阈值重新 RANSAC，独立求解
        var fm = ransac(loose.srcAll.map(un), loose.dstAll.map(ur), 4.0, loose.kpRef, loose.kpNew)
        val grid = roiGrid(refQr, refDots.mapIndexed { i, dt -> Dot(refD[i].x, refD[i].y, dt.d) }, rm.workW, rm.workH)
        val roiArea = max(1.0, (grid.maxOf { it.x } - grid.minOf { it.x }) * (grid.maxOf { it.y } - grid.minOf { it.y }))
        fun cov(f: FeatureMatch) = if (f.h != null && f.inliers >= 4) convexHullArea(f.dstIn) / roiArea else 0.0
        // 直接匹配不够（多半是视角太斜）：先用靶标矩阵把新照片“摆正”到参考视角再匹配。
        // 若靶标其实被搬到了别的表面，摆正后照样对不上，仍会判为“需复核”。
        if (fm.h == null || fm.inliers < FEATURE_MIN_INLIERS || cov(fm) < FEATURE_MIN_COVERAGE) {
            val size = Size(rm.workW.toDouble(), rm.workH.toDouble())
            val rect = Mat(); Imgproc.warpPerspective(newG, rect, hAll.toMat(), size, Imgproc.INTER_LINEAR)
            val ones = Mat(newG.rows(), newG.cols(), CvType.CV_8U, Scalar(255.0))
            val rmask = Mat(); Imgproc.warpPerspective(ones, rmask, hAll.toMat(), size, Imgproc.INTER_NEAREST)
            Imgproc.erode(rmask, rmask, Mat.ones(9, 9, CvType.CV_8U)); Core.bitwise_and(rmask, refMask, rmask)
            val f2 = featureHomography(refG, rect, refMask, rmask, ransacPx = 4.0, refKey = refKey?.let { "$it|m" })
            if (f2.h != null && f2.inliers > fm.inliers) {
                val inv = hAll.inverse()
                val r2 = ransac(f2.srcIn.map { inv.apply(it) }, f2.dstIn, 4.0, loose.kpRef, loose.kpNew)
                if (r2.h != null) fm = r2
            }
        }
        val coverage = cov(fm)
        val featuresOk = fm.h != null && fm.inliers >= FEATURE_MIN_INLIERS && coverage >= FEATURE_MIN_COVERAGE

        // 圆点是否移动
        val valid = allValid.toMutableSet()
        val moved = ArrayList<Int>()
        if (featuresOk) {
            // 有可靠表面纹理：直接用特征矩阵判断（独立于靶标，且不受小码外推误差影响）
            val hfi = fm.h!!.inverse()
            for (i in allValid.sorted()) {
                if (hfi.apply(refD[i]).dist(matched[i]!!) > DOT_MOVE_TOL_FRAC * dotDNew[i]) { moved.add(i); valid.remove(i) }
            }
        } else {
            // 只能靠靶标：留一法。阈值考虑外推不确定度——给点加 0.7px 噪声重算，看预测漂多远
            val rnd = java.util.Random(0)
            fun loo(vs: Set<Int>): Pair<List<Double?>, List<Double>> {
                val out = MutableList<Double?>(refDots.size) { null }; val tol = MutableList(refDots.size) { 0.0 }
                for (i in vs.sorted()) {
                    val (s0, d0) = ptsFor(i, vs)
                    val hl = fitH(s0, d0) ?: continue
                    out[i] = hl.inverse().apply(refD[i]).dist(matched[i]!!)
                    val preds = (0 until 20).mapNotNull {
                        fitH(s0.map { p -> P(p.x + rnd.nextGaussian() * 0.7, p.y + rnd.nextGaussian() * 0.7) }, d0)?.inverse()?.apply(refD[i])
                    }
                    val spread = if (preds.isEmpty()) 0.0 else {
                        val mx = preds.sumOf { it.x } / preds.size; val my = preds.sumOf { it.y } / preds.size
                        sqrt(preds.sumOf { (it.x - mx) * (it.x - mx) + (it.y - my) * (it.y - my) } / preds.size)
                    }
                    tol[i] = max(DOT_MOVE_TOL_FRAC * dotDNew[i], 3.0 * spread)
                }
                return out to tol
            }
            // 一次只剔除最离谱的一个；至少剩 3 个圆点才继续判断（再少就没有冗余）
            var (res, tols) = loo(valid)
            while (valid.size >= 3) {
                val worst = valid.filter { res[it] != null && res[it]!! > tols[it] }.maxByOrNull { res[it]!! / tols[it] } ?: break
                moved.add(worst); valid.remove(worst)
                val r2 = loo(valid); res = r2.first; tols = r2.second
            }
        }

        val hTmp = ptsFor(null, valid).let { fitH(it.first, it.second) } ?: h0
        val predAll = hTmp.inverse().apply(refD).map { distort(it, kN, nm.workW, nm.workH) }
        val outOfFrame = ArrayList<Int>(); val missing = ArrayList<Int>()
        matched.forEachIndexed { i, mt ->
            if (mt == null) {
                val p = predAll[i]
                if (p.x < 0 || p.y < 0 || p.x >= nm.workW || p.y >= nm.workH) outOfFrame.add(i) else missing.add(i)
            }
        }

        val (s, d) = ptsFor(null, valid)
        val hMk = fitH(s, d) ?: h0
        var markerRms = rms(hMk, s, d)
        // 一致性：靶标点（码角 + 未移动的圆点）在“特征矩阵”下的最大残差；在靶标点本身上比较，不外推
        val consistency = if (fm.h != null && fm.inliers >= 4) s.indices.maxOf { fm.h!!.apply(s[it]).dist(d[it]) } else null
        // 新照片比参考小（拍得远）时，参考坐标里的测量误差按比例放大
        val consTol = max(CONSISTENCY_TOL_PX, 0.06 * sideRef) * max(1.0, sideRef / max(sideNew, 1e-6))

        var status = "confirmed"; var h = hMk; var note = ""; var roi: List<P>? = null
        if (!featuresOk) {
            if (fm.kpRef >= TEXTURE_RICH_KP && fm.kpNew >= TEXTURE_RICH_KP) {
                status = "conflict"; note = "表面纹理丰富（${fm.kpRef}/${fm.kpNew} 个特征点）但几乎无法匹配"
            } else status = "markers_only"
        } else if (consistency != null && consistency > consTol) {
            status = "conflict"; note = "靶标与表面纹理相差 %.1fpx".format(consistency)
        } else {
            // 融合精算：靶标点 + 特征内点，每个点同等权重（圆点质心精度约 0.2~0.8px，
            // 上千个特征点合起来精得多；给靶标加大权重反而会把矩阵带偏）。特征少时靶标自然占更大比重。
            val w = 1
            val srcAll = ArrayList<P>(); val dstAll = ArrayList<P>()
            repeat(w) { srcAll.addAll(s); dstAll.addAll(d) }
            srcAll.addAll(fm.srcIn); dstAll.addAll(fm.dstIn)
            val hf = Homography.fromMat(Calib3d.findHomography(srcAll.toMat2f(), dstAll.toMat2f(), Calib3d.RANSAC, 3.0))
            if (hf != null) { h = hf; markerRms = rms(hf, s, d) }
            roi = convexHull(fm.dstIn + d).map { distort(it, kR, rm.workW, rm.workH) }
        }
        if (roi == null) {   // 没有可靠特征：只比较靶标附近（外扩 25%）
            val x0 = grid.minOf { it.x }; val x1 = grid.maxOf { it.x }; val y0 = grid.minOf { it.y }; val y1 = grid.maxOf { it.y }
            roi = listOf(P(x0, y0), P(x1, y0), P(x1, y1), P(x0, y1)).map { distort(it, kR, rm.workW, rm.workH) }
        }
        val movedU = moved.distinct().sorted()
        if (status == "confirmed" && (movedU.isNotEmpty() || missing.isNotEmpty())) status = "dot_moved"
        val extra = ArrayList<String>()
        if (note.isNotEmpty()) extra.add(note)
        if (missing.isNotEmpty()) extra.add("画面内未找到圆点 " + missing.joinToString(",") { "${it + 1}" })
        if (movedU.isNotEmpty()) extra.add("疑似移动圆点 " + movedU.joinToString(",") { "${it + 1}" })
        if (outOfFrame.isNotEmpty()) extra.add("圆点 " + outOfFrame.joinToString(",") { "${it + 1}" } + " 不在画面内")
        if (kN != 0.0 || kR != 0.0) extra.add("已校正镜头畸变（k=%+.3f/%+.3f）".format(kN, kR))
        val msg = STATUS_TEXT[status]!! + if (extra.isEmpty()) "" else "；" + extra.joinToString("；")
        return AlignResult(status, msg, h, matchedRaw, movedU, markerRms, fm.inliers, coverage, consistency, roi,
            kN, kR, nm.workW, nm.workH, rm.workW, rm.workH)
    }

    // ------------------------------------------------------------------ 2b. 无码：特征识别与纯特征对齐
    class Signature(val pts: List<P>, val desc: Mat, val w: Int, val h: Int)

    fun signature(work: Mat, side: Int = ID_SIDE, nFeatures: Int = 3000): Signature {
        val g = gray(work)
        val s = min(1.0, side.toDouble() / max(g.cols(), g.rows()))
        val small = if (s < 1.0) Mat().also { Imgproc.resize(g, it, Size(), s, s, Imgproc.INTER_AREA) } else g
        val orb = ORB.create(nFeatures, 1.2f, 8, 31, 0, 2, 0, 31, 10)
        val kp = MatOfKeyPoint(); val desc = Mat()
        orb.detectAndCompute(small, Mat(), kp, desc)
        return Signature(kp.toArray().map { P(it.pt.x, it.pt.y) }, desc, small.cols(), small.rows())
    }

    /** 返回 (内点数, 覆盖率, H 新→参考[识别分辨率])。 */
    fun matchSignatures(n: Signature, r: Signature): Triple<Int, Double, Homography?> {
        if (n.pts.size < 8 || r.pts.size < 8 || n.desc.empty() || r.desc.empty()) return Triple(0, 0.0, null)
        val knn = ArrayList<MatOfDMatch>()
        BFMatcher.create(Core.NORM_HAMMING, false).knnMatch(n.desc, r.desc, knn, 2)
        val src = ArrayList<P>(); val dst = ArrayList<P>()
        for (pair in knn) {
            val m = pair.toArray()
            if (m.size == 2 && m[0].distance < 0.8f * m[1].distance) { src.add(n.pts[m[0].queryIdx]); dst.add(r.pts[m[0].trainIdx]) }
        }
        if (src.size < 8) return Triple(0, 0.0, null)
        val mask = Mat()
        val h = Homography.fromMat(Calib3d.findHomography(src.toMat2f(), dst.toMat2f(), Calib3d.RANSAC, 5.0, mask, 3000, 0.995))
            ?: return Triple(0, 0.0, null)
        val inl = ByteArray(mask.total().toInt()); mask.get(0, 0, inl)
        val di = dst.filterIndexed { i, _ -> inl[i].toInt() != 0 }
        return Triple(di.size, convexHullArea(di) / (r.w.toDouble() * r.h), h)
    }

    data class Candidate<K>(val key: K, val inliers: Int, val coverage: Double)
    data class Identify<K>(val decision: String, val best: K?, val ranking: List<Candidate<K>>, val message: String)

    /** 判断新照片属于哪个系列；背景相似、分不清时返回 ambiguous。 */
    fun <K> identify(n: Signature, refs: List<Pair<K, Signature>>,
                     minInliers: Int = ID_MIN_INLIERS, minCoverage: Double = ID_MIN_COVERAGE): Identify<K> {
        val rank = refs.map { (k, s) -> val (i, c, _) = matchSignatures(n, s); Candidate(k, i, c) }
            .sortedByDescending { it.inliers }
        if (rank.isEmpty() || rank[0].inliers < minInliers || rank[0].coverage < minCoverage)
            return Identify("no_match", null, rank, "没有找到特征足够吻合的系列")
        val best = rank[0]; val second = rank.getOrNull(1)?.inliers ?: 0
        if (second >= ID_MIN_INLIERS * 0.5 && best.inliers < ID_UNIQUE_RATIO * second)
            return Identify("ambiguous", null, rank, "与多个系列都相似（${best.inliers} vs $second 个匹配点），请贴二维码或手动选择")
        return Identify("unique", best.key, rank, "按图像特征识别（${best.inliers} 个匹配点）")
    }

    fun alignFeaturesOnly(refWork: Mat, newWork: Mat, refKey: String? = null): AlignResult {
        val nw = newWork.cols(); val nh = newWork.rows(); val rw = refWork.cols(); val rh = refWork.rows()
        val loose = featureHomography(gray(refWork), gray(newWork), null, null, ransacPx = 10.0, refKey = refKey?.let { "$it|n" })
        if (loose.h == null || loose.inliers < FEAT_ALIGN_MIN_INLIERS)
            return AlignResult("failed", STATUS_TEXT["failed"] + "：图像特征不足（${loose.inliers} 个匹配点）", featureInliers = loose.inliers)
        val (kN, kR) = distortionFrom(loose, nw, nh, rw, rh)
        val fm = ransac(loose.srcAll.map { undistort(it, kN, nw, nh) }, loose.dstAll.map { undistort(it, kR, rw, rh) }, 4.0)
        if (fm.h == null || fm.inliers < FEAT_ALIGN_MIN_INLIERS)
            return AlignResult("failed", STATUS_TEXT["failed"] + "：图像特征不足（${fm.inliers} 个匹配点）", featureInliers = fm.inliers)
        val cov = convexHullArea(fm.dstIn) / (rw.toDouble() * rh)
        var msg = STATUS_TEXT["features_only"] + "（${fm.inliers} 个匹配点，覆盖 ${(cov * 100).roundToInt()}%）"
        if (kN != 0.0 || kR != 0.0) msg += "；已校正镜头畸变（k=%+.3f/%+.3f）".format(kN, kR)
        return AlignResult("features_only", msg, fm.h, featureInliers = fm.inliers, featureCoverage = cov,
            roi = convexHull(fm.dstIn).map { distort(it, kR, rw, rh) }, kNew = kN, kRef = kR, newW = nw, newH = nh, refW = rw, refH = rh)
    }

    /**
     * 新照片的二维码读不出（太小、太糊、太斜）但参考照片有码时：先按特征对齐，把参考码所在区域
     * “摆正并放大”后再解码；读出且内容一致，就把四角映射回新照片。decoder 为解码函数（OpenCV / ML Kit）。
     */
    fun recoverQr(refWork: Mat, rm: Markers, newWork: Mat, decoder: (Mat) -> Pair<String, List<P>>?): Pair<String, List<P>>? {
        val qr = rm.qrCorners ?: return null
        val fo = alignFeaturesOnly(refWork, newWork)
        if (fo.h == null) return null
        val side = rm.qrSide()
        val x0 = max(0.0, qr.minOf { it.x } - side); val y0 = max(0.0, qr.minOf { it.y } - side)
        val x1 = min(rm.workW - 1.0, qr.maxOf { it.x } + side); val y1 = min(rm.workH - 1.0, qr.maxOf { it.y } + side)
        val (aligned, _) = warpToRef(newWork, fo)
        val crop = Mat(aligned, org.opencv.core.Rect(x0.toInt(), y0.toInt(), (x1 - x0).toInt(), (y1 - y0).toInt()))
        val scale = max(1.0, 300.0 / side)          // 放大到码边长约 300 像素再解码
        val big = Mat(); Imgproc.resize(crop, big, Size(), scale, scale, Imgproc.INTER_CUBIC)
        val g = gray(big)
        val dec = detectQr(g) ?: decoder(big) ?: return null
        if (dec.first != rm.qrText) return null
        val corners = dec.second.map { fo.mapRefToNew(P(x0.toInt() + it.x / scale, y0.toInt() + it.y / scale)) }
        return dec.first to corners
    }

    // ------------------------------------------------------------------ 3. 对比
    /** 对比区域：对齐平面上的可信区域（略外扩）去掉靶标本身；墙外背景等视差区不参与比较。 */
    fun comparisonMask(w: Int, h: Int, res: AlignResult, qr: List<P>?, dots: List<Dot>): Mat {
        val m = roiMask(w, h, res.roi)
        if (qr != null) Core.bitwise_and(m, markerMask(w, h, qr, dots), m)
        return m
    }

    /** 把新照片变换到参考坐标，返回 (对齐图, 有效区域)。kNew/kRef 为镜头畸变（0 = 无）。 */
    fun warpToRef(newWork: Mat, h: Homography, w: Int, hgt: Int, kNew: Double = 0.0, kRef: Double = 0.0): Pair<Mat, Mat> {
        val aligned = Mat()
        if (kNew == 0.0 && kRef == 0.0) {
            Imgproc.warpPerspective(newWork, aligned, h.toMat(), Size(w.toDouble(), hgt.toDouble()),
                Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0, 255.0))
        } else {
            val (mx, my) = remapMaps(newWork.cols(), newWork.rows(), h, w, hgt, kNew, kRef)
            Imgproc.remap(newWork, aligned, mx, my, Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0, 255.0))
        }
        return aligned to validMask(newWork.cols(), newWork.rows(), h, w, hgt, kNew, kRef)
    }

    fun warpToRef(newWork: Mat, r: AlignResult): Pair<Mat, Mat> = warpToRef(newWork, r.h!!, r.refW, r.refH, r.kNew, r.kRef)

    /** 参考像素 → 去畸变 → H⁻¹ → 加新照片畸变 → 新照片像素。映射很平滑，先在 8 像素网格上算再双线性插值。 */
    private fun remapMaps(srcW: Int, srcH: Int, h: Homography, w: Int, hgt: Int, kNew: Double, kRef: Double): Pair<Mat, Mat> {
        val step = 8
        val gw = w / step + 2; val gh = hgt / step + 2
        val inv = h.inverse()
        val qx = DoubleArray(gw * gh); val qy = DoubleArray(gw * gh)
        for (j in 0 until gh) for (i in 0 until gw) {
            val q = distort(inv.apply(undistort(P(i * step.toDouble(), j * step.toDouble()), kRef, w, hgt)), kNew, srcW, srcH)
            qx[j * gw + i] = q.x; qy[j * gw + i] = q.y
        }
        val mx = FloatArray(w * hgt); val my = FloatArray(w * hgt)
        for (y in 0 until hgt) {
            val gy = y / step; val fy = (y - gy * step).toDouble() / step
            for (x in 0 until w) {
                val gx = x / step; val fx = (x - gx * step).toDouble() / step
                val i00 = gy * gw + gx; val i01 = i00 + 1; val i10 = i00 + gw; val i11 = i10 + 1
                mx[y * w + x] = ((qx[i00] * (1 - fx) + qx[i01] * fx) * (1 - fy) + (qx[i10] * (1 - fx) + qx[i11] * fx) * fy).toFloat()
                my[y * w + x] = ((qy[i00] * (1 - fx) + qy[i01] * fx) * (1 - fy) + (qy[i10] * (1 - fx) + qy[i11] * fx) * fy).toFloat()
            }
        }
        val a = Mat(hgt, w, CvType.CV_32F); a.put(0, 0, mx)
        val b = Mat(hgt, w, CvType.CV_32F); b.put(0, 0, my)
        return a to b
    }

    /** 新照片（srcW×srcH）变换到参考坐标后的有效区域；边缘插值不可靠，收进约 0.8% 画面尺寸。 */
    fun validMask(srcW: Int, srcH: Int, h: Homography, w: Int, hgt: Int, kNew: Double = 0.0, kRef: Double = 0.0): Mat {
        val ones = Mat(srcH, srcW, CvType.CV_8U, Scalar(255.0))
        val valid = Mat()
        if (kNew == 0.0 && kRef == 0.0) {
            Imgproc.warpPerspective(ones, valid, h.toMat(), Size(w.toDouble(), hgt.toDouble()), Imgproc.INTER_NEAREST)
        } else {
            val (mx, my) = remapMaps(srcW, srcH, h, w, hgt, kNew, kRef)
            Imgproc.remap(ones, valid, mx, my, Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, Scalar(0.0))
        }
        val k = max(7, (0.008 * max(w, hgt)).toInt()) or 1
        Imgproc.erode(valid, valid, Mat.ones(k, k, CvType.CV_8U))
        return valid
    }

    /** 多边形区域（不外扩：外扩会漏到平面外的背景）；roi 为空时整幅可用。 */
    fun roiMask(w: Int, h: Int, roi: List<P>?): Mat {
        if (roi == null || roi.size < 3) return Mat(h, w, CvType.CV_8U, Scalar(255.0))
        val m = Mat.zeros(h, w, CvType.CV_8U)
        Imgproc.fillConvexPoly(m, MatOfPoint(*roi.map { Point(it.x, it.y) }.toTypedArray()), Scalar(255.0))
        return m
    }

    const val ILLUM_SIGMA = 10.0        // 光照归一化窗口：越小越能压住阴影
    const val LIGHT_MEDIAN_HALF = 21    // 明暗/结构分类的中值窗口（半分辨率像素）
    const val STRUCT_MIN = 0.10         // 结构残差（对数亮度比）下限

    /**
     * 对数亮度比去掉“台阶状”分量后的残差。阴影、水渍是乘性的明暗变化：对数比在区域内近似常数、
     * 边界是台阶，大窗口中值能保留；裂缝、剥落等结构变化是细线/纹理改变，中值会去掉 → 残差大。
     */
    private fun lightResidual(refWork: Mat, aligned: Mat): Mat {
        fun logOf(m: Mat): Mat {
            val f = Mat(); gray(m).convertTo(f, CvType.CV_32F); Core.add(f, Scalar(4.0), f); Core.log(f, f); return f
        }
        val lr = Mat(); Core.subtract(logOf(aligned), logOf(refWork), lr)
        val small = Mat(); Imgproc.resize(lr, small, Size(), 0.5, 0.5, Imgproc.INTER_AREA)
        val q = Mat(); small.convertTo(q, CvType.CV_8U, 255.0 / 4.0, 2.0 * 255.0 / 4.0)   // [-2,2] → [0,255]
        Imgproc.medianBlur(q, q, LIGHT_MEDIAN_HALF)
        val med = Mat(); q.convertTo(med, CvType.CV_32F, 4.0 / 255.0, -2.0)
        val medFull = Mat(); Imgproc.resize(med, medFull, lr.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val out = Mat(); Core.absdiff(lr, medFull, out)
        return out
    }

    const val GLARE_SAT = 245.0
    const val GLARE_DARK = 4.0
    val GLARE_HALO = kotlin.math.ln(1.3)

    /**
     * 两张中任一张过曝/反光/死黑的区域（不可比较）。镜面反光是加性的：中心饱和、周围纹理被冲淡，
     * 既不是结构变化也不是乘性阴影，只能排除。光晕：与饱和核连通、且局部亮度朝“变亮”方向明显偏离的区域。
     */
    private fun glareMask(refWork: Mat, aligned: Mat, valid: Mat): Mat {
        val ga = gray(refWork); val gb = gray(aligned)
        val k3 = Mat.ones(3, 3, CvType.CV_8U)
        fun op(m: Mat): Mat { Core.bitwise_and(m, valid, m); Imgproc.morphologyEx(m, m, Imgproc.MORPH_OPEN, k3); return m }
        fun sat(g: Mat) = op(Mat().also { Imgproc.threshold(g, it, GLARE_SAT - 1, 255.0, Imgproc.THRESH_BINARY) })
        val satA = sat(ga); val satB = sat(gb)
        val dA = Mat(); Imgproc.threshold(ga, dA, GLARE_DARK, 255.0, Imgproc.THRESH_BINARY_INV)
        val dB = Mat(); Imgproc.threshold(gb, dB, GLARE_DARK, 255.0, Imgproc.THRESH_BINARY_INV)
        val out = Mat(); Core.bitwise_or(dA, dB, out); op(out)          // 死黑只排除本身，不向外生长
        val nA = Core.countNonZero(satA); val nB = Core.countNonZero(satB)
        if (nA == 0 && nB == 0 && Core.countNonZero(out) == 0) return out
        val w = valid.cols(); val h = valid.rows()
        if (nA > 0 || nB > 0) {
            val vf = Mat(); valid.convertTo(vf, CvType.CV_32F, 1.0 / 255)
            val den = Mat(); Imgproc.GaussianBlur(vf, den, Size(0.0, 0.0), 6.0); Core.add(den, Scalar(1e-3), den)
            fun logMean(g: Mat): Mat {
                val f = Mat(); g.convertTo(f, CvType.CV_32F); Core.multiply(f, vf, f)
                Imgproc.GaussianBlur(f, f, Size(0.0, 0.0), 6.0); Core.divide(f, den, f)
                Core.add(f, Scalar(4.0), f); Core.log(f, f); return f
            }
            val d = Mat(); Core.subtract(logMean(gb), logMean(ga), d)
            val df = FloatArray(w * h); d.get(0, 0, df)
            val vb = ByteArray(w * h); valid.get(0, 0, vb)
            val sample = ArrayList<Float>()
            for (i in df.indices step 7) if (vb[i].toInt() != 0) sample.add(df[i])
            if (sample.isEmpty()) return out
            sample.sort(); val med = sample[sample.size / 2]
            val ob = ByteArray(w * h); out.get(0, 0, ob)
            // 新照片反光 → 新照片更亮（d>0）；参考照片反光 → d<0
            for ((core, sign) in listOf(satB to 1, satA to -1)) {
                if (Core.countNonZero(core) == 0) continue
                val cb = ByteArray(w * h); core.get(0, 0, cb)
                val grow = ByteArray(w * h)
                for (i in df.indices) if (cb[i].toInt() != 0 || (vb[i].toInt() != 0 && sign * (df[i] - med) > GLARE_HALO)) grow[i] = 1
                val gm = Mat(h, w, CvType.CV_8U); gm.put(0, 0, grow)
                val labels = Mat(); val n = Imgproc.connectedComponents(gm, labels, 8, CvType.CV_32S)
                val lab = IntArray(w * h); labels.get(0, 0, lab)
                val keep = BooleanArray(n)
                for (i in lab.indices) if (cb[i].toInt() != 0) keep[lab[i]] = true
                keep[0] = false
                for (i in lab.indices) if (keep[lab[i]]) ob[i] = 0xFF.toByte()
            }
            out.put(0, 0, ob)
        }
        Imgproc.dilate(out, out, Mat.ones(7, 7, CvType.CV_8U))
        return out
    }

    private val BLUR_SIGMAS = doubleArrayOf(0.5, 0.8, 1.1, 1.5, 2.0, 2.6, 3.4)

    /**
     * 两张清晰度不同（对焦、手抖、远近）时，把较清晰的一张模糊到与另一张一致：半分辨率上搜索使平均差最小的 σ。
     * 返回 σ：>0 表示 a 被模糊，<0 表示 b 被模糊（原地修改 a / b）。
     */
    private fun matchSharpness(a: Mat, b: Mat, m: Mat): Double {
        val sa = Mat(); Imgproc.resize(a, sa, Size(), 0.5, 0.5, Imgproc.INTER_AREA)
        val sb = Mat(); Imgproc.resize(b, sb, Size(), 0.5, 0.5, Imgproc.INTER_AREA)
        val sm = Mat(); Imgproc.resize(m, sm, sa.size(), 0.0, 0.0, Imgproc.INTER_NEAREST)
        if (Core.countNonZero(sm) < 100) return 0.0
        val d = Mat(); val bl = Mat()
        fun err(x: Mat, y: Mat): Double { Core.absdiff(x, y, d); return Core.mean(d, sm).`val`[0] }
        var best = err(sa, sb); var bestS = 0.0
        for ((sign, src, other) in listOf(Triple(1, sa, sb), Triple(-1, sb, sa))) for (s in BLUR_SIGMAS) {
            Imgproc.GaussianBlur(src, bl, Size(0.0, 0.0), s / 2)
            val e = err(bl, other)
            if (e < best * 0.97) { best = e; bestS = sign * s }
        }
        if (bestS > 0) Imgproc.GaussianBlur(a, a, Size(0.0, 0.0), bestS)
        else if (bestS < 0) Imgproc.GaussianBlur(b, b, Size(0.0, 0.0), -bestS)
        return bestS
    }

    /** 拍摄质量提示：反光/过曝、清晰度不一致、偏暗。 */
    fun qualityWarnings(c: CompareResult, newWork: Mat? = null): List<String> {
        val w = ArrayList<String>()
        if (c.glareRatio > 0.025) w.add("反光/过曝区域 %.0f%% 未参与比较，建议换个角度或避开强光重拍".format(c.glareRatio * 100))
        if (c.blurSigma >= 1.5) w.add("新照片比参考模糊（对焦或手抖），细小裂缝可能漏检，建议重拍")
        else if (c.blurSigma <= -1.5) w.add("参考照片比新照片模糊，可考虑把更清晰的一张设为参考")
        if (newWork != null && Core.mean(gray(newWork)).`val`[0] < 45) w.add("照片偏暗，建议补光后重拍")
        return w
    }

    private fun illumNorm(g: Mat, valid: Mat): Mat {
        val f = Mat(); g.convertTo(f, CvType.CV_32F); Core.add(f, Scalar(1.0), f)
        val v = Mat(); valid.convertTo(v, CvType.CV_32F, 1.0 / 255)
        val fv = Mat(); Core.multiply(f, v, fv)
        val num = Mat(); Imgproc.GaussianBlur(fv, num, Size(0.0, 0.0), ILLUM_SIGMA)
        val den = Mat(); Imgproc.GaussianBlur(v, den, Size(0.0, 0.0), ILLUM_SIGMA); Core.add(den, Scalar(1e-3), den)
        val local = Mat(); Core.divide(num, den, local); Core.max(local, Scalar(1.0), local)
        val out = Mat(); Core.divide(f, local, out)
        return out
    }

    private fun minShiftDiff(a: Mat, b: Mat): Mat {
        var diff: Mat? = null
        val d = Mat(); val bs = Mat()
        for (dy in -1..1) for (dx in -1..1) {
            val mt = Mat(2, 3, CvType.CV_64F); mt.put(0, 0, 1.0, 0.0, dx.toDouble(), 0.0, 1.0, dy.toDouble())
            Imgproc.warpAffine(b, bs, mt, b.size(), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)
            Core.absdiff(a, bs, d)
            if (diff == null) diff = d.clone() else Core.min(diff, d, diff)
        }
        return diff!!
    }

    /** 参考照片相对新照片的线分辨率比：>1 表示新照片拍得更远/更小，放大后比参考模糊。 */
    fun resolutionRatio(r: AlignResult): Double {
        val roi = r.roi; val h = r.h
        if (h == null || roi.isNullOrEmpty()) return 1.0
        val cx = roi.sumOf { it.x } / roi.size; val cy = roi.sumOf { it.y } / roi.size
        val inv = h.inverse()
        val back = listOf(P(cx - 20, cy - 20), P(cx + 20, cy - 20), P(cx + 20, cy + 20), P(cx - 20, cy + 20)).map {
            distort(inv.apply(undistort(it, r.kRef, r.refW, r.refH)), r.kNew, r.newW, r.newH) }
        return sqrt(1600.0 / max(convexHullArea(back), 1e-6))
    }

    /**
     * resRatio = resolutionRatio(res)。新照片分辨率更低（拍得远）时，在两者中较低的分辨率下比较：
     * 参考坐标里 1 个新照片像素可能对应好几个参考像素，按参考分辨率比较会把配准误差和细纹理当成变化。
     */
    fun compare(refWork: Mat, aligned: Mat, valid: Mat, exclude: Mat?, sensitivity: Double = 6.0, resRatio: Double = 1.0): CompareResult {
        if (resRatio > 1.2) {
            val f = 1.0 / min(resRatio, 4.0)
            val w0 = refWork.cols(); val h0 = refWork.rows()
            val sz = Size(max(16, (w0 * f).toInt()).toDouble(), max(16, (h0 * f).toInt()).toDouble())
            fun rs(m: Mat, interp: Int) = Mat().also { Imgproc.resize(m, it, sz, 0.0, 0.0, interp) }
            val c = compare(rs(refWork, Imgproc.INTER_AREA), rs(aligned, Imgproc.INTER_AREA), rs(valid, Imgproc.INTER_NEAREST),
                exclude?.let { rs(it, Imgproc.INTER_NEAREST) }, sensitivity, 1.0)
            val sx = w0 / sz.width; val sy = h0 / sz.height
            fun up(l: List<IntArray>) = l.map { intArrayOf((it[0] * sx).toInt(), (it[1] * sy).toInt(),
                kotlin.math.ceil(it[2] * sx).toInt(), kotlin.math.ceil(it[3] * sy).toInt()) }
            val mask = Mat(); Imgproc.resize(c.changeMask, mask, Size(w0.toDouble(), h0.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
            return c.copy(regions = up(c.regions), lightRegions = up(c.lightRegions), changeMask = mask, valid = valid)
        }
        val m = valid.clone()
        if (exclude != null) Core.bitwise_and(m, exclude, m)
        val glare = glareMask(refWork, aligned, valid)
        val mAll = m.clone()
        val notGlare = Mat(); Core.bitwise_not(glare, notGlare); Core.bitwise_and(m, notGlare, m)
        val nAll = Core.countNonZero(mAll)
        val glareRatio = if (nAll > 0) (nAll - Core.countNonZero(m)).toDouble() / nAll else 0.0
        val a = illumNorm(gray(refWork), valid); val b = illumNorm(gray(aligned), valid)
        val blurS = matchSharpness(a, b, m)
        Imgproc.GaussianBlur(a, a, Size(3.0, 3.0), 0.0); Imgproc.GaussianBlur(b, b, Size(3.0, 3.0), 0.0)
        val diff = minShiftDiff(a, b)
        Core.max(diff, minShiftDiff(b, a), diff)
        val w = m.cols(); val h = m.rows()
        val mb = ByteArray(w * h); m.get(0, 0, mb)
        val df = FloatArray(w * h); diff.get(0, 0, df)
        val nValid = Core.countNonZero(m)
        // 稳健统计只需抽样：每隔 3 个像素取一个，用原始类型数组避免上百万次装箱
        val vals = sampleWhere(mb, df, 3)
        val empty = Mat.zeros(h, w, CvType.CV_8U)
        if (vals.size < 100) return CompareResult(0.0, emptyList(), nValid.toDouble() / (w * h), empty, valid, 0.0,
            glareRatio = glareRatio, blurSigma = blurS)
        // 整体相似度（相关系数）：自适应阈值只能发现局部变化，整体都不同要靠它提示
        val ma = org.opencv.core.MatOfDouble(); val sa = org.opencv.core.MatOfDouble()
        val mb2 = org.opencv.core.MatOfDouble(); val sb = org.opencv.core.MatOfDouble()
        Core.meanStdDev(a, ma, sa, m); Core.meanStdDev(b, mb2, sb, m)
        val ab = Mat(); Core.multiply(a, b, ab)
        val cov = Core.mean(ab, m).`val`[0] - ma.toArray()[0] * mb2.toArray()[0]
        val sim = cov / (sa.toArray()[0] * sb.toArray()[0] + 1e-9)
        val (med, mad0) = medianMad(vals)
        val mad = mad0 * 1.4826 + 1e-6
        val thr = max(med + sensitivity * mad, 0.12)

        val mask = Mat()
        Imgproc.threshold(diff, mask, thr, 255.0, Imgproc.THRESH_BINARY)
        mask.convertTo(mask, CvType.CV_8U)
        Core.bitwise_and(mask, m, mask)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, Mat.ones(2, 2, CvType.CV_8U))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, Mat.ones(5, 5, CvType.CV_8U))
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(mask, labels, stats, cents, 8, CvType.CV_32S)
        val minArea = max(12, (2e-5 * w * h).toInt())
        val keep = BooleanArray(n)
        val st = IntArray(5)
        for (i in 1 until n) { stats.get(i, 0, st); if (st[4] >= minArea) keep[i] = true }
        val lab = IntArray(w * h); labels.get(0, 0, lab)

        // 结构 / 明暗 分类：区域内结构残差的 90% 分位超过阈值 → 结构变化
        val resid = lightResidual(refWork, aligned)
        val rf = FloatArray(w * h); resid.get(0, 0, rf)
        val (rMed, rMad) = medianMad(sampleWhere(mb, rf, 3))
        val structThr = max(rMed + 6 * (rMad * 1.4826 + 1e-6), STRUCT_MIN)
        val perLabel = HashMap<Int, ArrayList<Float>>()
        for (i in lab.indices) if (keep[lab[i]]) perLabel.getOrPut(lab[i]) { ArrayList() }.add(rf[i])
        val structural = BooleanArray(n)
        val regions = ArrayList<IntArray>(); val lightRegions = ArrayList<IntArray>()
        for ((l, v) in perLabel) {
            v.sort()
            structural[l] = v[(v.size * 0.9).toInt().coerceAtMost(v.size - 1)] > structThr
            stats.get(l, 0, st)
            (if (structural[l]) regions else lightRegions).add(intArrayOf(st[0], st[1], st[2], st[3]))
        }
        val ob = ByteArray(w * h)
        val glb = ByteArray(w * h); glare.get(0, 0, glb)
        val allb = ByteArray(w * h); mAll.get(0, 0, allb)
        for (i in ob.indices) if (glb[i].toInt() != 0 && allb[i].toInt() != 0) ob[i] = 0x40    // 64 = 反光/过曝，未比较
        var changed = 0; var lightN = 0
        for (i in lab.indices) {
            val l = lab[i]
            if (!keep[l]) continue
            if (structural[l]) { ob[i] = 0xFF.toByte(); changed++ } else { ob[i] = 0x80.toByte(); lightN++ }
        }
        val out = Mat(h, w, CvType.CV_8U); out.put(0, 0, ob)
        return CompareResult(changed.toDouble() / max(1, nValid), regions, nValid.toDouble() / (w * h), out, valid, sim,
            lightN.toDouble() / max(1, nValid), lightRegions, glareRatio, blurS)
    }

    private fun sampleWhere(mask: ByteArray, v: FloatArray, step: Int): FloatArray {
        var n = 0
        for (i in mask.indices step step) if (mask[i].toInt() != 0) n++
        val out = FloatArray(n); var j = 0
        for (i in mask.indices step step) if (mask[i].toInt() != 0) out[j++] = v[i]
        return out
    }

    /** 中位数与绝对中位差（未乘 1.4826）。会打乱输入数组。 */
    private fun medianMad(v: FloatArray): Pair<Double, Double> {
        if (v.isEmpty()) return 0.0 to 0.0
        java.util.Arrays.sort(v)
        val med = v[v.size / 2]
        for (i in v.indices) v[i] = abs(v[i] - med)
        java.util.Arrays.sort(v)
        return med.toDouble() to v[v.size / 2].toDouble()
    }

    /** 变化图：结构变化红色 + 黄框，明暗变化浅蓝 + 蓝框，无效区变暗。输入输出均为 RGBA。 */
    fun changesView(refWork: Mat, c: CompareResult): Mat {
        val heat = refWork.clone()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(7.0, 7.0))
        // 明暗变化（值 128）
        val lightM = Mat(); Core.inRange(c.changeMask, Scalar(100.0), Scalar(160.0), lightM)
        Imgproc.dilate(lightM, lightM, kernel)
        val blue = Mat(heat.size(), heat.type(), Scalar(0.0, 140.0, 255.0, 255.0))
        val bl = Mat(); Core.addWeighted(heat, 0.6, blue, 0.4, 0.0, bl); bl.copyTo(heat, lightM)
        for (r in c.lightRegions) Imgproc.rectangle(heat, Point(r[0] - 6.0, r[1] - 6.0),
            Point(r[0] + r[2] + 6.0, r[1] + r[3] + 6.0), Scalar(0.0, 200.0, 255.0, 255.0), 3)
        // 结构变化（值 255）；细裂缝只有一两个像素宽，显示时加粗
        val red = Mat(heat.size(), heat.type(), Scalar(255.0, 0.0, 0.0, 255.0))
        val blend = Mat(); Core.addWeighted(heat, 0.35, red, 0.65, 0.0, blend)
        val shown = Mat(); Core.inRange(c.changeMask, Scalar(200.0), Scalar(255.0), shown)
        Imgproc.dilate(shown, shown, kernel)
        blend.copyTo(heat, shown)
        // 反光/过曝（值 64）：灰白蒙版
        val glM = Mat(); Core.inRange(c.changeMask, Scalar(50.0), Scalar(80.0), glM)
        val white = Mat(heat.size(), heat.type(), Scalar(200.0, 200.0, 200.0, 255.0))
        val gw = Mat(); Core.addWeighted(heat, 0.35, white, 0.65, 0.0, gw); gw.copyTo(heat, glM)
        val inv = Mat(); Core.bitwise_not(c.valid, inv)
        val dim = Mat(); heat.convertTo(dim, -1, 0.4, 0.0)
        dim.copyTo(heat, inv)
        for (r in c.regions) Imgproc.rectangle(heat, Point(r[0] - 6.0, r[1] - 6.0),
            Point(r[0] + r[2] + 6.0, r[1] + r[3] + 6.0), Scalar(255.0, 230.0, 0.0, 255.0), 5)
        return heat
    }
}

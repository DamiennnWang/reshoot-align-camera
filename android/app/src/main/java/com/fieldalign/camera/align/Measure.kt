package com.fieldalign.camera.align

import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 病害参数测量（与 python/measure.py 逐步对应）：
 *  比例尺：参考照片中二维码（已知打印尺寸）→ 墙面毫米；
 *  裂缝：光照归一化 → 黑帽 → 12 方向线段滤波 → 双阈值 → 细化 → 去毛刺 → 长度/宽度（面积法）；
 *  对比：复拍骨架中离参考裂缝 > 4px 的部分为新增；
 *  结论：按阈值规则生成文字。
 * 所有测量都在参考照片工作图坐标下（复拍已对齐到这里）。
 */
object Measure {
    const val DEFAULT_QR_MM = 40.0
    const val MIN_LEN_PX = 30.0
    const val MAX_MEAN_W_PX = 8.0
    const val GROW_TOL_PX = 4
    const val RES_LIMIT_PX = 2.0
    const val K_HI = 6.0; const val K_LO = 3.0
    val SCALES = intArrayOf(1, 3, 9)          // 原图检细裂缝；缩小 3、9 倍检宽裂缝
    const val MESH_DENSITY = 0.08             // 暗色粗糙区域里的“网状”假裂缝
    const val MIN_ELONG = 5.0                 // 长度至少是平均宽度的 5 倍
    const val PRESENT_RATIO = 0.5             // 另一张照片同处响应达到这一比例，就认为“本来就有”
    const val GROW_MIN_PX = 40.0              // 新增至少这么长才算“延长/新增”
    const val LINK_GAP = 14                   // 断开的裂缝段：端点沿走向 14px 内能接上另一段就连起来（弯曲、粗细变化处常断开）
    const val LINK_ANGLE = 30.0               // 连接方向与端点走向的夹角上限（度）
    const val LINK_MIN_SEG = 8.0              // 参与连接的段至少这么长（像素），避免把噪声点连成线
    const val LONG_WEAK_PX = 80.0             // 弱响应连通区外接框对角线 ≥ 80px 且很细（面积 ≤ 5×对角线）也算裂缝
    const val LONG_WEAK_FILL = 5.0
    private val BLUR_SIGMAS = doubleArrayOf(0.5, 0.8, 1.1, 1.5, 2.0, 2.6, 3.4)
    private const val SQRT2 = 1.4142135623730951

    // ------------------------------------------------------------ 比例尺
    /** 二维码内容里写了尺寸（如 “Z023;40mm”）就用它，否则 40mm。 */
    fun qrSizeMm(text: String?): Double {
        val m = Regex("(\\d+(?:\\.\\d+)?)\\s*mm", RegexOption.IGNORE_CASE).find(text ?: "") ?: return DEFAULT_QR_MM
        val v = m.groupValues[1].toDouble()
        return if (v in 5.0..500.0) v else DEFAULT_QR_MM
    }

    class Scale(val h: Homography, val qrMm: Double, val qrSidePx: Double) {
        /** 某点处 1 像素对应的毫米数（透视下各处不同）。 */
        fun mmPerPx(at: P): Double {
            val o = h.apply(at); val a = h.apply(P(at.x + 1, at.y)); val b = h.apply(P(at.x, at.y + 1))
            val ax = a.x - o.x; val ay = a.y - o.y; val bx = b.x - o.x; val by = b.y - o.y
            return sqrt(abs(ax * by - ay * bx))
        }
    }

    fun scaleFromQr(corners: List<P>?, qrMm: Double): Scale? {
        if (corners == null || corners.size != 4) return null
        val dst = listOf(P(0.0, 0.0), P(qrMm, 0.0), P(qrMm, qrMm), P(0.0, qrMm))
        val h = Homography.fromMat(Imgproc.getPerspectiveTransform(corners.toMat2f(), dst.toMat2f())) ?: return null
        val side = (0..3).sumOf { corners[it].dist(corners[(it + 1) % 4]) } / 4
        return Scale(h, qrMm, side)
    }

    // ------------------------------------------------------------ 裂缝
    data class Crack(
        val lengthPx: Double, val meanWPx: Double, val maxWPx: Double, val angleDeg: Double,
        val x: Int, val y: Int, val w: Int, val h: Int, val cx: Double, val cy: Double,
        val limited: Boolean, var growPx: Double = 0.0, var isNew: Boolean = false,
    )

    class CrackMap(val cracks: List<Crack>, val region: ByteArray, val skel: ByteArray, val w: Int, val h: Int,
                   val totalPx: Double, val maxWPx: Double,
                   /** 裂缝响应强度图（各尺度线段响应取最大）及其背景水平：比较两张照片“那里本来有没有”用。 */
                   val evid: Mat, val bgMed: Double, val bgMad: Double, val meanWPx: Double)

    class Change(val ref: CrackMap, val new: CrackMap, val growPx: Double, val grow: ByteArray, val lostPx: Double)

    private fun floats(m: Mat): FloatArray { val a = FloatArray((m.total() * m.channels()).toInt()); m.get(0, 0, a); return a }
    private fun bytes(m: Mat): ByteArray { val a = ByteArray((m.total() * m.channels()).toInt()); m.get(0, 0, a); return a }
    private fun ints(m: Mat): IntArray { val a = IntArray(m.total().toInt()); m.get(0, 0, a); return a }
    private fun matOf(b: ByteArray, w: Int, h: Int): Mat = Mat(h, w, CvType.CV_8U).also { it.put(0, 0, b) }

    /** 黑帽响应：细长暗线 → 亮。gray 为 8U，mask 非零为有效区。 */
    private fun response(gray: Mat, mask: Mat): Mat {
        val g = Mat(); gray.convertTo(g, CvType.CV_32F); Core.add(g, Scalar(1.0), g)
        val m = Mat(); Imgproc.threshold(mask, m, 0.0, 1.0, Imgproc.THRESH_BINARY); m.convertTo(m, CvType.CV_32F)
        val gm = Mat(); Core.multiply(g, m, gm)
        val num = Mat(); Imgproc.GaussianBlur(gm, num, Size(0.0, 0.0), 15.0)
        val den = Mat(); Imgproc.GaussianBlur(m, den, Size(0.0, 0.0), 15.0)
        Core.max(den, Scalar(1e-3), den)
        val bg = Mat(); Core.divide(num, den, bg); Core.max(bg, Scalar(1.0), bg)
        val r = Mat(); Core.divide(g, bg, r); Core.min(r, Scalar(2.0), r)
        val cl = Mat(); Imgproc.morphologyEx(r, cl, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(13.0, 13.0)))
        val bh = Mat(); Core.subtract(cl, r, bh)
        Imgproc.GaussianBlur(bh, bh, Size(0.0, 0.0), 0.8)
        Core.multiply(bh, m, bh)
        return bh
    }

    private val lineKernels: List<Mat> by lazy {
        val n = 15; val c = n / 2
        (0 until 12).map { i ->
            val a = PI * i / 12
            val k = Mat.zeros(n, n, CvType.CV_32F)
            val dx = cos(a) * c; val dy = sin(a) * c
            Imgproc.line(k, Point((c - dx).roundToInt().toDouble(), (c - dy).roundToInt().toDouble()),
                Point((c + dx).roundToInt().toDouble(), (c + dy).roundToInt().toDouble()), Scalar(1.0), 1, Imgproc.LINE_AA)
            Core.divide(k, Scalar(Core.sumElems(k).`val`[0]), k); k
        }
    }

    /** 12 方向、长 15px 的线段平均取最大：裂缝保持高响应，圆形斑点被摊薄。 */
    private fun lineResponse(bh: Mat): Mat {
        val out = Mat.zeros(bh.size(), CvType.CV_32F); val t = Mat()
        for (k in lineKernels) { Imgproc.filter2D(bh, t, -1, k, Point(-1.0, -1.0), 0.0, Core.BORDER_CONSTANT); Core.max(out, t, out) }
        return out
    }

    private fun medMad(v: FloatArray, mk: ByteArray): Pair<Double, Double> {
        val a = FloatArray(v.size / 7 + 1); var n = 0
        var i = 0
        while (i < v.size) { if (mk[i].toInt() != 0) a[n++] = v[i]; i += 7 }
        if (n < 100) return Double.NaN to 0.0
        a.sort(0, n)
        val med = a[n / 2].toDouble()
        for (k in 0 until n) a[k] = abs(a[k] - med.toFloat())
        a.sort(0, n)
        return med to a[n / 2] * 1.4826 + 1e-4
    }

    /** Zhang–Suen 细化（0/1 字节数组，原地）。 */
    fun thin(b: ByteArray, w: Int, h: Int) {
        val rm = IntArray(w * h); var nrm: Int
        do {
            var changed = false
            for (step in 0..1) {
                nrm = 0
                for (y in 1 until h - 1) {
                    val r = y * w
                    for (x in 1 until w - 1) {
                        val i = r + x
                        if (b[i].toInt() == 0) continue
                        val p2 = b[i - w].toInt(); val p3 = b[i - w + 1].toInt(); val p4 = b[i + 1].toInt(); val p5 = b[i + w + 1].toInt()
                        val p6 = b[i + w].toInt(); val p7 = b[i + w - 1].toInt(); val p8 = b[i - 1].toInt(); val p9 = b[i - w - 1].toInt()
                        val bn = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                        if (bn < 2 || bn > 6) continue
                        var a = 0
                        if (p2 == 0 && p3 == 1) a++; if (p3 == 0 && p4 == 1) a++; if (p4 == 0 && p5 == 1) a++; if (p5 == 0 && p6 == 1) a++
                        if (p6 == 0 && p7 == 1) a++; if (p7 == 0 && p8 == 1) a++; if (p8 == 0 && p9 == 1) a++; if (p9 == 0 && p2 == 1) a++
                        if (a != 1) continue
                        if (step == 0) { if (p2 * p4 * p6 != 0 || p4 * p6 * p8 != 0) continue }
                        else { if (p2 * p4 * p8 != 0 || p2 * p6 * p8 != 0) continue }
                        rm[nrm++] = i
                    }
                }
                for (k in 0 until nrm) b[rm[k]] = 0
                if (nrm > 0) changed = true
            }
        } while (changed)
    }

    private fun nb(b: ByteArray, w: Int, h: Int, i: Int): Int {
        val x = i % w; val y = i / w; var n = 0
        for (dy in -1..1) for (dx in -1..1) {
            if (dx == 0 && dy == 0) continue
            val xx = x + dx; val yy = y + dy
            if (xx in 0 until w && yy in 0 until h && b[yy * w + xx].toInt() != 0) n++
        }
        return n
    }

    /** 去掉短于 n 像素的毛刺，再把主干两端长回来。 */
    fun prune(sk: ByteArray, w: Int, h: Int, n: Int = 8): ByteArray {
        val s = sk.copyOf()
        var cnt = 0
        for (b in sk) if (b.toInt() != 0) cnt++
        val idx = IntArray(cnt); var k = 0
        for (i in sk.indices) if (sk[i].toInt() != 0) idx[k++] = i
        val buf = IntArray(cnt)
        repeat(n) {
            var m = 0
            for (i in idx) if (s[i].toInt() != 0 && nb(s, w, h, i) <= 1) buf[m++] = i
            for (j in 0 until m) s[buf[j]] = 0
        }
        repeat(n) {
            var m = 0
            for (i in idx) if (s[i].toInt() != 0 && nb(s, w, h, i) == 1) buf[m++] = i
            for (j in 0 until m) {
                val i = buf[j]; val x = i % w; val y = i / w
                for (dy in -1..1) for (dx in -1..1) {
                    val xx = x + dx; val yy = y + dy
                    if (xx in 0 until w && yy in 0 until h) { val q = yy * w + xx; if (sk[q].toInt() != 0 && s[q].toInt() == 0) s[q] = 1 }
                }
            }
        }
        return s
    }

    /** 骨架长度（横竖 1、斜向 √2，去掉三角形中多余的斜连接）。labels 非空时按标签分别累计。 */
    fun linkLength(s: ByteArray, w: Int, h: Int, labels: IntArray? = null, n: Int = 0, roi: Rect? = null): DoubleArray {
        val out = DoubleArray(if (labels == null) 1 else n)
        fun on(x: Int, y: Int) = x in 0 until w && y in 0 until h && s[y * w + x].toInt() != 0
        val x0 = roi?.x ?: 0; val y0 = roi?.y ?: 0
        val x1 = roi?.let { it.x + it.width } ?: w; val y1 = roi?.let { it.y + it.height } ?: h
        fun hz(x: Int, y: Int) = on(x, y) && on(x + 1, y)
        fun vt(x: Int, y: Int) = on(x, y) && on(x, y + 1)
        for (y in y0 until y1) for (x in x0 until x1) {
            if (!on(x, y)) continue
            val l = labels?.get(y * w + x) ?: 0
            if (hz(x, y)) out[l] += 1.0
            if (vt(x, y)) out[l] += 1.0
            // 斜向 (x,y)-(x+1,y+1)：若已有横竖连接构成三角形则不计
            if (on(x + 1, y + 1) && !(hz(x, y) && vt(x + 1, y)) && !(vt(x, y) && hz(x, y + 1))) out[l] += SQRT2
            // 斜向 (x,y)-(x-1,y+1)
            val xp = x - 1
            if (on(xp, y + 1) && !(hz(xp, y) && vt(xp, y)) && !(vt(x, y) && hz(xp, y + 1))) out[l] += SQRT2
        }
        return out
    }

    private val N8 = arrayOf(intArrayOf(-1, -1), intArrayOf(-1, 0), intArrayOf(-1, 1), intArrayOf(0, -1),
        intArrayOf(0, 1), intArrayOf(1, -1), intArrayOf(1, 0), intArrayOf(1, 1))   // (dy, dx)

    /** 从端点沿骨架往回走 steps 步，返回端点处指向外侧的单位走向 (dx, dy)。 */
    private fun endTangent(s: ByteArray, w: Int, h: Int, x: Int, y: Int, steps: Int = 10): DoubleArray? {
        var px = -1; var py = -1; var cx = x; var cy = y
        for (k in 0 until steps) {
            var nx = -1; var ny = -1
            for (d in N8) {
                val yy = cy + d[0]; val xx = cx + d[1]
                if (yy in 0 until h && xx in 0 until w && s[yy * w + xx].toInt() != 0 && !(yy == py && xx == px) && !(yy == y && xx == x)) { nx = xx; ny = yy; break }
            }
            if (nx < 0) break
            px = cx; py = cy; cx = nx; cy = ny
        }
        val vx = (x - cx).toDouble(); val vy = (y - cy).toDouble(); val n = hypot(vx, vy)
        return if (n >= 3) doubleArrayOf(vx / n, vy / n) else null
    }

    /**
     * 把同一条裂缝断开的几段连起来（与 measure.link_gaps 相同）：每个端点沿自己的走向往前找 LINK_GAP 内的另一段骨架，
     * 夹角不超过 LINK_ANGLE、连线上的线段响应高于 tGap 才连。直接在 lo（0/1）上画连线。
     */
    private fun linkGaps(lo: Mat, lnA: FloatArray, tGap: Double, w: Int, h: Int) {
        val sk = bytes(lo); for (i in sk.indices) sk[i] = if (sk[i].toInt() != 0) 1 else 0
        thin(sk, w, h); val s = prune(sk, w, h, 4)
        val labM = Mat(); val n = Imgproc.connectedComponents(matOf(s, w, h), labM, 8, CvType.CV_32S)
        if (n <= 2) return
        val lab = ints(labM); val lens = linkLength(s, w, h, lab, n)
        val g = LINK_GAP; val cosA = cos(Math.toRadians(LINK_ANGLE))
        for (i in s.indices) {
            if (s[i].toInt() == 0 || nb(s, w, h, i) != 1) continue
            val li = lab[i]; if (lens[li] < LINK_MIN_SEG) continue
            val x = i % w; val y = i / w
            val t = endTangent(s, w, h, x, y) ?: continue
            var best = Double.MAX_VALUE; var bx = -1; var by = -1; var bd = 0.0
            for (yy in max(y - g, 0)..min(y + g, h - 1)) for (xx in max(x - g, 0)..min(x + g, w - 1)) {
                val lq = lab[yy * w + xx]
                if (lq == 0 || lq == li || lens[lq] < LINK_MIN_SEG) continue
                val vx = (xx - x).toDouble(); val vy = (yy - y).toDouble(); val d = hypot(vx, vy)
                if (d > g || d < 1e-6) continue
                val cs = (vx * t[0] + vy * t[1]) / d
                if (cs < cosA) continue
                val sc = d * (2 - cs); if (sc < best) { best = sc; bx = xx; by = yy; bd = d }
            }
            if (bx < 0) continue
            val k = max(bd.toInt(), 2); var sum = 0.0
            for (j in 0..k) {
                val qx = Math.round(x + (bx - x) * j / k.toDouble()).toInt(); val qy = Math.round(y + (by - y) * j / k.toDouble()).toInt()
                sum += lnA[qy * w + qx]
            }
            if (sum / (k + 1) < tGap) continue
            Imgproc.line(lo, Point(x.toDouble(), y.toDouble()), Point(bx.toDouble(), by.toDouble()), Scalar(1.0), 1)
        }
    }

    private class Cand(var srcW: Int = 0, val L: Double, val meanW: Double, val maxW: Double, val ang: Double, val pts: IntArray,
                       val x0: Int, val y0: Int, val bw: Int, val bh: Int, val reg: ByteArray)

    /** 单一尺度检测。返回候选（坐标为该尺度像素，pts 为骨架像素下标 y*w+x）和线段响应图。 */
    @Volatile private var detTiming = ""
    private fun detectScale(gray: Mat, mask: Mat, minLen: Double): Pair<List<Cand>, Mat> {
        val w = gray.cols(); val h = gray.rows()
        val q0 = System.currentTimeMillis()
        val bh = response(gray, mask)
        val q1 = System.currentTimeMillis()
        val ln = lineResponse(bh)
        val q2 = System.currentTimeMillis()
        val mk = bytes(mask)
        val lnA = floats(ln); val bhA = floats(bh)
        val (med, mad) = medMad(lnA, mk)
        if (med.isNaN()) return emptyList<Cand>() to ln
        val tHi = max(med + K_HI * mad, 0.20); val tLo = max(med + K_LO * mad, 0.10)
        val lo = ByteArray(w * h) { if (lnA[it] > tLo && bhA[it] > tLo) 1 else 0 }
        val loM = matOf(lo, w, h)
        Imgproc.morphologyEx(loM, loM, Imgproc.MORPH_CLOSE, Mat.ones(3, 3, CvType.CV_8U))
        if (minLen == MIN_LEN_PX) linkGaps(loM, lnA, med + mad, w, h)   // 只在原图尺度连接（细裂缝才会断段；粗尺度连线容易把纹理连成假裂缝）
        val labM = Mat(); val stM = Mat(); val ctM = Mat()
        val n = Imgproc.connectedComponentsWithStats(loM, labM, stM, ctM, 8, CvType.CV_32S)
        val lab = ints(labM)
        val strong = BooleanArray(n)
        for (i in lab.indices) if (lnA[i] > tHi && lab[i] > 0) strong[lab[i]] = true
        // 又长又细的弱响应也算：淡裂缝整条都只略高于弱阈值，但纹理噪声连不成这么长的细线
        val stA = IntArray(5)
        for (c in 1 until n) {
            stM.get(c, 0, stA)
            val diag = hypot(stA[2].toDouble(), stA[3].toDouble())
            if (diag >= LONG_WEAK_PX && stA[4] <= LONG_WEAK_FILL * diag) strong[c] = true
        }
        val region = ByteArray(w * h) { if (strong[lab[it]]) 1 else 0 }
        val th = region.copyOf(); thin(th, w, h)
        val q3 = System.currentTimeMillis()
        val skel = prune(th, w, h, 8)
        val q4 = System.currentTimeMillis()
        if (w > 1000) detTiming = "响应 ${q1 - q0} 线段 ${q2 - q1} 阈值+细化 ${q3 - q2} 去毛刺 ${q4 - q3}"
        val skM = matOf(skel, w, h)
        val sLabM = Mat(); val n2 = Imgproc.connectedComponents(skM, sLabM, 8, CvType.CV_32S)
        val sLab = ints(sLabM)
        val lens = linkLength(skel, w, h, sLab, n2)
        val rLabM = Mat(); Imgproc.connectedComponents(matOf(region, w, h), rLabM, 8, CvType.CV_32S)
        val rLab = ints(rLabM)
        // 面积法：只积分高出背景的响应，且只取骨架两侧 5px 以内
        val (base, _) = medMad(bhA, mk)
        val near = Mat(); Imgproc.dilate(skM, near, Mat.ones(11, 11, CvType.CV_8U))
        val nearA = bytes(near)
        val pts = Array(n2) { ArrayList<Int>() }
        for (i in sLab.indices) if (skel[i].toInt() != 0) pts[sLab[i]].add(i)
        val out = ArrayList<Cand>()
        for (c in 1 until n2) {
            val L = lens[c]; val pp = pts[c]
            if (L < minLen || pp.isEmpty()) continue
            var minX = w; var minY = h; var maxX = 0; var maxY = 0
            for (i in pp) { val x = i % w; val y = i / w; minX = min(minX, x); minY = min(minY, y); maxX = max(maxX, x); maxY = max(maxY, y) }
            val x1 = maxX + 1; val y1 = maxY + 1
            // 网状假裂缝：骨架又密又曲折
            val diag = sqrt(((x1 - minX) * (x1 - minX) + (y1 - minY) * (y1 - minY)).toDouble())
            if (L / max((x1 - minX) * (y1 - minY), 1) > MESH_DENSITY && L > 2.5 * diag) continue
            val rid = pp.groupingBy { rLab[it] }.eachCount().maxByOrNull { it.value }!!.key
            val bhv = FloatArray(pp.size) { bhA[pp[it]] }; bhv.sort()
            val depth = max(pctl(bhv, 90.0) - base, 0.25)
            val pad = 12
            val X0 = max(minX - pad, 0); val Y0 = max(minY - pad, 0); val X1 = min(x1 + pad, w); val Y1 = min(y1 + pad, h)
            val bw = X1 - X0; val bhh = Y1 - Y0
            var sumR = 0.0
            val rr = FloatArray(bw * bhh); val ss = FloatArray(bw * bhh); val reg = ByteArray(bw * bhh)
            for (y in Y0 until Y1) for (x in X0 until X1) {
                val i = y * w + x
                if (rLab[i] == rid) {
                    val k = (y - Y0) * bw + (x - X0); reg[k] = 1
                    if (region[i].toInt() != 0 && nearA[i].toInt() != 0) { val v = max(bhA[i] - base.toFloat(), 0f); rr[k] = v; sumR += v }
                }
            }
            val meanW = sumR / (depth * L)
            if (meanW > MAX_MEAN_W_PX || L < MIN_ELONG * meanW) continue
            val per = (L / pp.size).toFloat()
            for (i in pp) ss[(i / w - Y0) * bw + (i % w - X0)] = per
            val rM = Mat(bhh, bw, CvType.CV_32F).also { it.put(0, 0, rr) }
            val sM = Mat(bhh, bw, CvType.CV_32F).also { it.put(0, 0, ss) }
            Imgproc.boxFilter(rM, rM, -1, Size(21.0, 21.0), Point(-1.0, -1.0), false)
            Imgproc.boxFilter(sM, sM, -1, Size(21.0, 21.0), Point(-1.0, -1.0), false)
            val sr = floats(rM); val sss = floats(sM)
            val loc = FloatArray(pp.size) { j -> val i = pp[j]; val k = (i / w - Y0) * bw + (i % w - X0); (sr[k] / max(depth * sss[k], 1e-6)).toFloat() }
            loc.sort()
            val maxW = pctl(loc, 90.0)
            // 走向：骨架点主方向（图像 y 向下，换成逆时针为正）
            val mx = pp.sumOf { it % w } / pp.size.toDouble(); val my = pp.sumOf { it / w } / pp.size.toDouble()
            var sxx = 0.0; var syy = 0.0; var sxy = 0.0
            for (i in pp) { val dx = i % w - mx; val dy = i / w - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy }
            val ang = ((-Math.toDegrees(0.5 * atan2(2 * sxy, sxx - syy))) % 180 + 180) % 180
            out.add(Cand(w, L, meanW, max(maxW, meanW), ang, pp.toIntArray(), X0, Y0, bw, bhh, reg))
        }
        return out to ln
    }

    /** numpy 式线性插值分位数（a 已升序）。 */
    private fun pctl(a: FloatArray, p: Double): Double {
        if (a.isEmpty()) return 0.0
        val pos = (a.size - 1) * p / 100.0; val i = pos.toInt(); val f = pos - i
        return if (i + 1 < a.size) a[i] * (1 - f) + a[i + 1] * f else a[i].toDouble()
    }

    /** 多尺度检测并合并：重叠的候选取“在自己尺度下宽度最接近 4 像素”的那个（测得最准）。 */
    private fun detect(gray: Mat, mask: Mat): CrackMap {
        val w = gray.cols(); val h = gray.rows()
        val cands = ArrayList<Triple<Double, Int, Cand>>()
        val evid = Mat()
        for (f in SCALES) {
            if (f > 1 && min(h, w) / f < 100) break
            val g: Mat; val m: Mat
            if (f == 1) { g = gray; m = mask } else {
                g = Mat(); Imgproc.resize(gray, g, Size((w / f).toDouble(), (h / f).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
                m = Mat(); Imgproc.resize(mask, m, Size((w / f).toDouble(), (h / f).toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
            }
            val (found, ln) = detectScale(g, m, if (f == 1) MIN_LEN_PX else 20.0)
            if (f == 1) ln.copyTo(evid) else {
                val up = Mat(); Imgproc.resize(ln, up, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR); Core.max(evid, up, evid)
            }
            for (c in found) cands.add(Triple(abs(kotlin.math.ln(max(c.meanW, 0.5) / 4.0)), f, c))
        }
        cands.sortWith(compareBy({ it.first }, { it.second }))
        val region = Mat.zeros(h, w, CvType.CV_8U); val skel = Mat.zeros(h, w, CvType.CV_8U)
        val cracks = ArrayList<Crack>()
        for ((_, f, c) in cands) {
            val X0 = c.x0 * f; val Y0 = c.y0 * f
            val X1 = min(X0 + c.bw * f, w); val Y1 = min(Y0 + c.bh * f, h)
            val rw = X1 - X0; val rh = Y1 - Y0
            val skS = ByteArray(c.bw * c.bh)
            for (i in c.pts) skS[(i / c.srcW - c.y0) * c.bw + (i % c.srcW - c.x0)] = 1
            val skB: ByteArray; val regM: Mat
            if (f == 1) { skB = skS; regM = matOf(c.reg, c.bw, c.bh) } else {
                val up = Mat(); Imgproc.resize(matOf(skS, c.bw, c.bh), up, Size((c.bw * f).toDouble(), (c.bh * f).toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
                skB = bytes(up.submat(0, rh, 0, rw).clone()); thin(skB, rw, rh)
                val ur = Mat(); Imgproc.resize(matOf(c.reg, c.bw, c.bh), ur, Size((c.bw * f).toDouble(), (c.bh * f).toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
                regM = ur.submat(0, rh, 0, rw).clone()
            }
            var nSk = 0; var minX = rw; var minY = rh; var maxX = 0; var maxY = 0; var sx = 0.0; var sy = 0.0
            for (i in skB.indices) if (skB[i].toInt() != 0) {
                nSk++; val x = i % rw; val y = i / rw
                minX = min(minX, x); minY = min(minY, y); maxX = max(maxX, x); maxY = max(maxY, y); sx += x; sy += y
            }
            if (nSk == 0) continue
            // 已被接受的裂缝覆盖了一半以上：重复，跳过
            val roi = Rect(X0, Y0, rw, rh)
            val kk = 2 * f + 3
            val nearM = Mat(); Imgproc.dilate(region.submat(roi), nearM, Mat.ones(kk, kk, CvType.CV_8U))
            val nearB = bytes(nearM)
            var ov = 0
            for (i in skB.indices) if (skB[i].toInt() != 0 && nearB[i].toInt() != 0) ov++
            if (ov > 0.5 * nSk) continue
            cracks.add(Crack(c.L * f, c.meanW * f, c.maxW * f, c.ang, X0 + minX, Y0 + minY, maxX - minX + 1, maxY - minY + 1,
                X0 + sx / nSk, Y0 + sy / nSk, f == 1 && c.meanW < RES_LIMIT_PX))
            Core.bitwise_or(region.submat(roi), regM, region.submat(roi))
            Core.bitwise_or(skel.submat(roi), matOf(skB, rw, rh), skel.submat(roi))
        }
        cracks.sortByDescending { it.lengthPx }
        val (bmed, bmad) = medMad(floats(evid), bytes(mask))
        val tot = cracks.sumOf { it.lengthPx }
        return CrackMap(cracks, bytes(region), bytes(skel), w, h, tot, cracks.maxOfOrNull { it.maxWPx } ?: 0.0,
            evid, if (bmed.isNaN()) 0.0 else bmed, if (bmed.isNaN()) 0.0 else bmad - 1e-4,
            if (tot > 0) cracks.sumOf { it.meanWPx * it.lengthPx } / tot else 0.0)
    }

    private fun localRatio(gray: Mat, mask: Mat, sigma: Double = 15.0): Mat {
        val g = Mat(); gray.convertTo(g, CvType.CV_32F); Core.add(g, Scalar(1.0), g)
        val m = Mat(); Imgproc.threshold(mask, m, 0.0, 1.0, Imgproc.THRESH_BINARY); m.convertTo(m, CvType.CV_32F)
        val gm = Mat(); Core.multiply(g, m, gm)
        val num = Mat(); Imgproc.GaussianBlur(gm, num, Size(0.0, 0.0), sigma)
        val den = Mat(); Imgproc.GaussianBlur(m, den, Size(0.0, 0.0), sigma); Core.max(den, Scalar(1e-3), den)
        val bg = Mat(); Core.divide(num, den, bg); Core.max(bg, Scalar(1.0), bg)
        val r = Mat(); Core.divide(g, bg, r); Core.min(r, Scalar(2.0), r)
        return r
    }

    /**
     * 两张照片清晰度不同时，把较清晰的一张模糊到与另一张一致（原地修改）。否则清晰那张噪声大、阈值高，
     * 同一条淡裂缝一张检出、一张检不出，会被误判成新增或消失。返回 σ：>0 表示 a 被模糊，<0 表示 b。
     */
    fun matchSharpness(a: Mat, b: Mat, mask: Mat): Double {
        // 在半分辨率上做（局部均值的 σ 也减半），结果与全分辨率几乎相同、快 4 倍
        val ha = Mat(); Imgproc.resize(a, ha, Size(), 0.5, 0.5, Imgproc.INTER_AREA)
        val hb = Mat(); Imgproc.resize(b, hb, Size(), 0.5, 0.5, Imgproc.INTER_AREA)
        val sm = Mat(); Imgproc.resize(mask, sm, ha.size(), 0.0, 0.0, Imgproc.INTER_NEAREST)
        val ra = localRatio(ha, sm, 7.5); val rb = localRatio(hb, sm, 7.5)
        if (Core.countNonZero(sm) < 100) return 0.0
        val d = Mat(); val bl = Mat()
        fun err(x: Mat, y: Mat): Double { Core.absdiff(x, y, d); return Core.mean(d, sm).`val`[0] }
        var best = err(ra, rb); var bestS = 0.0
        for ((sign, src, other) in listOf(Triple(1, ra, rb), Triple(-1, rb, ra))) for (sg in BLUR_SIGMAS) {
            Imgproc.GaussianBlur(src, bl, Size(0.0, 0.0), sg / 2)
            val e = err(bl, other)
            if (e < best * 0.97) { best = e; bestS = sign * sg }
        }
        if (bestS > 0) Imgproc.GaussianBlur(a, a, Size(0.0, 0.0), bestS)
        else if (bestS < 0) Imgproc.GaussianBlur(b, b, Size(0.0, 0.0), -bestS)
        return bestS
    }

    private fun dropShort(b: ByteArray, w: Int, h: Int, minPx: Int): ByteArray {
        val lab = Mat(); val stats = Mat(); val cen = Mat()
        val n = Imgproc.connectedComponentsWithStats(matOf(b, w, h), lab, stats, cen, 8, CvType.CV_32S)
        val keep = BooleanArray(n) { it > 0 && stats.get(it, Imgproc.CC_STAT_AREA)[0] >= minPx }
        val la = ints(lab)
        return ByteArray(w * h) { if (keep[la[it]]) 1 else 0 }
    }

    /** 参考与复拍（已对齐）比较裂缝。gray 为 8U 灰度（会被原地模糊以匹配清晰度）；mask 非零为比较区。 */
    @Volatile var lastTiming = ""
    fun compare(refGray: Mat, newGray: Mat, mask: Mat, ignore: Mat? = null): Change {
        val t0 = System.currentTimeMillis()
        matchSharpness(refGray, newGray, mask)
        val t1 = System.currentTimeMillis()
        val ig = ignore?.let { bytes(it) }
        var r = detect(refGray, mask); val t2 = System.currentTimeMillis(); var n = detect(newGray, mask)
        if (ig != null) { r = dropInside(r, ig); n = dropInside(n, ig) }
        val t3 = System.currentTimeMillis()
        val w = r.w; val h = r.h
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(2.0 * GROW_TOL_PX + 1, 2.0 * GROW_TOL_PX + 1))
        val nr = Mat(); Imgproc.dilate(matOf(r.region, w, h), nr, k); val nearRef = bytes(nr)
        val nn = Mat(); Imgproc.dilate(matOf(n.region, w, h), nn, k); val nearNew = bytes(nn)
        // 新增 = 这次有、上次那里“确实没有”：上次没检出，且上次该处高出背景的响应不到这次的一半。
        val ke = Mat.ones(2 * GROW_TOL_PX + 1, 2 * GROW_TOL_PX + 1, CvType.CV_8U); val k3 = Mat.ones(3, 3, CvType.CV_8U)
        fun dil(m: Mat, kk: Mat): FloatArray { val o = Mat(); Imgproc.dilate(m, o, kk); return floats(o) }
        val rNear = dil(r.evid, ke); val nNear = dil(n.evid, ke); val rAt = dil(r.evid, k3); val nAt = dil(n.evid, k3)
        val rOff = (r.bgMed + 2.5 * r.bgMad).toFloat(); val nOff = (n.bgMed + 2.5 * n.bgMad).toFloat()
        // 比较区边缘 8px 内不算：裂缝在一张照片里刚好被边界截断
        val inM = Mat(); Imgproc.erode(mask, inM, Mat.ones(17, 17, CvType.CV_8U), Point(-1.0, -1.0), 1, Core.BORDER_CONSTANT, Scalar(0.0))
        val inner = bytes(inM)
        if (ig != null) for (i in inner.indices) if (ig[i].toInt() != 0) inner[i] = 0
        val pr = PRESENT_RATIO.toFloat()
        val grow0 = ByteArray(w * h) {
            if (n.skel[it].toInt() != 0 && nearRef[it].toInt() == 0 && inner[it].toInt() != 0 &&
                rNear[it] - rOff < pr * max(nAt[it] - n.bgMed.toFloat(), 0f)) 1 else 0 }
        val lost0 = ByteArray(w * h) {
            if (r.skel[it].toInt() != 0 && nearNew[it].toInt() == 0 && inner[it].toInt() != 0 &&
                nNear[it] - nOff < pr * max(rAt[it] - r.bgMed.toFloat(), 0f)) 1 else 0 }
        val grow = dropShort(grow0, w, h, 6); val lost = dropShort(lost0, w, h, 6)
        for (c in n.cracks) {
            val roi = Rect(c.x, c.y, c.w, c.h)
            c.growPx = linkLength(grow, w, h, roi = roi)[0]
            c.isNew = c.growPx > 0.8 * linkLength(n.skel, w, h, roi = roi)[0] && c.growPx >= GROW_MIN_PX
        }
        lastTiming = "清晰度匹配 ${t1 - t0} | 检测 ${t2 - t1}+${t3 - t2} | 对比 ${System.currentTimeMillis() - t3} ms；" + detTiming
        return Change(r, n, linkLength(grow, w, h)[0], grow, linkLength(lost, w, h)[0])
    }

    /** 去掉大半落在 ignore 区域里的裂缝（剥落、掉块的断口边缘看起来像裂缝）。与 measure.drop_inside 相同。 */
    private fun dropInside(cm: CrackMap, ig: ByteArray): CrackMap {
        val w = cm.w; val h = cm.h
        val lm = Mat(); val n = Imgproc.connectedComponents(matOf(cm.skel, w, h), lm, 8, CvType.CV_32S)
        val lab = ints(lm)
        val tot = IntArray(n); val inside = IntArray(n)
        for (i in lab.indices) { tot[lab[i]]++; if (ig[i].toInt() != 0) inside[lab[i]]++ }
        val bad = BooleanArray(n) { it > 0 && inside[it] > 0.5 * tot[it] }
        if (bad.none { it }) return cm
        val skel = cm.skel.copyOf(); for (i in skel.indices) if (bad[lab[i]]) skel[i] = 0
        val rm = Mat(); val nr = Imgproc.connectedComponents(matOf(cm.region, w, h), rm, 8, CvType.CV_32S)
        val rl = ints(rm)
        val badr = BooleanArray(nr); val keepr = BooleanArray(nr)
        for (i in rl.indices) { if (bad[lab[i]]) badr[rl[i]] = true; if (skel[i].toInt() != 0) keepr[rl[i]] = true }
        badr[0] = false
        val region = cm.region.copyOf(); for (i in region.indices) if (badr[rl[i]] && !keepr[rl[i]]) region[i] = 0
        fun cnt(b: ByteArray, c: Crack): Int { var k = 0; for (y in c.y until c.y + c.h) for (x in c.x until c.x + c.w) if (b[y * w + x].toInt() != 0) k++; return k }
        val cracks = cm.cracks.filter { cnt(skel, it) > 0.5 * cnt(cm.skel, it) }
        val tot2 = cracks.sumOf { it.lengthPx }
        return CrackMap(cracks, region, skel, w, h, tot2, cracks.maxOfOrNull { it.maxWPx } ?: 0.0, cm.evid, cm.bgMed, cm.bgMad,
            if (tot2 > 0) cracks.sumOf { it.meanWPx * it.lengthPx } / tot2 else 0.0)
    }

    // ------------------------------------------------------------ 结论
    /** 判定阈值（可按工程要求调整）。 */
    data class Rules(
        val growMinPx: Double = GROW_MIN_PX, val growMinRatio: Double = 0.05,
        val widenMinPx: Double = 1.5, val widenMinRatio: Double = 0.3, val widenMeanRatio: Double = 0.15,
        val widenMeanPx: Double = 0.8, val widenMeanBig: Double = 0.4,
        val attentionMm: Double = 0.3,
    )

    fun fmtLen(px: Double, mmpp: Double?) = if (mmpp != null) "%.1f mm".format(px * mmpp) else "%.0f px".format(px)
    fun fmtW(px: Double, mmpp: Double?) = if (mmpp != null) "%.2f mm".format(px * mmpp) else "%.1f px".format(px)

    /** 生成结论所需的统计量（可由检测结果得到，也可由数据库里存的明细 JSON 还原）。 */
    data class Stats(val refCount: Int, val refTotal: Double, val refMaxW: Double, val newCount: Int, val newTotal: Double,
                     val newMaxW: Double, val grow: Double, val lost: Double, val newCrackCount: Int, val newCrackLen: Double,
                     val allLimited: Boolean, val refMeanW: Double = 0.0, val newMeanW: Double = 0.0)

    /** 算作“新增裂缝”：全新且够长；很细（接近分辨率极限）的要两倍长——若隐若现的分叉末梢一张检出、一张没检出很常见。 */
    private fun counted(c: Crack) = c.isNew && c.growPx >= GROW_MIN_PX * (if (c.limited) 2 else 1)

    fun stats(ch: Change) = Stats(ch.ref.cracks.size, ch.ref.totalPx, ch.ref.maxWPx, ch.new.cracks.size, ch.new.totalPx, ch.new.maxWPx,
        ch.growPx, ch.lostPx, ch.new.cracks.count { counted(it) }, ch.new.cracks.filter { counted(it) }.sumOf { it.lengthPx },
        ch.new.cracks.isNotEmpty() && ch.new.cracks.all { it.limited }, ch.ref.meanWPx, ch.new.meanWPx)

    fun stats(json: String): Stats? = runCatching {
        fun cnt(k: JSONObject) = k.getBoolean("new") && k.optDouble("grow_px", GROW_MIN_PX) >= GROW_MIN_PX * (if (k.optBoolean("limited")) 2 else 1)
        val o = JSONObject(json); val a = o.getJSONArray("cracks")
        val ks = (0 until a.length()).map { a.getJSONObject(it) }
        Stats(o.getInt("ref_count"), o.getDouble("ref_total_px"), o.getDouble("ref_max_w_px"), ks.size,
            ks.sumOf { it.getDouble("len_px") }, ks.maxOfOrNull { it.getDouble("max_w_px") } ?: 0.0,
            ks.sumOf { it.getDouble("grow_px") }.let { g -> if (o.has("grow_px")) o.getDouble("grow_px") else g },
            o.optDouble("lost_px", 0.0), ks.count { cnt(it) }, ks.filter { cnt(it) }.sumOf { it.getDouble("len_px") },
            ks.isNotEmpty() && ks.all { it.getBoolean("limited") }, o.optDouble("ref_mean_w_px", 0.0),
            ks.sumOf { it.getDouble("len_px") }.let { t -> if (t > 0) ks.sumOf { it.getDouble("mean_w_px") * it.getDouble("len_px") } / t else 0.0 })
    }.getOrNull()

    fun conclude(ch: Change, mmpp: Double?, lightRegions: Int = 0, lightRatio: Double = 0.0, rules: Rules = Rules()) =
        conclude(stats(ch), mmpp, lightRegions, lightRatio, rules)

    /** 返回 (等级, 文字结论)。等级：无裂缝 / 稳定 / 延长 / 变宽 / 延长且变宽 / 新增。 */
    fun conclude(st: Stats, mmpp: Double?, lightRegions: Int = 0, lightRatio: Double = 0.0, rules: Rules = Rules()): Pair<String, String> {
        val sb = StringBuilder()
        val longer = st.grow >= max(rules.growMinPx, rules.growMinRatio * st.refTotal)
        // 变宽：最大宽度和整体平均宽度都明显增加（最大宽度是局部估计，单看它容易受清晰度影响）
        // 或按长度加权的平均宽度大幅增加（更稳：细裂缝被拆成几段时某一段的局部估计可能偏大）
        val wider = st.refCount > 0 && st.newCount > 0 && (
            ((st.newMaxW - st.refMaxW) >= max(rules.widenMinPx, rules.widenMinRatio * st.refMaxW) &&
                st.newMeanW >= (1 + rules.widenMeanRatio) * st.refMeanW) ||
            (st.newMeanW - st.refMeanW) >= max(rules.widenMeanPx, rules.widenMeanBig * st.refMeanW))
        val allLimited = st.allLimited
        val level: String
        if (st.newCount == 0 && st.refCount == 0) {
            level = "无裂缝"; sb.append("对比区域内未检出裂缝。")
        } else {
            sb.append("检出裂缝 ${st.newCount} 条，总长 ${fmtLen(st.newTotal, mmpp)}（上次 ${st.refCount} 条、${fmtLen(st.refTotal, mmpp)}），" +
                "最大宽度 ${if (allLimited) "≤" else ""}${fmtW(st.newMaxW, mmpp)}（上次 ${fmtW(st.refMaxW, mmpp)}）。")
            val changes = ArrayList<String>()
            if (st.newCrackCount > 0) changes.add("新增裂缝 ${st.newCrackCount} 条（共 ${fmtLen(st.newCrackLen, mmpp)}）")
            if (longer) changes.add("裂缝延长 ${fmtLen(st.grow, mmpp)}" + if (st.refTotal > 0) "，+%.0f%%".format(st.grow / st.refTotal * 100) else "")
            if (wider && !allLimited) changes.add("最大宽度增加 ${fmtW(st.newMaxW - st.refMaxW, mmpp)}")
            level = when {
                changes.isEmpty() -> "稳定"
                st.newCrackCount > 0 && !(longer && st.refTotal > 0) -> "新增"
                longer && wider && !allLimited -> "延长且变宽"
                longer -> "延长"
                wider && !allLimited -> "变宽"
                else -> "新增"
            }
            if (changes.isEmpty()) sb.append("与上次相比未见明显发展。") else sb.append("变化：" + changes.joinToString("；") + "。")
            if (allLimited) sb.append("注意：裂缝宽度接近照片分辨率极限（约 ${fmtW(RES_LIMIT_PX, mmpp)}），实际可能更细；需要测细裂缝宽度时请靠近拍摄。")
            if (st.lost >= rules.growMinPx) sb.append("有 ${fmtLen(st.lost, mmpp)} 上次的裂缝这次未检出（可能已修补、被遮挡或拍摄不清）。")
            if (mmpp != null && st.newMaxW * mmpp >= rules.attentionMm && !allLimited)
                sb.append("最大宽度已达 ${fmtW(st.newMaxW, mmpp)}，超过 ${rules.attentionMm} mm，建议重点关注。")
            else if (level != "稳定") sb.append("建议持续观察，缩短复拍间隔。")
        }
        if (lightRegions > 0) sb.append("另有 $lightRegions 处明暗变化（占 %.1f%%，可能为渗水、水渍或阴影），请现场确认。".format(lightRatio * 100))
        if (mmpp == null) sb.append("（参考照片中没有可用的二维码，尺寸以像素表示。）")
        return level to sb.toString()
    }

    /** 明细 JSON（存数据库）：参考与复拍的统计 + 每条裂缝。 */
    fun toJson(ch: Change, mmpp: Double?, level: String, imagePath: String?): String = JSONObject().apply {
        put("level", level); put("mm_per_px", mmpp ?: JSONObject.NULL); put("image", imagePath ?: JSONObject.NULL)
        put("ref_count", ch.ref.cracks.size); put("ref_total_px", ch.ref.totalPx); put("ref_max_w_px", ch.ref.maxWPx); put("ref_mean_w_px", ch.ref.meanWPx)
        put("lost_px", ch.lostPx); put("grow_px", ch.growPx)
        put("cracks", JSONArray(ch.new.cracks.map { c ->
            JSONObject().put("len_px", c.lengthPx).put("mean_w_px", c.meanWPx).put("max_w_px", c.maxWPx)
                .put("angle", c.angleDeg).put("grow_px", c.growPx).put("new", c.isNew).put("limited", c.limited)
                .put("bbox", JSONArray(listOf(c.x, c.y, c.w, c.h)))
        }))
    }.toString()

    /** 在复拍（已对齐，RGBA）图上画裂缝：原有黄色、新增红色、上次有这次没有蓝色，并标注编号/长度/宽度。 */
    fun draw(aligned: Mat, ch: Change, mmpp: Double?): Mat {
        val out = aligned.clone(); val w = ch.new.w; val h = ch.new.h
        val k3 = Mat.ones(3, 3, CvType.CV_8U)
        fun dil(b: ByteArray, k: Mat = k3): Mat { val m = Mat(); Imgproc.dilate(matOf(b, w, h), m, k); return m }
        val nearNew = bytes(dil(ch.new.region, Mat.ones(9, 9, CvType.CV_8U)))
        val lost = ByteArray(w * h) { if (ch.ref.skel[it].toInt() != 0 && nearNew[it].toInt() == 0) 1 else 0 }
        val old = ByteArray(w * h) { if (ch.new.skel[it].toInt() != 0 && ch.grow[it].toInt() == 0) 1 else 0 }
        out.setTo(Scalar(60.0, 160.0, 255.0, 255.0), dil(lost))
        out.setTo(Scalar(255.0, 220.0, 0.0, 255.0), dil(old))
        out.setTo(Scalar(255.0, 40.0, 40.0, 255.0), dil(ch.grow))
        ch.new.cracks.forEachIndexed { i, c ->
            val label = "#${i + 1} L${fmtLen(c.lengthPx, mmpp).replace(" ", "")} W${if (c.limited) "<=" else ""}${fmtW(c.maxWPx, mmpp).replace(" ", "")}"
            val fs = max(w, h) / 1300.0                       // 字号随图像大小，手机上看得清
            val org = Point(c.x.toDouble(), max(c.y - 12, (30 * fs).toInt()).toDouble())
            Imgproc.putText(out, label, org, Imgproc.FONT_HERSHEY_SIMPLEX, fs, Scalar(0.0, 0.0, 0.0, 255.0), (6 * fs).toInt().coerceAtLeast(3), Imgproc.LINE_AA)
            Imgproc.putText(out, label, org, Imgproc.FONT_HERSHEY_SIMPLEX, fs, Scalar(255.0, 255.0, 255.0, 255.0), (2 * fs).toInt().coerceAtLeast(1), Imgproc.LINE_AA)
        }
        return out
    }
}

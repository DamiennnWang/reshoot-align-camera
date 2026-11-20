package com.fieldalign.camera.align

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 小场景病害：测点类型 + 检测基元（与 python/defects.py 同一套算法和阈值）。输入都已对齐到参考坐标，图像为 RGBA。
 *
 *  面状 areas()        剥落/掉块、锈蚀、渗水/湿痕、泛碱/白化
 *  位移 displacement() 伸缩缝张开、错动/错台、支座位移（分块相位相关，找整块一起移动的区域）
 *  标记线 marks()      螺栓防松标记线：方向变了 → 疑似松动；断了 → 残缺
 *  有无 parts()        部件缺失 / 新增
 * 线状（裂缝）在 Measure 里。手机上只给大致判断，精细分析导出档案后在电脑上做。
 */
object Defects {

    enum class Kind(val code: String, val label: String, val group: String, val aliases: List<String>) {
        CRACK("crack", "裂缝", "线状", listOf("裂缝", "裂纹", "焊缝", "crack")),
        JOINT("joint", "伸缩缝/接缝", "位移", listOf("伸缩缝", "接缝", "轨缝", "沉降缝", "joint")),
        OFFSET("offset", "错台/位移", "位移", listOf("错台", "错动", "位移", "支座", "offset")),
        SPALL("spall", "剥落/掉块", "面状", listOf("剥落", "掉块", "脱落", "露筋", "spall")),
        SEEP("seep", "渗水/水渍", "面状", listOf("渗水", "渗漏", "水渍", "湿渍", "seep")),
        EFFL("effl", "泛碱/白化", "面状", listOf("泛碱", "白化", "析出", "effl")),
        RUST("rust", "锈蚀/涂层", "面状", listOf("锈蚀", "锈", "涂层", "rust")),
        BOLT("bolt", "螺栓", "部件", listOf("螺栓", "螺母", "bolt")),
        PART("part", "扣件/部件", "部件", listOf("扣件", "部件", "缺失", "part")),
    }

    val GROUPS = listOf("线状", "面状", "位移", "部件")

    fun parse(s: String?): Set<Kind> =
        s?.split(',')?.mapNotNull { c -> Kind.values().firstOrNull { it.code == c.trim() } }?.toSet() ?: emptySet()

    fun encode(k: Set<Kind>): String? = if (k.isEmpty()) null else Kind.values().filter { it in k }.joinToString(",") { it.code }

    fun labels(k: Set<Kind>): String = if (k.isEmpty()) "通用" else Kind.values().filter { it in k }.joinToString("、") { it.label }

    /** 从二维码内容读类型，例如 “B01-Z023;40mm;螺栓” 或 “B01-Z023;bolt,joint”。 */
    fun fromQr(text: String?): Set<Kind> {
        if (text == null) return emptySet()
        val toks = text.split(Regex("[;,|，；、\\s]+")).drop(1).map { it.trim().lowercase() }.filter { it.isNotEmpty() }
        return Kind.values().filter { k -> toks.any { t -> k.aliases.any { a -> t.contains(a.lowercase()) } } }.toSet()
    }

    /** 测点类型 → 要跑的基元。空集合 = 通用（裂缝 + 面状 + 一致区内的位移）。 */
    class Plan(val kinds: Set<Kind>) {
        val general = kinds.isEmpty()
        val line = general || Kind.CRACK in kinds
        val area = general || kinds.any { it.group == "面状" }
        val disp = general || Kind.JOINT in kinds || Kind.OFFSET in kinds
        val dispFull = Kind.JOINT in kinds || Kind.OFFSET in kinds   // 位移类测点：移动的一侧本来就不在特征一致区里
        val marks = Kind.BOLT in kinds
        val parts = Kind.BOLT in kinds || Kind.PART in kinds
    }

    // ================================================================== 工具
    private fun floats(m: Mat): FloatArray {
        val f = Mat(); m.convertTo(f, CvType.CV_32F)
        val a = FloatArray(m.rows() * m.cols()); f.get(0, 0, a); return a
    }

    private fun bytes(m: Mat): ByteArray { val a = ByteArray(m.rows() * m.cols()); m.get(0, 0, a); return a }

    private fun mad(v: FloatArray, n: Int = v.size): Double {
        if (n == 0) return 1e-6
        val c = v.copyOf(n); java.util.Arrays.sort(c)
        val med = c[n / 2]
        for (i in 0 until n) c[i] = abs(c[i] - med)
        java.util.Arrays.sort(c)
        return 1.4826 * c[n / 2] + 1e-6
    }

    private fun toRgb(img: Mat): Mat {
        if (img.channels() == 3) return img
        val o = Mat(); Imgproc.cvtColor(img, o, if (img.channels() == 4) Imgproc.COLOR_RGBA2RGB else Imgproc.COLOR_GRAY2RGB); return o
    }

    /** 7 元最小二乘（法方程 + 高斯消元）。 */
    private fun lstsq(rows: List<DoubleArray>, y: DoubleArray, keep: BooleanArray): DoubleArray {
        val k = 7; val a = Array(k) { DoubleArray(k + 1) }
        for ((i, r) in rows.withIndex()) {
            if (!keep[i]) continue
            for (p in 0 until k) { for (q in 0 until k) a[p][q] += r[p] * r[q]; a[p][k] += r[p] * y[i] }
        }
        for (p in 0 until k) a[p][p] += 1e-6
        for (c in 0 until k) {
            var piv = c
            for (r in c + 1 until k) if (abs(a[r][c]) > abs(a[piv][c])) piv = r
            val t = a[c]; a[c] = a[piv]; a[piv] = t
            for (r in 0 until k) if (r != c) {
                val f = a[r][c] / a[c][c]
                for (q in c..k) a[r][q] -= f * a[c][q]
            }
        }
        return DoubleArray(k) { a[it][k] / a[it][it] }
    }

    private fun pca(xs: DoubleArray, ys: DoubleArray): Triple<Double, Double, DoubleArray> {
        val n = xs.size; val mx = xs.average(); val my = ys.average()
        var sxx = 0.0; var syy = 0.0; var sxy = 0.0
        for (i in 0 until n) { val dx = xs[i] - mx; val dy = ys[i] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy }
        sxx /= max(n - 1, 1); syy /= max(n - 1, 1); sxy /= max(n - 1, 1)
        val tr = sxx + syy; val det = sxx * syy - sxy * sxy
        val disc = sqrt(max(tr * tr / 4 - det, 0.0))
        val l1 = tr / 2 + disc; val l2 = tr / 2 - disc
        var ux = l1 - syy; var uy = sxy
        if (abs(ux) + abs(uy) < 1e-12) { ux = if (sxx >= syy) 1.0 else 0.0; uy = if (sxx >= syy) 0.0 else 1.0 }
        val nn = hypot(ux, uy)
        return Triple(sqrt(12 * max(l1, 0.0)), sqrt(12 * max(l2, 0.0)), doubleArrayOf(ux / nn, uy / nn))
    }

    // ================================================================== 面状
    data class Area(val cls: String, val x: Int, val y: Int, val w: Int, val h: Int, val px: Double, val mm2: Double?,
                    val dL: Double, val contour: List<P>)

    const val F = 4

    /**
     * @param mask 255 = 参与比较（有效、在一致区内、非靶标）
     * @param change AlignEngine.compare 的变化图（64 = 反光/过曝，不比较）
     */
    fun areas(ref: Mat, al: Mat, mask: Mat, change: Mat, mmppAt: ((P) -> Double)?): List<Area> {
        val w0 = ref.cols(); val h0 = ref.rows()
        val sz = Size((w0 / F).toDouble(), (h0 / F).toDouble())
        fun lab(im: Mat): Mat { val s = Mat(); Imgproc.resize(toRgb(im), s, sz, 0.0, 0.0, Imgproc.INTER_AREA)
            val l = Mat(); Imgproc.cvtColor(s, l, Imgproc.COLOR_RGB2Lab); return l }
        val la = lab(ref); val lb = lab(al)
        val ca = ArrayList<Mat>(); Core.split(la, ca); val cb = ArrayList<Mat>(); Core.split(lb, cb)
        val A = ca.map { floats(it) }; val B = cb.map { floats(it) }
        val glare = Mat(); Core.compare(change, Scalar(64.0), glare, Core.CMP_NE)
        val mm0 = Mat(); Core.bitwise_and(mask, glare, mm0)
        val ms = Mat(); Imgproc.resize(mm0, ms, sz, 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.threshold(ms, ms, 250.0, 255.0, Imgproc.THRESH_BINARY)
        Imgproc.erode(ms, ms, Mat.ones(3, 3, CvType.CV_8U))
        val w = ms.cols(); val h = ms.rows(); val n = w * h
        val mb = bytes(ms)
        val idx = ArrayList<Int>(); for (i in 0 until n step 2) if (mb[i].toInt() != 0) idx.add(i)
        if (idx.size < 200) return emptyList()
        fun design(i: Int): DoubleArray {
            val x = (i % w).toDouble() / w - 0.5; val y = (i / w).toDouble() / h - 0.5
            return doubleArrayOf(1.0, x, y, x * x, y * y, x * y, A[0][i] / 255.0)
        }
        val rows = idx.map { design(it) }
        // 每个通道：差值 ≈ 光照梯度 + 曝光差（随亮度），稳健拟合后取残差
        val res = Array(3) { FloatArray(n) }
        val sig = DoubleArray(3)
        for (ch in 0 until 3) {
            val y = DoubleArray(idx.size) { (B[ch][idx[it]] - A[ch][idx[it]]).toDouble() }
            val keep = BooleanArray(idx.size) { true }
            var c = lstsq(rows, y, keep)
            repeat(3) {
                val r = FloatArray(idx.size) { (y[it] - rows[it].indices.sumOf { k -> rows[it][k] * c[k] }).toFloat() }
                val kept = r.filterIndexed { i, _ -> keep[i] }.toFloatArray()
                val s = mad(kept)
                for (i in r.indices) keep[i] = abs(r[i]) < 2.5 * s
                c = lstsq(rows, y, keep)
            }
            for (i in 0 until n) {
                val d = design(i)
                var fit = 0.0; for (k in 0 until 7) fit += d[k] * c[k]
                res[ch][i] = (B[ch][i] - A[ch][i] - fit).toFloat()
            }
            val sm = FloatArray(idx.size) { res[ch][idx[it]] }
            sig[ch] = mad(sm)
        }
        val rL = res[0]; val ra = res[1]; val rb = res[2]
        // 局部相关（7×7）：纹理是否被替换
        val fa = Mat(); ca[0].convertTo(fa, CvType.CV_32F); val fb = Mat(); cb[0].convertTo(fb, CvType.CV_32F)
        fun box(x: Mat): Mat { val o = Mat(); Imgproc.boxFilter(x, o, -1, Size(7.0, 7.0)); return o }
        val ma = box(fa); val mbx = box(fb)
        val faa = Mat(); Core.multiply(fa, fa, faa); val fbb = Mat(); Core.multiply(fb, fb, fbb); val fab = Mat(); Core.multiply(fa, fb, fab)
        val Ma = floats(ma); val Mb = floats(mbx); val Saa = floats(box(faa)); val Sbb = floats(box(fbb)); val Sab = floats(box(fab))
        val tL = max(10.0, 4 * sig[0])
        val dark = BooleanArray(n); val white = BooleanArray(n); val rusty = BooleanArray(n); val tex = BooleanArray(n)
        val u = ByteArray(n)
        for (i in 0 until n) {
            if (mb[i].toInt() == 0) continue
            val va = max(Saa[i] - Ma[i] * Ma[i], 1e-3f); val vb = max(Sbb[i] - Mb[i] * Mb[i], 1e-3f)
            val ncc = (Sab[i] - Ma[i] * Mb[i]) / sqrt(va * vb)
            val chroma = hypot(B[1][i] - 128.0, B[2][i] - 128.0)
            dark[i] = rL[i] < -tL
            white[i] = rL[i] > tL && chroma < 20
            rusty[i] = ra[i] > max(5.0, 4 * sig[1]) && rb[i] > max(4.0, 3 * sig[2]) && B[1][i] - 128 > 6
            tex[i] = ncc < 0.2 && sqrt(va) > 3 && abs(rL[i]) > 0.5 * tL
            if (dark[i] || white[i] || rusty[i] || tex[i]) u[i] = 0xFF.toByte()
        }
        val um = Mat(h, w, CvType.CV_8U); um.put(0, 0, u)
        Imgproc.morphologyEx(um, um, Imgproc.MORPH_CLOSE, Mat.ones(5, 5, CvType.CV_8U))
        Imgproc.morphologyEx(um, um, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8U))
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val nc = Imgproc.connectedComponentsWithStats(um, labels, stats, cents, 8, CvType.CV_32S)
        val dist = Mat(); Imgproc.distanceTransform(um, dist, Imgproc.DIST_L2, 3)
        val df = floats(dist); val lab = IntArray(n); labels.get(0, 0, lab)
        class Acc { var n = 0; var dk = 0; var wt = 0; var rs = 0; var tx = 0; var sl = 0.0; var dmax = 0f }
        val acc = Array(nc) { Acc() }
        for (i in 0 until n) {
            val l = lab[i]; if (l == 0) continue
            val a = acc[l]; a.n++; a.sl += rL[i]; a.dmax = max(a.dmax, df[i])
            if (dark[i]) a.dk++; if (white[i]) a.wt++; if (rusty[i]) a.rs++; if (tex[i]) a.tx++
        }
        val out = ArrayList<Area>(); val st = IntArray(5)
        for (l in 1 until nc) {
            stats.get(l, 0, st)
            val (x, y, bw, bh, npx) = st.toList()
            val a = acc[l]
            if (a.dmax < 2f || min(bw, bh) < 0.15 * max(bw, bh)) continue        // 细线、细长条交给线状/位移基元
            val cx = (x + bw / 2.0) * F; val cy = (y + bh / 2.0) * F
            val mmpp = mmppAt?.invoke(P(cx, cy))
            val areaPx = npx.toDouble() * F * F
            val mm2 = mmpp?.let { areaPx * it * it }
            if ((mm2 ?: (areaPx / 400)) < 100) continue                           // < 1 cm² 不报
            val f = { k: Int -> k.toDouble() / a.n }
            val meanL = a.sl / a.n
            val cls = when {
                f(a.rs) >= 0.3 -> "锈蚀"
                f(a.wt) >= 0.4 -> "泛碱/白化"
                f(a.tx) >= 0.25 -> if (meanL < 0) "剥落/掉块" else "表面缺损"
                f(a.dk) >= 0.4 -> "渗水/湿痕"
                else -> "表面变化"
            }
            val one = Mat(); Core.compare(labels, Scalar(l.toDouble()), one, Core.CMP_EQ)
            val cs = ArrayList<MatOfPoint>(); Imgproc.findContours(one, cs, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            val ct = cs.maxByOrNull { Imgproc.contourArea(it) }?.toArray()?.map { P(it.x * F, it.y * F) } ?: emptyList()
            out.add(Area(cls, x * F, y * F, bw * F, bh * F, areaPx, mm2, meanL, ct))
        }
        return out.sortedByDescending { it.px }
    }

    // ================================================================== 位移
    data class Cell(val cx: Double, val cy: Double, val dx: Double, val dy: Double, val r: Double)
    data class Disp(val dx: Double, val dy: Double, val px: Double, val mm: Double?, val opening: Double?, val shear: Double?,
                    val blocks: Int, val box: IntArray, val cells: List<Cell>, val cdx: Double = dx, val cdy: Double = dy)

    const val BLK = 128; const val STEP = 64

    fun displacement(refGray: Mat, newGray: Mat, mask: Mat, mmppAt: ((P) -> Double)?): Disp? {
        fun hp(g: Mat): Mat { val f = Mat(); g.convertTo(f, CvType.CV_32F); val b = Mat()
            Imgproc.GaussianBlur(f, b, Size(0.0, 0.0), 8.0); Core.subtract(f, b, f); return f }
        val a = hp(refGray); val b = hp(newGray)
        val win = Mat(); Imgproc.createHanningWindow(win, Size(BLK.toDouble(), BLK.toDouble()), CvType.CV_32F)
        val H = a.rows(); val W = a.cols()
        val cells = HashMap<Long, Cell>()
        fun key(gx: Int, gy: Int) = gx.toLong() shl 32 or (gy.toLong() and 0xffffffffL)
        val resp = DoubleArray(1)
        val ms = org.opencv.core.MatOfDouble(); val sd = org.opencv.core.MatOfDouble()
        var gy = 0
        var y = 0
        while (y + BLK <= H) {
            var gx = 0; var x = 0
            while (x + BLK <= W) {
                val r = org.opencv.core.Rect(x, y, BLK, BLK)
                if (Core.countNonZero(mask.submat(r)) >= 0.95 * BLK * BLK) {
                    val pa = a.submat(r); val pb = b.submat(r)
                    Core.meanStdDev(pa, ms, sd); val s1 = sd.toArray()[0]
                    Core.meanStdDev(pb, ms, sd); val s2 = sd.toArray()[0]
                    if (s1 >= 4 && s2 >= 4) {
                        val p = Imgproc.phaseCorrelate(pa.clone(), pb.clone(), win, resp)
                        if (resp[0] >= 0.08) cells[key(gx, gy)] = Cell(x + BLK / 2.0, y + BLK / 2.0, p.x, p.y, resp[0])
                    }
                }
                gx++; x += STEP
            }
            gy++; y += STEP
        }
        if (cells.size < 8) return null
        // 不动的平面：RANSAC 拟合仿射位移场（吸收对齐时的小偏差），残差大的块才算“移动”
        val keys = cells.keys.toList(); val nCell = keys.size
        val cl = keys.map { cells[it]!! }
        val NP = 6                                                   // 二次位移场：dx、dy 各 6 个系数
        val U = Array(nCell) { val u = (cl[it].cx - W / 2.0) / W; val v = (cl[it].cy - H / 2.0) / H; doubleArrayOf(1.0, u, v, u * u, v * v, u * v) }
        fun gauss(m: Array<DoubleArray>, ncol: Int): Boolean {
            for (c in 0 until NP) {
                var piv = c; for (r in c + 1 until NP) if (abs(m[r][c]) > abs(m[piv][c])) piv = r
                if (abs(m[piv][c]) < 1e-12) return false
                val t = m[c]; m[c] = m[piv]; m[piv] = t
                for (r in 0 until NP) if (r != c) { val f = m[r][c] / m[c][c]; for (q in c until ncol) m[r][q] -= f * m[c][q] }
            }
            return true
        }
        fun solveRows(rows: List<Int>): Array<DoubleArray>? {         // NP 个点的精确解
            val m = Array(NP) { r -> DoubleArray(NP + 2).also { a -> for (q in 0 until NP) a[q] = U[rows[r]][q]; a[NP] = cl[rows[r]].dx; a[NP + 1] = cl[rows[r]].dy } }
            if (!gauss(m, NP + 2)) return null
            return arrayOf(DoubleArray(NP) { m[it][NP] / m[it][it] }, DoubleArray(NP) { m[it][NP + 1] / m[it][it] })
        }
        fun fitLs(sel: BooleanArray): Array<DoubleArray> {
            val m = Array(NP) { DoubleArray(NP + 2) }
            for (i in 0 until nCell) if (sel[i]) for (p in 0 until NP) {
                for (q in 0 until NP) m[p][q] += U[i][p] * U[i][q]; m[p][NP] += U[i][p] * cl[i].dx; m[p][NP + 1] += U[i][p] * cl[i].dy }
            for (p in 0 until NP) m[p][p] += 1e-9
            gauss(m, NP + 2)
            return arrayOf(DoubleArray(NP) { m[it][NP] / m[it][it] }, DoubleArray(NP) { m[it][NP + 1] / m[it][it] })
        }
        fun resid(c: Array<DoubleArray>, i: Int): Pair<Double, Double> {
            var fx = 0.0; var fy = 0.0; for (q in 0 until NP) { fx += c[0][q] * U[i][q]; fy += c[1][q] * U[i][q] }
            return Pair(cl[i].dx - fx, cl[i].dy - fy)
        }
        val rnd = java.util.Random(1)
        var bestIn = BooleanArray(nCell); var bestN = -1
        repeat(500) {
            val s3 = HashSet<Int>(); while (s3.size < NP) s3.add(rnd.nextInt(nCell))
            val c = solveRows(s3.toList()) ?: return@repeat
            val inl = BooleanArray(nCell) { i -> val (rx, ry) = resid(c, i); hypot(rx, ry) < 0.6 }
            val n = inl.count { it }
            if (n > bestN) { bestN = n; bestIn = inl }
        }
        if (bestN < NP) return null
        val model = fitLs(bestIn)
        val res = HashMap<Long, Pair<Double, Double>>()
        val mags = FloatArray(nCell); val inMags = ArrayList<Float>()
        for (i in 0 until nCell) { val r = resid(model, i); res[keys[i]] = r; mags[i] = hypot(r.first, r.second).toFloat()
            if (bestIn[i]) inMags.add(mags[i]) }
        val thr = max(0.8, 3 * mad(inMags.toFloatArray()))
        val moved = keys.filterIndexed { i, _ -> mags[i] >= thr }.toHashSet()
        fun gxy(k: Long) = Pair((k shr 32).toInt(), k.toInt())
        val seen = HashSet<Long>(); var best = emptyList<Long>()
        for (k0 in moved) {
            if (k0 in seen) continue
            val comp = ArrayList<Long>(); val q = ArrayDeque<Long>(); q.add(k0); seen.add(k0)
            while (q.isNotEmpty()) {
                val c = q.removeLast(); comp.add(c)
                val cc = res[c]!!; val (cx, cy) = gxy(c)
                for (i in -1..1) for (j in -1..1) {
                    val nb = key(cx + i, cy + j)
                    val nc = res[nb] ?: continue
                    if (nb in moved && nb !in seen && hypot(nc.first - cc.first, nc.second - cc.second) < max(1.0, 0.35 * hypot(cc.first, cc.second))) {
                        seen.add(nb); q.add(nb)
                    }
                }
            }
            if (comp.size > best.size) best = comp
        }
        if (best.size < 4) return null
        fun med(v: List<Double>) = v.sorted()[v.size / 2]
        var dx = med(best.map { res[it]!!.first }); var dy = med(best.map { res[it]!!.second })     // 相对不动平面的位移
        val cdx = med(best.map { cells[it]!!.dx }); val cdy = med(best.map { cells[it]!!.dy })      // 对齐图里的实际偏移（搬回原位用）
        val bs = best.toHashSet()
        // 缝两侧：离缝 2~5 个网格的块（紧贴缝的块同时含两侧纹理，不用），各自拟合平面，在缝的位置求差
        fun ring(k: Long, other: (Long) -> Boolean): Int { val (x0, y0) = gxy(k)
            for (d in 1..5) for (i in -d..d) for (j in -d..d) if ((abs(i) == d || abs(j) == d) && other(key(x0 + i, y0 + j))) return d
            return 99 }
        val dStat = cells.keys.filter { it !in moved }.associateWith { ring(it) { n -> n in bs } }
        val dMov = best.associateWith { ring(it) { n -> n in cells && n !in moved } }
        var statAdj = dStat.filter { it.value in 2..5 }.keys.toList(); var movAdj = dMov.filter { it.value in 2..5 }.keys.toList()
        if (statAdj.size < 3 || movAdj.size < 3) { statAdj = dStat.filter { it.value <= 5 }.keys.toList(); movAdj = dMov.filter { it.value <= 5 }.keys.toList() }
        var opening: Double? = null; var shear: Double? = null
        if (statAdj.size >= 3 && movAdj.size >= 3) {
            val sx = statAdj.map { cells[it]!!.cx }.average(); val sy = statAdj.map { cells[it]!!.cy }.average()
            val mx = movAdj.map { cells[it]!!.cx }.average(); val my = movAdj.map { cells[it]!!.cy }.average()
            val px = (sx + mx) / 2; val py = (sy + my) / 2
            fun planeAt(ks: List<Long>): Pair<Double, Double> {          // 最小二乘平面 d = a + b·x + c·y，在 (px, py) 处取值
                val m = Array(3) { DoubleArray(5) }
                for (k in ks) { val c = cells[k]!!; val r = doubleArrayOf(1.0, (c.cx - px) / W, (c.cy - py) / H)
                    for (p in 0 until 3) { for (q in 0 until 3) m[p][q] += r[p] * r[q]; m[p][3] += r[p] * c.dx; m[p][4] += r[p] * c.dy } }
                for (p in 0 until 3) m[p][p] += 1e-6
                for (c in 0 until 3) { var piv = c; for (r in c + 1 until 3) if (abs(m[r][c]) > abs(m[piv][c])) piv = r
                    val t = m[c]; m[c] = m[piv]; m[piv] = t
                    for (r in 0 until 3) if (r != c) { val f = m[r][c] / m[c][c]; for (q in c until 5) m[r][q] -= f * m[c][q] } }
                return Pair(m[0][3] / m[0][0], m[0][4] / m[0][0])
            }
            val ps = planeAt(statAdj); val pm = planeAt(movAdj)
            dx = pm.first - ps.first; dy = pm.second - ps.second
            var nx = mx - sx; var ny = my - sy
            val nn = hypot(nx, ny) + 1e-9; nx /= nn; ny /= nn
            // 缝的走向：紧贴缝的移动块排成一条线，主方向即缝的方向，法向与之垂直（朝移动一侧）
            val edge = dMov.filter { it.value == 1 }.keys.map { cells[it]!! }
            if (edge.size >= 3) {
                val (_, _, t) = pca(edge.map { it.cx }.toDoubleArray(), edge.map { it.cy }.toDoubleArray())
                var ex = -t[1]; var ey = t[0]
                if (ex * nx + ey * ny < 0) { ex = -ex; ey = -ey }
                nx = ex; ny = ey
            }
            opening = dx * nx + dy * ny; shear = abs(dx * ny - dy * nx)
        }
        val cs = best.map { cells[it]!! }
        val ccx = cs.map { it.cx }.average(); val ccy = cs.map { it.cy }.average()
        val mmpp = mmppAt?.invoke(P(ccx, ccy))
        val k = mmpp ?: 1.0
        val x0 = cs.minOf { it.cx } - BLK / 2; val y0 = cs.minOf { it.cy } - BLK / 2
        val box = intArrayOf(x0.toInt(), y0.toInt(), (cs.maxOf { it.cx } + BLK / 2 - x0).toInt(), (cs.maxOf { it.cy } + BLK / 2 - y0).toInt())
        return Disp(dx, dy, hypot(dx, dy), mmpp?.let { hypot(dx, dy) * it }, opening?.let { it * k }, shear?.let { it * k }, best.size, box, cs, cdx, cdy)
    }

    /** 把整块移动的区域按位移搬回原位，之后的面状/部件分析就不会把“整体移动”误判为表面变化。 */
    fun compensate(al: Mat, d: Disp?): Mat {
        if (d == null || d.cells.isEmpty()) return al
        val region = Mat.zeros(al.rows(), al.cols(), CvType.CV_8U)
        for (c in d.cells) Imgproc.rectangle(region, Point(c.cx - BLK / 2, c.cy - BLK / 2), Point(c.cx + BLK / 2 - 1, c.cy + BLK / 2 - 1), Scalar(255.0), -1)
        val m = Mat(2, 3, CvType.CV_64F); m.put(0, 0, 1.0, 0.0, -d.cdx, 0.0, 1.0, -d.cdy)
        val sh = Mat(); Imgproc.warpAffine(al, sh, m, al.size(), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)
        val out = al.clone(); sh.copyTo(out, region)
        return out
    }

    // ================================================================== 螺栓防松标记线
    data class Mark(val cx: Double, val cy: Double, val ang: Double, val length: Double, val status: String,
                    val dang: Double, val coverage: Double)

    private fun markMask(img: Mat, mask: Mat): Mat {
        val hsv = Mat(); Imgproc.cvtColor(toRgb(img), hsv, Imgproc.COLOR_RGB2HSV)
        val m = Mat(); Core.inRange(hsv, Scalar(0.0, 100.0, 70.0), Scalar(180.0, 255.0, 255.0), m)
        Core.bitwise_and(m, mask, m)
        Imgproc.morphologyEx(m, m, Imgproc.MORPH_OPEN, Mat.ones(2, 2, CvType.CV_8U))
        return m
    }

    fun marks(ref: Mat, al: Mat, mask: Mat): List<Mark> {
        val mr = markMask(ref, mask); val mn = markMask(al, mask)
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(mr, labels, stats, cents, 8, CvType.CV_32S)
        val W = ref.cols(); val H = ref.rows()
        val lab = IntArray(W * H); labels.get(0, 0, lab)
        val nb = bytes(mn)
        val pix = HashMap<Int, Pair<ArrayList<Double>, ArrayList<Double>>>()
        val st = IntArray(5)
        val big = BooleanArray(n)
        for (i in 1 until n) { stats.get(i, 0, st); big[i] = st[4] >= 40 }
        for (i in lab.indices) { val l = lab[i]; if (l > 0 && big[l]) pix.getOrPut(l) { ArrayList<Double>() to ArrayList() }.let {
            it.first.add((i % W).toDouble()); it.second.add((i / W).toDouble()) } }
        val out = ArrayList<Mark>()
        for ((l, xy) in pix) {
            val xs = xy.first.toDoubleArray(); val ys = xy.second.toDoubleArray(); val area = xs.size
            val (L, wd, u) = pca(xs, ys)
            if (L < 25 || L < 3 * max(wd, 1.0)) continue
            val cx = xs.average(); val cy = ys.average()
            val ang = (Math.toDegrees(atan2(u[1], u[0])) + 180) % 180
            val R = L / 2 + 12
            val x0 = max((cx - R).toInt(), 0); val y0 = max((cy - R).toInt(), 0)
            val x1 = min((cx + R).toInt() + 1, W); val y1 = min((cy + R).toInt() + 1, H)
            var nn = 0; var on = 0
            val hx = arrayOf(ArrayList<Double>(), ArrayList<Double>()); val hy = arrayOf(ArrayList<Double>(), ArrayList<Double>())
            val perps = ArrayList<Double>()
            for (yy in y0 until y1) for (xx in x0 until x1) {
                if (nb[yy * W + xx].toInt() == 0 || hypot(xx - cx, yy - cy) > R) continue
                nn++
                val px = xx - cx; val py = yy - cy
                val along = px * u[0] + py * u[1]; val perp = abs(px * u[1] - py * u[0])
                if (perp <= max(3.0, wd) && abs(along) <= L / 2 + 3) on++
                val hh = if (along < 0) 0 else 1; hx[hh].add(px); hy[hh].add(py); perps.add(perp)
            }
            if (nn < 0.25 * area) { out.add(Mark(cx, cy, ang, L, "标记缺失", 0.0, nn.toDouble() / area)); continue }
            val cov = min(on.toDouble() / area, 1.0)
            // 标记线沿原方向分成两半，各自拟合方向：螺母那一半转了、钢板那一半没转（小角度时两段仍连在一起）
            var dmax = 0.0
            for (hh in 0..1) {
                if (hx[hh].size < max(12.0, 0.12 * area)) continue
                val (Lh, wh, uh) = pca(hx[hh].toDoubleArray(), hy[hh].toDoubleArray())
                if (Lh < 10 || Lh < 2 * max(wh, 1.0)) continue
                val ah = (Math.toDegrees(atan2(uh[1], uh[0])) + 180) % 180
                dmax = max(dmax, min(abs(ah - ang), 180 - abs(ah - ang)))
            }
            perps.sort()
            val dev = perps[(perps.size * 0.95).toInt().coerceAtMost(perps.size - 1)] - wd / 2     // 偏离原标记线的横向距离
            val loose = dmax >= 6 || dev > max(5.0, 0.12 * L)
            val status = if (loose) "疑似松动" else if (cov < 0.7) "标记残缺" else "正常"
            out.add(Mark(cx, cy, ang, L, status, dmax, cov))
        }
        return out.sortedWith(compareBy({ it.cy }, { it.cx }))
    }

    // ================================================================== 有无
    data class Part(val x: Int, val y: Int, val w: Int, val h: Int, val status: String, val ncc: Double, val eRatio: Double, val dMean: Double)

    fun parts(refGray: Mat, newGray: Mat, change: Mat, mask: Mat, minPx: Int = 300): List<Part> {
        val s = Mat(); Core.compare(change, Scalar(255.0), s, Core.CMP_EQ); Core.bitwise_and(s, mask, s)
        Imgproc.morphologyEx(s, s, Imgproc.MORPH_CLOSE, Mat.ones(9, 9, CvType.CV_8U))
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(s, labels, stats, cents, 8, CvType.CV_32S)
        val dist = Mat(); Imgproc.distanceTransform(s, dist, Imgproc.DIST_L2, 3)
        val W = refGray.cols(); val H = refGray.rows()
        val a = Mat(); refGray.convertTo(a, CvType.CV_32F); val b = Mat(); newGray.convertTo(b, CvType.CV_32F)
        fun grad(g: Mat): Mat { val gx = Mat(); val gy = Mat(); Imgproc.Sobel(g, gx, CvType.CV_32F, 1, 0); Imgproc.Sobel(g, gy, CvType.CV_32F, 0, 1)
            val m = Mat(); Core.magnitude(gx, gy, m); return m }
        val ga = grad(a); val gb = grad(b)
        val out = ArrayList<Part>(); val st = IntArray(5)
        for (i in 1 until n) {
            stats.get(i, 0, st)
            val (x, y, w, h, npx) = st.toList()
            if (npx < minPx) continue
            val one = Mat(); Core.compare(labels, Scalar(i.toDouble()), one, Core.CMP_EQ)
            if (Core.minMaxLoc(dist, one).maxVal < 4) continue
            val p = (0.3 * max(w, h)).toInt() + 6
            val X0 = max(x - p, 0); val Y0 = max(y - p, 0); val X1 = min(x + w + p, W); val Y1 = min(y + h + p, H)
            val outer = org.opencv.core.Rect(X0, Y0, X1 - X0, Y1 - Y0)
            val inner = Mat.zeros(Y1 - Y0, X1 - X0, CvType.CV_8U)
            Imgproc.rectangle(inner, Point((x - X0).toDouble(), (y - Y0).toDouble()), Point((x - X0 + w - 1).toDouble(), (y - Y0 + h - 1).toDouble()), Scalar(255.0), -1)
            val ring = Mat(); Core.bitwise_not(inner, ring)
            val pa = a.submat(outer); val pb = b.submat(outer)
            val m1 = org.opencv.core.MatOfDouble(); val s1 = org.opencv.core.MatOfDouble()
            Core.meanStdDev(pa, m1, s1); val ma = m1.toArray()[0]; val sa = s1.toArray()[0] + 1e-6
            Core.meanStdDev(pb, m1, s1); val mbv = m1.toArray()[0]; val sb = s1.toArray()[0] + 1e-6
            val za = Mat(); Core.subtract(pa, Scalar(ma), za); Core.multiply(za, Scalar(1 / sa), za)
            val zb = Mat(); Core.subtract(pb, Scalar(mbv), zb); Core.multiply(zb, Scalar(1 / sb), zb)
            val zz = Mat(); Core.multiply(za, zb, zz)
            val ncc = Core.mean(zz, inner).`val`[0]
            if (ncc > 0.6) continue
            val ea = Core.mean(ga.submat(outer), inner).`val`[0]; val eb = Core.mean(gb.submat(outer), inner).`val`[0]
            val ra = Core.mean(ga.submat(outer), ring).`val`[0]; val rb = Core.mean(gb.submat(outer), ring).`val`[0]
            val er = eb / (ea + 1e-6) / ((rb + 1e-6) / (ra + 1e-6))
            val dm = (Core.mean(pb, inner).`val`[0] - Core.mean(pb, ring).`val`[0]) - (Core.mean(pa, inner).`val`[0] - Core.mean(pa, ring).`val`[0])
            val status = if (er < 0.65 || dm < -25) "疑似缺失" else if (er > 1.5 || dm > 25) "新增物体" else "外观改变"
            out.add(Part(x, y, w, h, status, ncc, er, dm))
        }
        return out
    }
}

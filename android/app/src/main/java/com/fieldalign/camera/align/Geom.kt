package com.fieldalign.camera.align

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.hypot

/** 平面点。 */
data class P(val x: Double, val y: Double) {
    fun dist(o: P) = hypot(x - o.x, y - o.y)
    operator fun minus(o: P) = P(x - o.x, y - o.y)
    operator fun plus(o: P) = P(x + o.x, y + o.y)
    operator fun times(s: Double) = P(x * s, y * s)
}

/** 3x3 单应矩阵（行优先）。 */
class Homography(val m: DoubleArray) {
    init { require(m.size == 9) }

    fun apply(p: P): P {
        val w = m[6] * p.x + m[7] * p.y + m[8]
        return P((m[0] * p.x + m[1] * p.y + m[2]) / w, (m[3] * p.x + m[4] * p.y + m[5]) / w)
    }

    fun apply(ps: List<P>) = ps.map { apply(it) }

    fun inverse(): Homography {
        val a = m
        val det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
            a[2] * (a[3] * a[7] - a[4] * a[6])
        require(abs(det) > 1e-15) { "singular homography" }
        val inv = doubleArrayOf(
            a[4] * a[8] - a[5] * a[7], a[2] * a[7] - a[1] * a[8], a[1] * a[5] - a[2] * a[4],
            a[5] * a[6] - a[3] * a[8], a[0] * a[8] - a[2] * a[6], a[2] * a[3] - a[0] * a[5],
            a[3] * a[7] - a[4] * a[6], a[1] * a[6] - a[0] * a[7], a[0] * a[4] - a[1] * a[3],
        )
        return Homography(inv.map { it / det }.toDoubleArray())
    }

    operator fun times(o: Homography): Homography {
        val r = DoubleArray(9)
        for (i in 0..2) for (j in 0..2) {
            var s = 0.0
            for (k in 0..2) s += m[i * 3 + k] * o.m[k * 3 + j]
            r[i * 3 + j] = s
        }
        return Homography(r)
    }

    fun toMat(): Mat = Mat(3, 3, CvType.CV_64F).also { it.put(0, 0, *m) }

    fun toList() = m.toList()

    companion object {
        fun fromMat(mat: Mat?): Homography? {
            if (mat == null || mat.empty() || mat.rows() != 3 || mat.cols() != 3) return null
            val d = DoubleArray(9)
            val m64 = Mat()
            mat.convertTo(m64, CvType.CV_64F)
            m64.get(0, 0, d)
            if (d.any { it.isNaN() }) return null
            return Homography(d)
        }

        fun scale(s: Double) = Homography(doubleArrayOf(s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 1.0))
    }
}

fun List<P>.toMat2f(): MatOfPoint2f = MatOfPoint2f(*map { Point(it.x, it.y) }.toTypedArray())

/** 凸包（Andrew 单调链），逆时针。 */
fun convexHull(pts: List<P>): List<P> {
    if (pts.size < 3) return pts
    val s = pts.sortedWith(compareBy({ it.x }, { it.y }))
    fun cross(o: P, a: P, b: P) = (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    val lower = ArrayList<P>()
    for (p in s) {
        while (lower.size >= 2 && cross(lower[lower.size - 2], lower.last(), p) <= 0) lower.removeAt(lower.size - 1)
        lower.add(p)
    }
    val upper = ArrayList<P>()
    for (p in s.asReversed()) {
        while (upper.size >= 2 && cross(upper[upper.size - 2], upper.last(), p) <= 0) upper.removeAt(upper.size - 1)
        upper.add(p)
    }
    return lower.dropLast(1) + upper.dropLast(1)
}

/** 凸包面积（鞋带公式）。 */
fun convexHullArea(pts: List<P>): Double {
    if (pts.size < 3) return 0.0
    val hull = convexHull(pts)
    var a = 0.0
    for (i in hull.indices) {
        val p = hull[i]; val q = hull[(i + 1) % hull.size]
        a += p.x * q.y - q.x * p.y
    }
    return abs(a) / 2
}

/** 点是否在（以中心放大 grow 倍的）四边形内。 */
fun pointInQuad(q: List<P>, p: P, grow: Double = 1.0): Boolean {
    val cx = q.sumOf { it.x } / q.size
    val cy = q.sumOf { it.y } / q.size
    val g = q.map { P(cx + (it.x - cx) * grow, cy + (it.y - cy) * grow) }
    var sign = 0
    for (i in g.indices) {
        val a = g[i]; val b = g[(i + 1) % g.size]
        val c = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
        val s = if (c > 0) 1 else if (c < 0) -1 else 0
        if (s != 0) {
            if (sign == 0) sign = s else if (s != sign) return false
        }
    }
    return true
}

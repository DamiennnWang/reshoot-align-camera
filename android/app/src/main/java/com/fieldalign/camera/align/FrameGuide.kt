package com.fieldalign.camera.align

import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.sqrt

/**
 * 取景实时引导：
 *  1. ML Kit 读二维码（告诉界面这是哪个系列）；
 *  2. 已知目标系列时，用 ORB 特征（失败则用二维码四角）求“参考照片的取景框”在当前画面里的位置；
 *  3. 由取景框与画面边框的差别给出移动提示；连续几帧都对准时 aligned = true。
 * 所有坐标都归一化到 0..1（竖屏、已按传感器方向摆正）。
 */
class FrameGuide(private val onState: (LiveState) -> Unit) : ImageAnalysis.Analyzer {

    data class LiveState(
        val qrText: String? = null,
        val quad: List<P>? = null,       // 参考取景框在当前画面中的位置（归一化）
        val hint: String = "",
        val aligned: Boolean = false,
        val source: String = "",         // "特征" / "二维码"
        val error: Double? = null,
        val suggestedZone: Long? = null, // 无码时按画面特征唯一识别出的系列
    )

    /** 目标系列：参考照片的低分辨率签名，以及（若有）参考照片中二维码四角的归一化坐标。 */
    class Target(val sig: AlignEngine.Signature, val refQrNorm: List<P>?)

    @Volatile var target: Target? = null
    /** 所有系列的低分辨率参考签名：没有目标、也没扫到码时，用来按画面特征自动识别系列。 */
    @Volatile var candidates: List<Pair<Long, AlignEngine.Signature>> = emptyList()
    private var frameNo = 0
    private var suggested: Long? = null
    private var suggestVotes = 0

    private val scanner = BarcodeScanning.getClient(
        BarcodeScannerOptions.Builder().setBarcodeFormats(Barcode.FORMAT_QR_CODE).build())
    private var smooth: List<P>? = null
    private var goodFrames = 0
    private var lastQr: String? = null
    private var lastQrAt = 0L

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(image: ImageProxy) {
        try {
            val rot = image.imageInfo.rotationDegrees
            val media = image.image ?: return
            val upW = if (rot % 180 == 0) image.width else image.height
            val upH = if (rot % 180 == 0) image.height else image.width

            // 1. 二维码
            var qrText: String? = null
            var qrNorm: List<P>? = null
            try {
                val codes = Tasks.await(scanner.process(InputImage.fromMediaImage(media, rot)))
                val c = codes.firstOrNull { it.rawValue != null && it.cornerPoints?.size == 4 }
                if (c != null) {
                    qrText = c.rawValue
                    qrNorm = c.cornerPoints!!.map { P(it.x.toDouble() / upW, it.y.toDouble() / upH) }
                }
            } catch (_: Exception) { }
            val now = System.currentTimeMillis()
            if (qrText != null) { lastQr = qrText; lastQrAt = now }
            else if (now - lastQrAt < 1500) qrText = lastQr   // 短暂丢码时保持显示

            val t = target
            if (t == null) {
                // 无目标、无码：每隔几帧按画面特征在所有系列里找，连续两次指向同一系列才采纳
                var hint = if (qrText == null) "对准二维码或拍摄对象，或先选择系列" else ""
                val cands = candidates
                if (qrText == null && cands.isNotEmpty() && frameNo++ % 6 == 0) {
                    val sig = AlignEngine.signature(yPlaneUpright(image, rot), side = 640, nFeatures = 1500)
                    // 取景识别只是“建议”（用来启动引导），门槛放宽；拍照后还会再判断一次
                    val idr = AlignEngine.identify(sig, cands, minInliers = 30, minCoverage = 0.03)
                    if (idr.decision == "unique") {
                        if (idr.best == suggested) suggestVotes++ else { suggested = idr.best; suggestVotes = 1 }
                    } else { suggested = null; suggestVotes = 0 }
                    if (idr.decision == "ambiguous") hint = "画面与多个系列相似，请扫码或手动选择系列"
                }
                onState(LiveState(qrText = qrText, hint = hint, suggestedZone = if (suggestVotes >= 2) suggested else null))
                return
            }
            suggested = null; suggestVotes = 0

            // 2. 求参考取景框在当前画面中的位置（归一化）
            var quad: List<P>? = null
            var source = ""
            val gray = yPlaneUpright(image, rot)
            val sig = AlignEngine.signature(gray, side = 640, nFeatures = 1500)
            val (inl, _, hLiveToRef) = AlignEngine.matchSignatures(sig, t.sig)
            if (hLiveToRef != null && inl >= 30) {
                val inv = hLiveToRef.inverse()
                quad = listOf(P(0.0, 0.0), P(t.sig.w.toDouble(), 0.0), P(t.sig.w.toDouble(), t.sig.h.toDouble()), P(0.0, t.sig.h.toDouble()))
                    .map { inv.apply(it) }.map { P(it.x / sig.w, it.y / sig.h) }
                source = "特征"
            } else if (qrNorm != null && t.refQrNorm != null) {
                quad = quadFromQr(qrNorm, t.refQrNorm)
                source = "二维码"
            }
            if (quad == null || quad.any { it.x.isNaN() || abs(it.x) > 5 || abs(it.y) > 5 }) {
                smooth = null; goodFrames = 0
                onState(LiveState(qrText = qrText, hint = "没找到参考画面，对准拍摄对象")); return
            }
            val s = smooth?.let { old -> old.indices.map { old[it] * 0.5 + quad[it] * 0.5 } } ?: quad
            smooth = s
            val (hint, err) = hintFor(s)
            goodFrames = if (err < 0.05) goodFrames + 1 else 0
            onState(LiveState(qrText, s, if (goodFrames >= 3) "已对准，可以拍照" else hint, goodFrames >= 3, source, err))
        } catch (e: Exception) {
            onState(LiveState(hint = "引导出错：${e.message}"))
        } finally {
            image.close()
        }
    }

    private fun yPlaneUpright(image: ImageProxy, rot: Int): Mat {
        val plane = image.planes[0]
        val w = image.width; val h = image.height
        val buf = plane.buffer.duplicate().apply { rewind() }
        val row = plane.rowStride
        val bytes = ByteArray(w * h)
        if (row == w) buf.get(bytes, 0, w * h) else {
            for (y in 0 until h) { buf.position(y * row); buf.get(bytes, y * w, w) }
        }
        val m = Mat(h, w, CvType.CV_8U); m.put(0, 0, bytes)
        val s = 640.0 / max(w, h)
        val small = Mat(); Imgproc.resize(m, small, Size(), s, s, Imgproc.INTER_AREA)
        if (rot == 0) return small
        val out = Mat()
        Core.rotate(small, out, when (rot) { 90 -> Core.ROTATE_90_CLOCKWISE; 180 -> Core.ROTATE_180; else -> Core.ROTATE_90_COUNTERCLOCKWISE })
        return out
    }

    companion object {
        /** 用二维码四角求参考取景框位置；角点顺序不确定时取四种循环顺序中变化最小的。 */
        fun quadFromQr(live: List<P>, ref: List<P>): List<P>? {
            var best: List<P>? = null; var bestD = Double.MAX_VALUE
            for (r in 0..3) {
                val l = (0..3).map { live[(it + r) % 4] }
                val d = l.indices.sumOf { l[it].dist(ref[it]) }
                if (d < bestD) { bestD = d; best = l }
            }
            val src = ref.toMat2f(); val dst = best!!.toMat2f()
            val h = Homography.fromMat(Imgproc.getPerspectiveTransform(src, dst)) ?: return null  // 参考 → 当前
            return listOf(P(0.0, 0.0), P(1.0, 0.0), P(1.0, 1.0), P(0.0, 1.0)).map { h.apply(it) }
        }

        /** 根据参考取景框与当前画面边框的差别给出提示。返回 (提示, 误差)。 */
        fun hintFor(q: List<P>): Pair<String, Double> {
            val ideal = listOf(P(0.0, 0.0), P(1.0, 0.0), P(1.0, 1.0), P(0.0, 1.0))
            val err = q.indices.sumOf { q[it].dist(ideal[it]) } / 4
            val area = convexHullArea(q)
            val cx = q.sumOf { it.x } / 4 - 0.5
            val cy = q.sumOf { it.y } / 4 - 0.5
            val ang = Math.toDegrees(atan2(q[1].y - q[0].y, q[1].x - q[0].x))
            val hint = when {
                err < 0.05 -> "已对准"
                area > 1.3 -> "后退一点"
                area < 0.75 -> "靠近一点"
                abs(ang) > 5 -> if (ang > 0) "手机顺时针转一点" else "手机逆时针转一点"
                abs(cx) >= abs(cy) && abs(cx) > 0.04 -> if (cx > 0) "镜头向右移" else "镜头向左移"
                abs(cy) > 0.04 -> if (cy > 0) "镜头向下移" else "镜头向上移"
                else -> {
                    // 位置、大小都差不多，但形状是梯形：拍摄角度偏了。
                    // 当前相机站在参考位置右侧时，参考框右边显得更高 → 应向左绕回去。
                    val left = q[3].y - q[0].y; val right = q[2].y - q[1].y
                    if (right > left) "向左侧绕一点（角度偏了）" else "向右侧绕一点（角度偏了）"
                }
            }
            return hint to err
        }

        fun sqrtArea(q: List<P>) = sqrt(convexHullArea(q))
    }
}

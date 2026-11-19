package com.fieldalign.camera

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.fieldalign.camera.align.AlignEngine
import com.fieldalign.camera.align.Homography
import com.fieldalign.camera.align.ImageIO
import com.fieldalign.camera.align.Measure
import com.fieldalign.camera.align.P
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.File

/**
 * 在设备/模拟器上用合成照片验证 Kotlin 版算法：
 * 状态判断要与期望一致，对齐误差与 Python 版同一量级。
 * 照片和真值来自 python/synth_test.py（构建时复制进测试 APK）。
 */
@RunWith(AndroidJUnit4::class)
class AlignEngineDeviceTest {
    companion object {
        private const val TAG = "AlignTest"
        @BeforeClass @JvmStatic fun init() { assertTrue(OpenCVLoader.initLocal()) }

        private val ctx get() = InstrumentationRegistry.getInstrumentation().context
        private val target get() = InstrumentationRegistry.getInstrumentation().targetContext

        fun asset(name: String): File {
            val f = File(target.cacheDir, "samples/$name")
            if (!f.exists()) {
                f.parentFile!!.mkdirs()
                ctx.assets.open("samples/$name").use { i -> f.outputStream().use { i.copyTo(it) } }
            }
            return f
        }
    }

    @Test
    fun markerAndFeatureAlignment() {
        val truth = JSONObject(ctx.assets.open("samples/truth.json").bufferedReader().readText())
        val report = StringBuilder("\n用例 | 期望 | 结果 | 误差均值/最大(px) | 特征内点 | 变化 | 耗时ms\n")
        var failures = 0
        for (name in truth.keys()) {
            val t = truth.getJSONObject(name)
            val t0 = System.currentTimeMillis()
            val refWork = ImageIO.loadWorkMat(asset(t.getString("ref")).absolutePath)
            val newWork = ImageIO.loadWorkMat(asset(t.getString("new")).absolutePath)
            // 与应用一致：OpenCV 读不出码时用 ML Kit 后备
            val rm = com.fieldalign.camera.align.QrFallback.detectMarkers(refWork)
            val dots = AlignEngine.selectReferenceDots(rm)
            val nm0 = com.fieldalign.camera.align.QrFallback.detectMarkers(newWork)
            val nm = if (nm0.qrText == null) com.fieldalign.camera.align.QrFallback.recover(refWork, rm, newWork) ?: nm0 else nm0
            val res = AlignEngine.align(refWork, rm, dots, newWork, nm)
            var change = "—"
            if (res.h != null) {
                val (aligned, valid) = AlignEngine.warpToRef(newWork, res)
                val c = AlignEngine.compare(refWork, aligned, valid, AlignEngine.comparisonMask(rm.workW, rm.workH, res, rm.qrCorners, dots), resRatio = AlignEngine.resolutionRatio(res))
                change = "%.2f%% 相似%.2f".format(c.changedRatio * 100, c.similarity)
            }
            val ms = System.currentTimeMillis() - t0
            val ht = t.getJSONArray("H_true").let { a -> Homography(DoubleArray(9) { a.getJSONArray(it / 3).getDouble(it % 3) }) }
            var err = "—"; var maxErr = 0.0
            if (res.h != null) {
                val grid = AlignEngine.roiGrid(rm.qrCorners!!, dots, rm.workW, rm.workH)
                val e = grid.map { g -> res.mapNewToRef(ht.inverse().apply(g)).dist(g) }
                maxErr = e.max()
                err = "%.2f / %.2f".format(e.average(), maxErr)
            }
            val ok = res.status == t.getString("expect") && maxErr < 5.0
            if (!ok) failures++
            report.append("$name | ${t.getString("expect")} | ${res.status} | $err | ${res.featureInliers} | $change | $ms${if (ok) "" else "  ✗"}\n")
            if (!ok) report.append("    ${res.message}；码角 ${nm.qrCorners?.joinToString { "(%.1f,%.1f)".format(it.x, it.y) }}；圆点 ${res.dotsMatched.joinToString { it?.let { p -> "(%.1f,%.1f)".format(p.x, p.y) } ?: "-" }}\n")
        }
        Log.i(TAG, report.toString())
        assertEquals("有 $failures 个用例不符合预期：$report", 0, failures)
    }

    /** 镜头桶形畸变：要能估计出畸变、正确对齐，无变化时不出假变化。 */
    @Test
    fun lensDistortion() {
        val truth = JSONObject(ctx.assets.open("samples/distortion/truth.json").bufferedReader().readText())
        val report = StringBuilder("\n畸变用例 | 状态 | 估计k(新/参考) | 误差均值/最大 | 变化\n")
        var failures = 0
        for (name in truth.keys()) {
            val t = truth.getJSONObject(name)
            val refWork = ImageIO.loadWorkMat(asset("distortion/" + t.getString("ref")).absolutePath)
            val newWork = ImageIO.loadWorkMat(asset("distortion/$name").absolutePath)
            val rm = AlignEngine.detectMarkers(refWork); val dots = AlignEngine.selectReferenceDots(rm)
            val res = AlignEngine.align(refWork, rm, dots, newWork, AlignEngine.detectMarkers(newWork))
            val ht = t.getJSONArray("H_true").let { a -> Homography(DoubleArray(9) { a.getJSONArray(it / 3).getDouble(it % 3) }) }
            val kr = t.getDouble("k_ref"); val kn = t.getDouble("k_new")
            val w = rm.workW; val h = rm.workH
            val grid = AlignEngine.roiGrid(rm.qrCorners!!, dots, w, h)
            val e = grid.map { g ->
                val gn = AlignEngine.distort(ht.inverse().apply(AlignEngine.undistort(g, kr, w, h)), kn, w, h)
                res.mapNewToRef(gn).dist(g) }
            val (aligned, valid) = AlignEngine.warpToRef(newWork, res)
            val c = AlignEngine.compare(refWork, aligned, valid, AlignEngine.comparisonMask(w, h, res, rm.qrCorners, dots), resRatio = AlignEngine.resolutionRatio(res))
            val expectChange = t.getBoolean("expect_change")
            val ok = res.status == "confirmed" && e.max() < 3.0 &&
                (if (expectChange) c.regions.isNotEmpty() else c.changedRatio < 0.001)
            if (!ok) failures++
            report.append("$name | ${res.status} | %+.3f/%+.3f | %.2f/%.2f | %.3f%% %d处%s\n".format(
                res.kNew, res.kRef, e.average(), e.max(), c.changedRatio * 100, c.regions.size, if (ok) "" else "  ✗"))
        }
        Log.i(TAG, report.toString())
        assertEquals("有 $failures 个畸变用例不符合预期：$report", 0, failures)
    }

    /** 阴影：不能报结构变化；阴影下的裂缝延长仍要报。 */
    @Test
    fun lightingClassification() {
        val truth = JSONObject(ctx.assets.open("samples/lighting/truth.json").bufferedReader().readText())
        val refWork = ImageIO.loadWorkMat(asset("lighting/ref.jpg").absolutePath)
        val rm = AlignEngine.detectMarkers(refWork); val dots = AlignEngine.selectReferenceDots(rm)
        val report = StringBuilder("\n光影用例 | 结构变化 | 明暗变化\n")
        var failures = 0
        for (name in truth.keys()) {
            val t = truth.getJSONObject(name)
            val newWork = ImageIO.loadWorkMat(asset("lighting/$name").absolutePath)
            val res = AlignEngine.align(refWork, rm, dots, newWork, AlignEngine.detectMarkers(newWork))
            val (aligned, valid) = AlignEngine.warpToRef(newWork, res)
            val c = AlignEngine.compare(refWork, aligned, valid, AlignEngine.comparisonMask(rm.workW, rm.workH, res, rm.qrCorners, dots), resRatio = AlignEngine.resolutionRatio(res))
            val ok = c.regions.isNotEmpty() == t.getBoolean("expect_structural") && (c.lightRatio > 0.001) == t.getBoolean("expect_light")
            if (!ok) failures++
            report.append("$name | ${c.regions.size}处 %.3f%% | ${c.lightRegions.size}处 %.3f%%%s\n".format(
                c.changedRatio * 100, c.lightRatio * 100, if (ok) "" else "  ✗"))
        }
        Log.i(TAG, report.toString())
        assertEquals("有 $failures 个光影用例不符合预期：$report", 0, failures)
    }

    /** 真实拍摄的退化：噪声、欠/过曝、运动模糊、偏色、反光、竖拍、低分辨率。 */
    @Test
    fun robustness() {
        val truth = JSONObject(ctx.assets.open("samples/robustness/truth.json").bufferedReader().readText())
        val refWork = ImageIO.loadWorkMat(asset("00_reference.jpg").absolutePath)
        val rm = AlignEngine.detectMarkers(refWork); val dots = AlignEngine.selectReferenceDots(rm)
        val w = refWork.cols(); val h = refWork.rows()
        val report = StringBuilder("\n退化用例 | 状态 | 误差均/最大 | 结构变化 | 反光 σ | 耗时ms\n")
        var failures = 0
        for (name in truth.keys()) {
            val t = truth.getJSONObject(name)
            val t0 = System.currentTimeMillis()
            val newWork = ImageIO.loadWorkMat(asset("robustness/$name").absolutePath)
            val res = AlignEngine.align(refWork, rm, dots, newWork, AlignEngine.detectMarkers(newWork), refKey = "robust-ref")
            var ok = res.status == "confirmed"
            var line = "$name | ${res.status}"
            if (res.h != null) {
                val ht = t.getJSONArray("H_true").let { a -> Homography(DoubleArray(9) { a.getJSONArray(it / 3).getDouble(it % 3) }) }
                val inv = ht.inverse()
                val e = (1..8).flatMap { i -> (1..8).map { j -> P(w * i / 9.0, h * j / 9.0) } }
                    .filter { res.roi == null || pointInPolygon(res.roi!!, it) }
                    .map { g -> res.mapNewToRef(inv.apply(g)).dist(g) }
                val (aligned, valid) = AlignEngine.warpToRef(newWork, res)
                val c = AlignEngine.compare(refWork, aligned, valid, AlignEngine.comparisonMask(w, h, res, rm.qrCorners, dots),
                    resRatio = AlignEngine.resolutionRatio(res))
                ok = ok && (e.maxOrNull() ?: 0.0) < 1.5 && c.regions.isNotEmpty() == t.getBoolean("expect_structural")
                line += " | %.2f/%.2f | %d处 %.3f%% | %.1f%% %+.1f".format(e.average(), e.maxOrNull() ?: 0.0,
                    c.regions.size, c.changedRatio * 100, c.glareRatio * 100, c.blurSigma)
            }
            if (!ok) failures++
            report.append(line + " | ${System.currentTimeMillis() - t0}" + (if (ok) "" else "  ✗") + "\n")
        }
        Log.i(TAG, report.toString())
        assertEquals("有 $failures 个退化用例不符合预期：$report", 0, failures)
    }

    private fun pointInPolygon(poly: List<P>, p: P): Boolean {
        var c = false; var j = poly.size - 1
        for (i in poly.indices) {
            val a = poly[i]; val b = poly[j]
            if ((a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x) c = !c
            j = i
        }
        return c
    }

    @Test
    fun featureIdentification() {
        fun sig(n: String) = AlignEngine.signature(ImageIO.loadWorkMat(asset("series/$n").absolutePath))
        val refs = listOf("B" to sig("ref_B.jpg"), "C" to sig("ref_C.jpg"), "D" to sig("ref_D.jpg"))
        val c = AlignEngine.identify(sig("C_new.jpg"), refs)
        val b = AlignEngine.identify(sig("B_new.jpg"), refs)
        val bOnly = AlignEngine.identify(sig("B_new.jpg"), refs.filter { it.first != "D" })
        val x = AlignEngine.identify(sig("X_new.jpg"), refs)
        Log.i(TAG, "C→${c.decision}/${c.best} ${c.message}; B→${b.decision} ${b.message}; B(无D)→${bOnly.decision}/${bOnly.best}; X→${x.decision}")
        assertEquals("unique", c.decision); assertEquals("C", c.best)
        assertEquals("ambiguous", b.decision)
        assertEquals("unique", bOnly.decision); assertEquals("B", bOnly.best)
        assertEquals("no_match", x.decision)

        val r = AlignEngine.alignFeaturesOnly(ImageIO.loadWorkMat(asset("series/ref_C.jpg").absolutePath),
            ImageIO.loadWorkMat(asset("series/C_new.jpg").absolutePath))
        Log.i(TAG, "纯特征对齐：${r.message}")
        assertEquals("features_only", r.status)
    }

    /** 裂缝测量：与 Python 版同一批合成照片、同一真值和容差（长度 ±10%，新增 ±25%，宽度 ±30% 或 1 像素，细裂缝须标出“测不准”）。 */
    @Test
    fun crackMeasurement() {
        val truth = JSONObject(ctx.assets.open("samples/measure/truth.json").bufferedReader().readText())
        val report = StringBuilder("\n裂缝用例 | 比例 测/真 | 长度 测/真 | 最大宽 测/真 | 新增 测/真 | 等级 | 耗时ms\n")
        var failures = 0
        for (name in truth.keys()) {
            val t = truth.getJSONObject(name)
            val refWork = ImageIO.loadWorkMat(asset("measure/${name}_ref.jpg").absolutePath)
            val newWork = ImageIO.loadWorkMat(asset("measure/${name}_new.jpg").absolutePath)
            val rm = AlignEngine.detectMarkers(refWork); val dots = AlignEngine.selectReferenceDots(rm)
            val res = AlignEngine.align(refWork, rm, dots, newWork, AlignEngine.detectMarkers(newWork))
            val (aligned, valid) = AlignEngine.warpToRef(newWork, res)
            val excl = AlignEngine.comparisonMask(refWork.cols(), refWork.rows(), res, rm.qrCorners, dots)
            val m = org.opencv.core.Mat(); org.opencv.core.Core.bitwise_and(excl, valid, m)
            val t0 = System.currentTimeMillis()
            val ch = Measure.compare(AlignEngine.gray(refWork), AlignEngine.gray(aligned), m)
            val dt = System.currentTimeMillis() - t0
            val sc = Measure.scaleFromQr(rm.qrCorners, Measure.qrSizeMm(rm.qrText))!!
            val ks = ch.new.cracks
            val at = if (ks.isEmpty()) P(rm.qrCorners!!.sumOf { it.x } / 4, rm.qrCorners!!.sumOf { it.y } / 4)
                     else P(ks.sumOf { it.cx * it.lengthPx } / ks.sumOf { it.lengthPx }, ks.sumOf { it.cy * it.lengthPx } / ks.sumOf { it.lengthPx })
            val mmpp = sc.mmPerPx(at)
            val (level, text) = Measure.conclude(ch, mmpp)
            val wTrue = t.getDouble("width_mm"); val lNew = t.getDouble("len_new_mm"); val lRef = t.getDouble("len_ref_mm")
            var ok = kotlin.math.abs(mmpp / t.getDouble("mmpp_true") - 1) < 0.05
            if (wTrue > 0) {
                val ln = ch.new.totalPx * mmpp; val wn = ch.new.maxWPx * mmpp; val g = ch.growPx * mmpp
                ok = ok && kotlin.math.abs(ln / lNew - 1) < 0.10 && kotlin.math.abs(g / (lNew - lRef) - 1) < 0.25 && level.startsWith("延长")
                ok = ok && if (wTrue / mmpp < Measure.RES_LIMIT_PX) ks.isNotEmpty() && ks[0].limited && ch.new.maxWPx <= 2.8
                           else kotlin.math.abs(wn - wTrue) < maxOf(0.3 * wTrue, 1.25 * mmpp)
                report.append("$name | %.3f/%.3f | %.1f/%.1f | %.2f/%.2f | %.1f/%.1f | $level | $dt".format(
                    mmpp, t.getDouble("mmpp_true"), ln, lNew, wn, wTrue, g, lNew - lRef))
            } else {
                ok = ok && ks.isEmpty() && ch.ref.cracks.isEmpty()
                report.append("$name | %.3f/%.3f | 裂缝 ${ks.size}/${ch.ref.cracks.size} 条 | $level | $dt".format(mmpp, t.getDouble("mmpp_true")))
            }
            if (!ok) failures++
            report.append((if (ok) "" else "  ✗") + "\n")
            if (name == "w0.5mm") report.append("  结论：$text\n")
            report.append("  计时：${Measure.lastTiming}\n")
        }
        Log.i(TAG, report.toString())
        assertEquals("有 $failures 个裂缝用例不符合预期：$report", 0, failures)
    }
}

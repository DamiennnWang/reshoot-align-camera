package com.fieldalign.camera

import android.util.Log
import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.fieldalign.camera.AlignEngineDeviceTest.Companion.asset
import com.fieldalign.camera.align.AlignEngine
import com.fieldalign.camera.align.Analysis
import com.fieldalign.camera.align.Defects
import com.fieldalign.camera.align.Defects.Kind
import com.fieldalign.camera.align.ImageIO
import com.fieldalign.camera.align.Measure
import com.fieldalign.camera.align.P
import com.fieldalign.camera.data.AppDb
import com.fieldalign.camera.data.Outcome
import com.fieldalign.camera.data.Repository
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.Mat
import java.io.File
import kotlin.math.abs

/**
 * 小场景病害：与 python/test_defects.py 同一批合成样例（samples/defects，python/make_defect_samples.py 生成）、同一判定标准。
 * 每对照片“今天”那张都换了角度、偏暖偏暗、有光照梯度。
 */
@RunWith(AndroidJUnit4::class)
class DefectsDeviceTest {
    companion object {
        private const val TAG = "DefectsTest"
        @BeforeClass @JvmStatic fun init() { assertTrue(OpenCVLoader.initLocal()) }
        private val KINDS = mapOf(
            "spall" to setOf(Kind.SPALL), "rust" to setOf(Kind.RUST), "seep" to setOf(Kind.SEEP), "effl" to setOf(Kind.EFFL),
            "plain_nochange" to emptySet(), "offset" to setOf(Kind.OFFSET), "joint" to setOf(Kind.JOINT),
            "segments_nochange" to setOf(Kind.JOINT), "bolts" to setOf(Kind.BOLT), "bolts_nochange" to setOf(Kind.BOLT),
            "joint_1mm" to setOf(Kind.JOINT), "bolts_10deg" to setOf(Kind.BOLT))
        private val AREA_CLS = mapOf("spall" to "剥落/掉块", "rust" to "锈蚀", "seep" to "渗水/湿痕", "effl" to "泛碱/白化")
    }

    private fun analyze(name: String, kinds: Set<Kind>): Analysis.Result {
        val refWork = ImageIO.loadWorkMat(asset("defects/${name}_ref.jpg").absolutePath)
        val newWork = ImageIO.loadWorkMat(asset("defects/${name}_new.jpg").absolutePath)
        val rm = AlignEngine.detectMarkers(refWork); val dots = AlignEngine.selectReferenceDots(rm)
        val res = AlignEngine.align(refWork, rm, dots, newWork, AlignEngine.detectMarkers(newWork))
        val (aligned, valid) = AlignEngine.warpToRef(newWork, res)
        val w = refWork.cols(); val h = refWork.rows()
        val excl = AlignEngine.comparisonMask(w, h, res, rm.qrCorners, dots)
        val c = AlignEngine.compare(refWork, aligned, valid, excl, resRatio = AlignEngine.resolutionRatio(res))
        val cmp = Mat(); Core.bitwise_and(valid, excl, cmp)
        val full = Mat(); Core.bitwise_and(valid, AlignEngine.markerMask(w, h, rm.qrCorners, dots), full)
        val sc = Measure.scaleFromQr(rm.qrCorners, Measure.qrSizeMm(rm.qrText))
        val qc = P(rm.qrCorners!!.sumOf { it.x } / 4, rm.qrCorners!!.sumOf { it.y } / 4)
        assertEquals("$name 对齐", "confirmed", res.status)
        return Analysis.run(kinds, refWork, aligned, cmp, full, c, sc, qc)
    }

    @Test
    fun syntheticDefects() {
        val truth = JSONObject(asset("defects/truth.json").readText())
        val rep = StringBuilder("\n用例 | 类型 | 等级 | 结果 | 耗时\n"); var fails = 0
        for (name in truth.keys()) {
            val t = truth.getJSONObject(name)
            val kinds = KINDS.getValue(name)
            val r = analyze(name, kinds)
            val big = r.areas.filter { (it.mm2 ?: 0.0) >= 100 }
            val ok = when (val k = t.getString("kind")) {
                "spall", "rust", "seep", "effl" -> {
                    val hit = big.filter { it.cls == AREA_CLS[k] }
                    hit.size == 1 && big.size == 1 && abs(hit[0].mm2!! / t.getDouble("area_mm2") - 1) < 0.35 && r.level != "无变化"
                }
                "offset", "joint" -> r.disp != null && abs(r.disp!!.mm!! - t.getDouble("disp_mm")) < 0.5 && big.isEmpty() &&
                    (if (k == "joint") (r.disp!!.opening ?: 0.0) > 0.75 * t.getDouble("disp_mm") else (r.disp!!.shear ?: 0.0) > 0.8 * t.getDouble("disp_mm"))
                "bolt_small" -> r.marks.size == 4 && r.marks.count { it.status == "疑似松动" } == 1 && r.marks.count { it.status == "正常" } == 3
                "bolt" -> r.marks.size == 4 && r.marks.count { it.status == "疑似松动" } == 1 && r.marks.count { it.status == "正常" } == 2 &&
                    r.parts.any { it.status == "疑似缺失" } && r.level == "建议重点关注"
                else -> big.isEmpty() && r.disp == null && r.marks.all { it.status == "正常" } && r.parts.none { it.status == "疑似缺失" } &&
                    r.level in setOf("无变化", "无裂缝", "稳定", "轻微")
            }
            if (!ok) fails++
            r.disp?.let { d -> rep.append("    位移明细 dx %.1f dy %.1f px, 块 ${d.blocks}, box ${d.box.toList()}, 张开 ${d.opening} 错动 ${d.shear}\n".format(d.dx, d.dy)) }
            rep.append("${if (ok) "✓" else "✗"} $name | ${Defects.labels(kinds)} | ${r.level} | ${r.items.joinToString("；")} | ${r.timing}\n")
            rep.append("    ${r.text.replace("\n", " / ")}\n")
        }
        Log.i(TAG, rep.toString())
        assertEquals("有 $fails 个病害用例不符合预期：$rep", 0, fails)
    }

    /** 业务流程：建档带类型 → 复拍出结论和明细 → 改类型后重新分析 → 导出包里有明细数据。 */
    @Test
    fun kindsWorkflow() = runBlocking {
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val db = Room.inMemoryDatabaseBuilder(target, AppDb::class.java).build()
        val repo = Repository(target, db)
        fun copy(n: String): File { val f = repo.newPhotoFile(); Thread.sleep(3); asset(n).copyTo(f, overwrite = true); return f }
        val ref = copy("defects/bolts_ref.jpg")
        val out = repo.process(ref) as Outcome.NeedRegister
        val zid = repo.createZone("钢箱梁 螺栓群 3#", ref, out.markers, out.autoDots, kinds = setOf(Kind.BOLT))
        val a = repo.process(copy("defects/bolts_new.jpg")) as Outcome.Aligned
        val p = db.photos().get(a.photoId)!!
        Log.i(TAG, "螺栓测点结论：${p.conclusion}\n明细：${Analysis.itemsOf(p.findingsJson)}")
        assertTrue(p.conclusion!!.startsWith("【建议重点关注】") && "疑似松动" in p.conclusion!!)
        assertTrue(File(repo.annotated(p)!!).exists())
        // 改成“裂缝”后重新分析：不再检查标记线
        repo.setKinds(zid, setOf(Kind.CRACK))
        val p2 = db.photos().get(a.photoId)!!
        Log.i(TAG, "改为裂缝后：${p2.conclusion}")
        assertTrue("螺栓" !in p2.conclusion!! && JSONObject(p2.findingsJson!!).getJSONArray("marks").length() == 0)
        repo.setKinds(zid, setOf(Kind.BOLT))
        val zip = repo.exportZone(zid)!!
        val names = java.util.zip.ZipFile(zip).use { z -> z.entries().toList().map { it.name } }
        Log.i(TAG, "导出：$names")
        assertTrue("zone.json" in names && names.any { it.endsWith("_findings.json") } && names.any { it.endsWith("_annotated.jpg") })
        db.close()
    }
}

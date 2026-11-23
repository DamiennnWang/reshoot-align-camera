package com.fieldalign.camera

import android.util.Log
import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.fieldalign.camera.data.AppDb
import com.fieldalign.camera.data.Batch
import com.fieldalign.camera.data.CaptureMode
import com.fieldalign.camera.data.GeoFix
import com.fieldalign.camera.data.Pose
import com.fieldalign.camera.data.Repository
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.File

/**
 * 批量拍摄流程：第一次每张建一个测点（记位置、朝向）→ 下次批量复拍，按 二维码 / 位置 + 朝向 + 图像 匹配；
 * 认不出的进“待确认”，不乱归。B 和 D 背景相似（只看图像分不清），相距约 2 公里。
 */
@RunWith(AndroidJUnit4::class)
class BatchDeviceTest {
    companion object {
        private const val TAG = "BatchTest"
        @BeforeClass @JvmStatic fun init() { assertTrue(OpenCVLoader.initLocal()) }
        private val target get() = InstrumentationRegistry.getInstrumentation().targetContext
    }

    private val db = Room.inMemoryDatabaseBuilder(target, AppDb::class.java).build()
    private val repo = Repository(target, db)
    private fun copy(name: String): File { val f = repo.newPhotoFile(); Thread.sleep(3); AlignEngineDeviceTest.asset(name).copyTo(f, overwrite = true); return f }

    private val atB = GeoFix(31.2300, 121.4700, 8.0)
    private val atD = GeoFix(31.2480, 121.4700, 8.0)
    private val atC = GeoFix(31.2000, 121.5000, 8.0)

    @Test
    fun batchNewThenReshoot() = runBlocking {
        // ---- 第一次：批量建档（无码的三处 + 一处有码）
        val created = listOf("B" to atB, "C" to atC, "D" to atD).map { (n, loc) ->
            repo.capture(copy("series/ref_$n.jpg"), CaptureMode.NEW, loc, Pose(10.0, 0.0), emptySet())
        }
        Log.i(TAG, "建档：$created")
        assertTrue(created.all { it is Batch.Created })
        val names = created.map { (it as Batch.Created).name }
        assertTrue("自动命名：$names", names.all { it.startsWith("测点 ") } && names.toSet().size == 3)
        val idB = (created[0] as Batch.Created).zoneId
        val qr = repo.capture(copy("00_reference.jpg"), CaptureMode.NEW, null, null, setOf(com.fieldalign.camera.align.Defects.Kind.CRACK))
        assertTrue("有码按码建档：$qr", qr is Batch.Created && qr.name == "B01-Z023")
        assertEquals(4, db.zones().all().size)
        assertTrue(db.zones().get(idB)!!.lat != null && db.zones().get(idB)!!.azimuth != null)

        // ---- 下次：批量复拍
        // 1) 有码：按码归入（两种模式都一样）
        val r1 = repo.capture(copy("01_normal.jpg"), CaptureMode.RESHOOT, null, null, emptySet())
        Log.i(TAG, "有码复拍：$r1")
        assertTrue(r1 is Batch.Matched && r1.how == "二维码")
        // 新建模式下拍到已建档的码：也归入（码是唯一的）
        val r1b = repo.capture(copy("02_crack_grew.jpg"), CaptureMode.NEW, null, null, emptySet())
        assertTrue("新建模式遇到已建档的码应归入：$r1b", r1b is Batch.Matched)
        // 2) 无码、在 B 附近、朝向一致：位置缩小范围 + 图像确认 → B
        val r2 = repo.capture(copy("series/B_new.jpg"), CaptureMode.RESHOOT, GeoFix(atB.lat + 0.0002, atB.lon, 10.0), Pose(18.0, 2.0), emptySet())
        Log.i(TAG, "B 附近：$r2")
        assertTrue(r2 is Batch.Matched && r2.zoneId == idB)
        assertTrue(db.photos().get((r2 as Batch.Matched).photoId)!!.how!!.contains("位置"))
        // 3) 无定位：B、D 分不清 → 待确认
        val r3 = repo.capture(copy("series/B_new.jpg"), CaptureMode.RESHOOT, null, null, emptySet())
        Log.i(TAG, "无定位：$r3")
        assertTrue(r3 is Batch.Pending)
        // 4) 在 B 附近但镜头朝向反了（另一个方向的测点）→ 待确认
        val r4 = repo.capture(copy("series/B_new.jpg"), CaptureMode.RESHOOT, atB, Pose(190.0, 0.0), emptySet())
        Log.i(TAG, "朝向相反：$r4")
        assertTrue(r4 is Batch.Pending)
        // 5) 离所有测点都很远 → 待确认
        val r5 = repo.capture(copy("series/B_new.jpg"), CaptureMode.RESHOOT, GeoFix(30.0, 120.0, 8.0), null, emptySet())
        assertTrue(r5 is Batch.Pending)

        // ---- 待确认：给候选；建为新测点；清除
        val pend = db.photos().pending().first()
        assertEquals(3, pend.size)
        val out = repo.process(File(pend.last().path), autoAssign = false)
        Log.i(TAG, "待确认候选：$out")
        assertTrue(out is com.fieldalign.camera.data.Outcome.Unclassified && out.fromPending && out.ranking.isNotEmpty())
        repo.pendingToNewZone(pend[0], emptySet())
        assertEquals(2, db.photos().pending().first().size)
        assertEquals(5, db.zones().all().size)
        repo.clearPending(pend[1].path)
        assertEquals(1, db.photos().pending().first().size)
        // 待确认照片不算在任何测点里
        assertTrue(db.zones().summaries().first().sumOf { it.photoCount } == db.photos().allPhotos().size)

        // ---- 导出全部
        val zip = repo.exportAll()!!
        val entries = java.util.zip.ZipFile(zip).use { z -> z.entries().toList().map { it.name } }
        Log.i(TAG, "导出全部：${entries.size} 个文件，${entries.filter { it.endsWith("zone.json") }}")
        assertTrue("index.json" in entries)
        assertEquals(5, entries.count { it.endsWith("/zone.json") })
        assertTrue(entries.any { it.endsWith("/report.csv") } && entries.any { it.endsWith("_aligned.jpg") })
        db.close()
    }

    /** 拍照（AUTO）模式：新位置自动建档、认识的自动归入、拿不准进待确认；编号对应表改名；靶标 PDF 可解码。 */
    @Test
    fun autoModeCodeMapTargets() = runBlocking {
        File(target.filesDir, "code_map.json").delete()
        val db2 = Room.inMemoryDatabaseBuilder(target, AppDb::class.java).build()
        val repo2 = Repository(target, db2)
        fun cp(name: String): File { val f = repo2.newPhotoFile(); Thread.sleep(3); AlignEngineDeviceTest.asset(name).copyTo(f, overwrite = true); return f }
        val a = repo2.capture(cp("series/ref_B.jpg"), CaptureMode.AUTO, atB, Pose(10.0, 0.0), emptySet())
        Log.i(TAG, "AUTO 第一张：$a")
        assertTrue("库空：直接建档 $a", a is Batch.Created)
        val b = repo2.capture(cp("series/B_new.jpg"), CaptureMode.AUTO, GeoFix(atB.lat + 0.0002, atB.lon, 10.0), Pose(15.0, 0.0), emptySet())
        Log.i(TAG, "AUTO 复拍：$b")
        assertTrue("附近认出：归入 $b", b is Batch.Matched && b.zoneId == (a as Batch.Created).zoneId)
        val c = repo2.capture(cp("series/ref_C.jpg"), CaptureMode.AUTO, atC, Pose(10.0, 0.0), emptySet())
        assertTrue("3 公里外的新位置：建档 $c", c is Batch.Created)
        val d = repo2.capture(cp("series/B_new.jpg"), CaptureMode.AUTO, null, null, emptySet())
        Log.i(TAG, "AUTO 无定位：$d")
        assertTrue("没有定位又认不准：待确认，不重复建档 $d", d is Batch.Pending || d is Batch.Matched)
        assertEquals(2, db2.zones().all().size)

        // 编号对应：已建档的码改名；新码第一次拍到就用对应名称
        val q = repo2.capture(cp("00_reference.jpg"), CaptureMode.AUTO, null, null, emptySet())
        assertTrue(q is Batch.Created && q.name == "B01-Z023")
        repo2.codeMap.put(com.fieldalign.camera.data.CodeMap.Entry("B01-Z023", project = "K3+120", name = "拱腰"))
        assertEquals(1, repo2.applyCodeMap())
        assertEquals("K3+120 拱腰", db2.zones().byQr("B01-Z023")!!.name)
        assertEquals(2, repo2.codeMap.importCsv("\uFEFF编号,项目编号,名称,备注\nT0001,P-12,2#墩,\n\"T0002\",P-13,\"西侧,下部\",备注\n"))
        assertEquals("P-13 西侧,下部", repo2.nameForQr("T0002"))
        assertEquals("T0099", repo2.nameForQr("T0099"))

        // 靶标：PDF 生成；预览图里的二维码能解出编号
        val codes = com.fieldalign.camera.data.CodeMap.codes("T", 1, 7, 4)
        val pdf = com.fieldalign.camera.data.TargetSheet.pdf(codes, File(target.cacheDir, "t.pdf"))
        assertTrue(pdf.length() > 2000)
        val bmp = com.fieldalign.camera.data.TargetSheet.preview("T0007", 800)
        val px = IntArray(bmp.width * bmp.height).also { bmp.getPixels(it, 0, bmp.width, 0, 0, bmp.width, bmp.height) }
        val txt = com.google.zxing.qrcode.QRCodeReader().decode(com.google.zxing.BinaryBitmap(
            com.google.zxing.common.HybridBinarizer(com.google.zxing.RGBLuminanceSource(bmp.width, bmp.height, px)))).text
        Log.i(TAG, "靶标 PDF ${pdf.length()} 字节，预览解码 $txt")
        assertEquals("T0007", txt)
        db2.close()
    }
}

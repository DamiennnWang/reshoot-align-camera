package com.fieldalign.camera

import android.util.Log
import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.fieldalign.camera.align.ImageIO
import com.fieldalign.camera.align.QrFallback
import com.fieldalign.camera.data.AppDb
import com.fieldalign.camera.data.GeoFix
import com.fieldalign.camera.data.Outcome
import com.fieldalign.camera.data.Repository
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * 业务流程测试（内存数据库，不影响 App 数据）：
 *  1. 大场景：扫码锁定后拍全景，照片里读不到码 → 新建系列 / 归入锁定的系列；
 *  2. 无码：背景相似的两个系列，靠 GPS 位置区分。
 */
@RunWith(AndroidJUnit4::class)
class WorkflowDeviceTest {
    companion object {
        private const val TAG = "WorkflowTest"
        @BeforeClass @JvmStatic fun init() { assertTrue(OpenCVLoader.initLocal()) }
        private val target get() = InstrumentationRegistry.getInstrumentation().targetContext
    }

    private val db = Room.inMemoryDatabaseBuilder(target, AppDb::class.java).build()
    private val repo = Repository(target, db)

    private fun copy(name: String): File {
        val f = repo.newPhotoFile(); Thread.sleep(3)
        AlignEngineDeviceTest.asset(name).copyTo(f, overwrite = true)
        return f
    }

    /** 模拟“全景照片里二维码太小读不出”：把码涂掉（连同周围一圈）。 */
    private fun withoutQr(name: String): File {
        val work = ImageIO.loadWorkMat(AlignEngineDeviceTest.asset(name).absolutePath)
        val q = QrFallback.detectMarkers(work).qrCorners!!
        val cx = q.sumOf { it.x } / 4; val cy = q.sumOf { it.y } / 4
        val poly = MatOfPoint(*q.map { Point(cx + (it.x - cx) * 1.5, cy + (it.y - cy) * 1.5) }.toTypedArray())
        Imgproc.fillConvexPoly(work, poly, Scalar(128.0, 128.0, 128.0, 255.0))
        val f = repo.newPhotoFile(); Thread.sleep(3)
        ImageIO.saveJpeg(work, f, 95)
        assertEquals(null, QrFallback.detectMarkers(ImageIO.loadWorkMat(f.absolutePath)).qrText)
        return f
    }

    private val L1 = GeoFix(31.230000, 121.470000, 10.0)

    @Test
    fun scanThenWideShot() = runBlocking {
        // 扫到新码 SITE-001 后后退拍全景（照片里没有可读的码）→ 需要新建系列，编号用扫到的码
        val wide = withoutQr("00_reference.jpg")
        val o1 = repo.process(wide, newQr = "SITE-001", loc = L1)
        assertTrue("应进入新建：$o1", o1 is Outcome.NeedRegister)
        o1 as Outcome.NeedRegister
        assertEquals("SITE-001", o1.qrText)
        val zid = repo.createZone("全景 A", wide, o1.markers, o1.autoDots, o1.qrText)
        val z = db.zones().byQr("SITE-001")
        assertEquals(zid, z?.id)
        assertEquals(L1.lat, z!!.lat!!, 1e-9)

        // 复拍：扫码锁定后拍（照片里仍没有码）→ 强制归入并对齐
        val o2 = repo.process(withoutQr("01_normal.jpg"), forcedZoneId = zid, hintZoneId = zid, loc = L1)
        assertTrue("应对齐：$o2", o2 is Outcome.Aligned)
        val p2 = db.photos().get((o2 as Outcome.Aligned).photoId)!!
        Log.i(TAG, "锁定复拍：${p2.status} ${p2.message} | ${p2.how} 变化${p2.changeRegions}")
        assertEquals("扫码锁定", p2.how)
        assertEquals("features_only", p2.status)
        assertNotNull(p2.lat)

        // 只带取景时扫到的码（照片里读不到）也能归入已有系列；裂缝变长应检出
        val o3 = repo.process(withoutQr("02_crack_grew.jpg"), newQr = "SITE-001")
        assertTrue("应对齐：$o3", o3 is Outcome.Aligned)
        val p3 = db.photos().get((o3 as Outcome.Aligned).photoId)!!
        Log.i(TAG, "按扫到的码归入：${p3.status} ${p3.how} 变化${p3.changeRegions}处 ${p3.changedRatio}")
        assertEquals(zid, p3.zoneId)
        assertTrue(p3.how!!.startsWith("扫码锁定"))
        assertTrue("裂缝变化应检出", (p3.changeRegions ?: 0) > 0)
    }

    @Test
    fun gpsDisambiguatesSimilarBackgrounds() = runBlocking {
        // B 与 D 背景相似（只看特征分不清），两地相距约 2 公里
        val atB = GeoFix(31.2300, 121.4700, 12.0)
        val atD = GeoFix(31.2480, 121.4700, 12.0)
        val atC = GeoFix(31.2000, 121.5000, 12.0)
        for ((n, loc) in listOf("B" to atB, "D" to atD, "C" to atC)) {
            val f = copy("series/ref_$n.jpg")
            val m = QrFallback.detectMarkers(ImageIO.loadWorkMat(f.absolutePath))
            repo.createZone("系列 $n", f, m, emptyList(), null, loc)
        }
        val idOf = db.zones().all().associate { it.name to it.id }

        val noGps = repo.process(copy("series/B_new.jpg"))
        Log.i(TAG, "无定位：$noGps")
        assertTrue("无定位时应分不清", noGps is Outcome.Unclassified)
        assertTrue((noGps as Outcome.Unclassified).distances.isEmpty())

        val near = GeoFix(atB.lat + 0.0003, atB.lon, 15.0)       // 在 B 附近约 33 米
        val withGps = repo.process(copy("series/B_new.jpg"), loc = near)
        assertTrue("有定位时应识别：$withGps", withGps is Outcome.Aligned)
        val p = db.photos().get((withGps as Outcome.Aligned).photoId)!!
        Log.i(TAG, "有定位：${p.how} | ${p.status}")
        assertEquals(idOf["系列 B"], p.zoneId)
        assertTrue(p.how!!.contains("位置"))

        // 定位离所有系列都很远：不能靠位置猜，仍交给人工
        val far = repo.process(copy("series/B_new.jpg"), loc = GeoFix(30.0, 120.0, 10.0))
        assertTrue("远处应不能自动归类：$far", far is Outcome.Unclassified)
        assertTrue((far as Outcome.Unclassified).distances.values.all { it > 10_000 })
    }

    @Test
    fun lockedZoneEdgeCases() = runBlocking {
        val ref = copy("00_reference.jpg")
        val m = QrFallback.detectMarkers(ImageIO.loadWorkMat(ref.absolutePath))
        val zid = repo.createZone("特写参考", ref, m, emptyList())
        suspend fun log(tag: String, o: Outcome): com.fieldalign.camera.data.Photo {
            val p = (o as? Outcome.Aligned)?.let { db.photos().get(it.photoId) }
            Log.i(TAG, "PROBE $tag → ${p?.status} | ${p?.message} | 变化${p?.changeRegions} 相似${p?.similarity} 参考占比${p?.let { repo.refAreaInNew(it, db.photos().get(it.refPhotoId!!)!!) }}".take(400))
            return p!!
        }
        // 锁定的边界情况：
        // 1. 锁定后拍了完全不同的场景
        val diff = log("不同场景", repo.process(copy("series/X_new.jpg"), forcedZoneId = zid, hintZoneId = zid))
        assertEquals("锁定后拍了别的场景应报对不上", "failed", diff.status)
        // 2. 参考是特写，新照片后退拍全景（特写区域只占画面 45%）
        val bg = ImageIO.loadWorkMat(AlignEngineDeviceTest.asset("series/X_new.jpg").absolutePath)
        val near = ImageIO.loadWorkMat(AlignEngineDeviceTest.asset("01_normal.jpg").absolutePath)
        val small = org.opencv.core.Mat()
        Imgproc.resize(near, small, org.opencv.core.Size(), 0.45, 0.45, Imgproc.INTER_AREA)
        Imgproc.resize(bg, bg, org.opencv.core.Size(near.cols().toDouble(), near.rows().toDouble()))
        val x0 = (bg.cols() - small.cols()) / 2; val y0 = (bg.rows() - small.rows()) / 2
        small.copyTo(bg.submat(y0, y0 + small.rows(), x0, x0 + small.cols()))
        val f = repo.newPhotoFile(); Thread.sleep(3); ImageIO.saveJpeg(bg, f, 92)
        val w1 = log("特写参考+全景", repo.process(f, forcedZoneId = zid, hintZoneId = zid))
        assertTrue(w1.status == "confirmed" || w1.status == "features_only")
        assertEquals(0, w1.changeRegions)
        val share1 = repo.refAreaInNew(w1, db.photos().get(w1.refPhotoId!!)!!)!!
        assertTrue("参考应约占 20%：$share1", share1 in 0.15..0.26)
        // 3. 更远：只占 25%
        val small2 = org.opencv.core.Mat()
        Imgproc.resize(near, small2, org.opencv.core.Size(), 0.25, 0.25, Imgproc.INTER_AREA)
        val bg2 = ImageIO.loadWorkMat(AlignEngineDeviceTest.asset("series/X_new.jpg").absolutePath)
        Imgproc.resize(bg2, bg2, org.opencv.core.Size(near.cols().toDouble(), near.rows().toDouble()))
        val a0 = (bg2.cols() - small2.cols()) / 2; val b0 = (bg2.rows() - small2.rows()) / 2
        small2.copyTo(bg2.submat(b0, b0 + small2.rows(), a0, a0 + small2.cols()))
        val f2 = repo.newPhotoFile(); Thread.sleep(3); ImageIO.saveJpeg(bg2, f2, 92)
        val w2 = log("特写参考+远景", repo.process(f2, forcedZoneId = zid, hintZoneId = zid))
        assertTrue(w2.status == "confirmed" || w2.status == "features_only")
        assertTrue(repo.refAreaInNew(w2, db.photos().get(w2.refPhotoId!!)!!)!! < 0.1)
    }

    /** 同一点位、不同方向的两个系列：靠镜头朝向区分（铁路沿线同一处可能有多个方向的照片）。 */
    @Test
    fun headingSeparatesDirectionsAtSameSpot() = runBlocking {
        val here = GeoFix(31.2300, 121.4700, 8.0)
        // B 与 D 背景相似、位置相同，一个朝东一个朝西
        for ((n, az) in listOf("B" to 90.0, "D" to 270.0)) {
            val f = copy("series/ref_$n.jpg")
            val m = QrFallback.detectMarkers(ImageIO.loadWorkMat(f.absolutePath))
            repo.createZone("系列 $n", f, m, emptyList(), null, here, com.fieldalign.camera.data.Pose(az, 0.0))
        }
        val idOf = db.zones().all().associate { it.name to it.id }
        val noPose = repo.process(copy("series/B_new.jpg"), loc = here)
        Log.i(TAG, "同点位无朝向：${noPose.javaClass.simpleName}")
        assertTrue("同一位置、没有朝向时应分不清", noPose is Outcome.Unclassified)
        val east = repo.process(copy("series/B_new.jpg"), loc = here, pose = com.fieldalign.camera.data.Pose(80.0, 3.0))
        assertTrue("朝东应识别：$east", east is Outcome.Aligned)
        val pe = db.photos().get((east as Outcome.Aligned).photoId)!!
        Log.i(TAG, "朝东：${pe.how}")
        assertEquals(idOf["系列 B"], pe.zoneId)
        assertTrue(pe.how!!.contains("朝向"))
        assertEquals(80.0, pe.azimuth!!, 1e-9)
    }
}

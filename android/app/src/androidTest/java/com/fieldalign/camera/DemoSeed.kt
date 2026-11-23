package com.fieldalign.camera

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.fieldalign.camera.AlignEngineDeviceTest.Companion.asset
import com.fieldalign.camera.align.Defects.Kind
import com.fieldalign.camera.data.Repository
import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.File

/**
 * 演示数据（写进 App 真实数据库，用来人工检查界面）。平时跳过；手动运行：
 *   adb shell am instrument -w -e seed 1 -e class com.fieldalign.camera.DemoSeed \
 *     com.fieldalign.camera.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class DemoSeed {
    companion object { @BeforeClass @JvmStatic fun init() { OpenCVLoader.initLocal() } }

    @Test fun seed() = runBlocking {
        assumeTrue(InstrumentationRegistry.getArguments().getString("seed") == "1")
        val repo = Repository(InstrumentationRegistry.getInstrumentation().targetContext)
        fun copy(n: String): File { val f = repo.newPhotoFile(); Thread.sleep(5); asset(n).copyTo(f, overwrite = true); return f }
        for ((name, title, kinds) in listOf(Triple("bolts", "钢箱梁 螺栓群 3#", setOf(Kind.BOLT)),
                                            Triple("joint", "桥面伸缩缝 K12+300", setOf(Kind.JOINT)),
                                            Triple("spall", "二衬 剥落观测点 7#", setOf(Kind.SPALL, Kind.SEEP)))) {
            val ref = copy("defects/${name}_ref.jpg")
            val m = com.fieldalign.camera.align.QrFallback.detectMarkers(com.fieldalign.camera.align.ImageIO.loadWorkMat(ref.absolutePath))
            // 三组样例用的是同一个码：各自起一个编号，复拍时按系列锁定
            val zid = repo.createZone(title, ref, m, com.fieldalign.camera.align.AlignEngine.selectReferenceDots(m), "${m.qrText}-$name", kinds = kinds)
            repo.process(copy("defects/${name}_new.jpg"), forcedZoneId = zid)
        }
        for ((name, title, kinds) in listOf(Triple("curve_grow", "二衬 弯曲裂缝 K3+120", setOf(Kind.CRACK)),
                                            Triple("spall_light", "墩柱 剥落 2#", setOf(Kind.SPALL)),
                                            Triple("spall_on_crack", "箱梁腹板 裂缝+剥落", emptySet()))) {
            val ref = copy("hard/${name}_ref.jpg")
            val m = com.fieldalign.camera.align.QrFallback.detectMarkers(com.fieldalign.camera.align.ImageIO.loadWorkMat(ref.absolutePath))
            val zid = repo.createZone(title, ref, m, com.fieldalign.camera.align.AlignEngine.selectReferenceDots(m), "${m.qrText}-$name", kinds = kinds)
            repo.process(copy("hard/${name}_new.jpg"), forcedZoneId = zid)
        }
        android.util.Log.i("DemoSeed", "导出：" + repo.exportAll()?.absolutePath); Unit
    }
}

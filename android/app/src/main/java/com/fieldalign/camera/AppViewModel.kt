package com.fieldalign.camera

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.fieldalign.camera.align.AlignEngine
import com.fieldalign.camera.align.FrameGuide
import com.fieldalign.camera.align.ImageIO
import com.fieldalign.camera.align.P
import com.fieldalign.camera.data.Outcome
import com.fieldalign.camera.data.Repository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

sealed class Screen {
    data object Home : Screen()
    data class Camera(val zoneId: Long? = null) : Screen()
    data class ZoneDetail(val zoneId: Long) : Screen()
    data class Compare(val photoId: Long) : Screen()
    /** 新建系列（file 为新照片），或把已有照片设为参考（existingPhotoId）。 */
    data class Register(
        val file: File,
        val markers: AlignEngine.Markers,
        val autoDots: List<AlignEngine.Dot>,
        val existingPhotoId: Long? = null,
        /** 系列编号（二维码内容）：扫码锁定后拍全景时，照片里可能读不到码，用扫到的码。 */
        val qrText: String? = markers.qrText,
    ) : Screen()
    data class Unclassified(val outcome: Outcome.Unclassified) : Screen()
}

class AppViewModel(app: Application) : AndroidViewModel(app) {
    val repo = Repository(app)
    val stack = mutableStateListOf<Screen>(Screen.Home)
    var processing by mutableStateOf(false); private set
    var toast by mutableStateOf<String?>(null)

    val current: Screen get() = stack.last()

    // ---------------------------------------------------------------- 导入（相册选择 / 其他应用分享）
    private val importQueue = ArrayDeque<File>()
    private var importedOk = 0
    private var lastImported: Long? = null
    var importProgress by mutableStateOf<String?>(null); private set

    fun importUris(uris: List<android.net.Uri>) = viewModelScope.launch {
        val cr = getApplication<Application>().contentResolver
        val files = withContext(Dispatchers.IO) {
            uris.mapNotNull { u ->
                runCatching {
                    val f = repo.newPhotoFile()
                    cr.openInputStream(u)!!.use { i -> f.outputStream().use { i.copyTo(it) } }
                    Thread.sleep(2)   // 保证文件名（毫秒时间戳）不重复
                    f
                }.getOrNull()
            }
        }
        if (files.isEmpty()) { toast = "没有读到照片"; return@launch }
        importQueue.addAll(files); importedOk = 0; lastImported = null
        runQueue()
    }

    /** 逐张处理导入队列；遇到需要人工确认的照片就停下，确认后继续。 */
    private suspend fun runQueue() {
        while (importQueue.isNotEmpty()) {
            val f = importQueue.removeFirst()
            importProgress = "剩余 ${importQueue.size + 1} 张"
            processing = true
            val out = repo.process(f, null)
            processing = false
            if (out is Outcome.Aligned) { importedOk++; lastImported = out.photoId; continue }
            importProgress = null
            route(out)
            return
        }
        importProgress = null
        if (importedOk == 1 && lastImported != null) go(Screen.Compare(lastImported!!))
        else if (importedOk > 1) toast = "已自动归类并对比 $importedOk 张照片"
        importedOk = 0
    }

    private fun continueQueue() { if (importQueue.isNotEmpty() || importedOk > 0) viewModelScope.launch { runQueue() } }

    fun go(s: Screen) { stack.add(s) }
    fun back(): Boolean { if (stack.size <= 1) return false; stack.removeAt(stack.size - 1); return true }
    fun replaceTop(s: Screen) { stack[stack.size - 1] = s }

    // ---------------------------------------------------------------- 拍照后
    fun onCaptured(file: File, forcedZoneId: Long?, hintZoneId: Long? = null, newQr: String? = null,
                   loc: com.fieldalign.camera.data.GeoFix? = null, pose: com.fieldalign.camera.data.Pose? = null) = viewModelScope.launch {
        processing = true
        val out = repo.process(file, forcedZoneId, hintZoneId, newQr, loc, pose)
        processing = false
        route(out)
    }

    private fun route(out: Outcome) {
        when (out) {
            is Outcome.Aligned -> go(Screen.Compare(out.photoId))
            is Outcome.NeedRegister -> go(Screen.Register(out.file, out.markers, out.autoDots, qrText = out.qrText))
            is Outcome.Unclassified -> go(Screen.Unclassified(out))
            is Outcome.Error -> toast = out.message
        }
    }

    fun assignToZone(file: File, zoneId: Long) = viewModelScope.launch {
        processing = true
        val out = repo.process(file, zoneId)
        processing = false
        back()            // 关掉“未归类”页
        if (importQueue.isNotEmpty() && out is Outcome.Aligned) { importedOk++; lastImported = out.photoId; continueQueue() }
        else route(out)
    }

    fun newSeriesFrom(u: Outcome.Unclassified) {
        replaceTop(Screen.Register(u.file, u.markers, emptyList()))
    }

    fun discard(file: File) { repo.discard(file); back(); continueQueue() }

    fun saveRegister(r: Screen.Register, name: String, dots: List<AlignEngine.Dot>,
                     kinds: Set<com.fieldalign.camera.align.Defects.Kind> = emptySet()) = viewModelScope.launch {
        if (r.existingPhotoId != null) {
            repo.setReference(r.existingPhotoId, dots)
            toast = "已设为参考照片"
            back()
        } else {
            val zid = repo.createZone(name.ifBlank { r.qrText ?: "未命名系列" }, r.file, r.markers, dots, r.qrText, kinds = kinds)
            if (importQueue.isNotEmpty()) { back(); continueQueue() } else replaceTop(Screen.ZoneDetail(zid))
        }
    }

    /** 重新分析进度（改测点类型、改二维码尺寸后），null = 空闲。 */
    var reanalyzing by mutableStateOf<String?>(null); private set

    fun changeKinds(zoneId: Long, kinds: Set<com.fieldalign.camera.align.Defects.Kind>) = viewModelScope.launch {
        reanalyzing = "重新分析…"
        repo.setKinds(zoneId, kinds) { i, n -> reanalyzing = if (n == 0) null else "重新分析 $i / $n" }
        reanalyzing = null; toast = "已按「${com.fieldalign.camera.align.Defects.labels(kinds)}」重新分析"
    }

    fun changeQrMm(zoneId: Long, mm: Double) = viewModelScope.launch {
        reanalyzing = "重新分析…"
        repo.setQrMm(zoneId, mm) { i, n -> reanalyzing = if (n == 0) null else "重新分析 $i / $n" }
        reanalyzing = null
    }

    /** 把已有照片设为参考：先读出它的靶标，再进圆点确认页。 */
    fun startSetReference(photoId: Long) = viewModelScope.launch {
        val p = repo.db.photos().get(photoId) ?: return@launch
        val (m, _) = repo.markersFromJson(p.markersJson)
        go(Screen.Register(File(p.path), m, AlignEngine.selectReferenceDots(m), existingPhotoId = photoId))
    }

    // ---------------------------------------------------------------- 取景引导
    /** 为相机页准备引导目标（参考照片签名 + 参考二维码位置）。 */
    suspend fun guideTarget(zoneId: Long): Pair<FrameGuide.Target, String>? {
        val (sig, refPath) = repo.guideSignature(zoneId) ?: return null
        val z = repo.db.zones().get(zoneId) ?: return null
        val ref = repo.db.photos().get(z.refPhotoId ?: return null) ?: return null
        val (m, _) = repo.markersFromJson(ref.markersJson)
        val qrNorm = m.qrCorners?.map { P(it.x / m.workW, it.y / m.workH) }
        return FrameGuide.Target(sig, qrNorm) to refPath
    }

    suspend fun zoneIdForQr(qr: String): Long? = repo.db.zones().byQr(qr)?.id

    /**
     * 供相机页按画面特征自动识别系列的取景签名（低分辨率）。
     * 有定位时只取附近的系列（以及还没有位置记录的老系列），远处背景相似的系列不参与，识别更快更准。
     */
    suspend fun guideCandidates(here: com.fieldalign.camera.data.GeoFix?): List<Pair<Long, AlignEngine.Signature>> {
        val r = com.fieldalign.camera.data.Geo.nearRadius(here?.acc)
        return repo.db.zones().all().filter { it.refPhotoId != null }
            .filter { z -> here == null || z.lat == null || z.lon == null ||
                com.fieldalign.camera.data.Geo.distance(here.lat, here.lon, z.lat, z.lon) <= r }
            .mapNotNull { z -> repo.guideSignature(z.id)?.let { z.id to it.first } }
    }

    suspend fun loadBitmap(path: String, side: Int = AlignEngine.WORK_MAX_SIDE) =
        withContext(Dispatchers.IO) { runCatching { ImageIO.loadWorkBitmap(path, side) }.getOrNull() }
}

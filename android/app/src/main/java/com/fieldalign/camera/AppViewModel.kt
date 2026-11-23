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
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

sealed class Screen {
    data object Home : Screen()
    /** zoneId 非空：给指定测点复拍（拍完进对比页）；否则批量拍摄（mode 决定新建还是复拍，拍完不跳页）。 */
    data class Camera(val zoneId: Long? = null, val mode: com.fieldalign.camera.data.CaptureMode = com.fieldalign.camera.data.CaptureMode.AUTO) : Screen()
    data object Pending : Screen()
    /** 批量生成编号靶标。 */
    data object Targets : Screen()
    /** 编号对应表（二维码编号 → 项目编号）。 */
    data object CodeMap : Screen()
    /** 使用教程。 */
    data object Tutorial : Screen()
    /** 复拍指定测点：先选测点，再进带参考叠加的相机。 */
    data object PickZone : Screen()
    data object Settings : Screen()
    data object Map : Screen()
    /** 规划中的功能：说明页。 */
    data class Planned(val title: String, val body: String) : Screen()
    data class ZoneDetail(val zoneId: Long) : Screen()
    data class Compare(val photoId: Long) : Screen()
    /** 新建测点（file 为新照片），或把已有照片设为参考（existingPhotoId）。 */
    data class Register(
        val file: File,
        val markers: AlignEngine.Markers,
        val autoDots: List<AlignEngine.Dot>,
        val existingPhotoId: Long? = null,
        /** 测点编号（二维码内容）：扫码锁定后拍全景时，照片里可能读不到码，用扫到的码。 */
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
        repo.clearPending(file.absolutePath)
        val out = repo.process(file, zoneId)
        processing = false
        back()            // 关掉“未归类”页
        if (importQueue.isNotEmpty() && out is Outcome.Aligned) { importedOk++; lastImported = out.photoId; continueQueue() }
        else route(out)
    }

    fun newSeriesFrom(u: Outcome.Unclassified) {
        replaceTop(Screen.Register(u.file, u.markers, emptyList()))
    }

    fun discard(file: File) {
        viewModelScope.launch { repo.clearPending(file.absolutePath) }
        repo.discard(file); back(); continueQueue()
    }

    fun saveRegister(r: Screen.Register, name: String, dots: List<AlignEngine.Dot>,
                     kinds: Set<com.fieldalign.camera.align.Defects.Kind> = emptySet()) = viewModelScope.launch {
        if (r.existingPhotoId != null) {
            repo.setReference(r.existingPhotoId, dots)
            toast = "已设为参考照片"
            back()
        } else {
            repo.clearPending(r.file.absolutePath)
            val zid = repo.createZone(name.ifBlank { r.qrText?.let { repo.nameForQr(it) } ?: "未命名测点" }, r.file, r.markers, dots, r.qrText, kinds = kinds)
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

    // ---------------------------------------------------------------- 批量拍摄
    /** 批量建档时新测点的默认病害类型（相机页顶部可改）。 */
    private val prefs = app.getSharedPreferences("settings", android.content.Context.MODE_PRIVATE)
    var batchKinds by mutableStateOf(com.fieldalign.camera.align.Defects.parse(prefs.getString("batch_kinds", null)))
    fun updateBatchKinds(k: Set<com.fieldalign.camera.align.Defects.Kind>) {
        batchKinds = k; prefs.edit().putString("batch_kinds", com.fieldalign.camera.align.Defects.encode(k)).apply()
    }
    var batchQueued by mutableStateOf(0); private set
    var batchCreated by mutableStateOf(0); private set
    var batchMatched by mutableStateOf(0); private set
    var batchPending by mutableStateOf(0); private set
    /** 最近一张的结果（相机页底部一行提示）及其测点 id。 */
    var batchLast by mutableStateOf<String?>(null); private set
    var batchLastZone by mutableStateOf<Long?>(null); private set
    private val batchLock = kotlinx.coroutines.sync.Mutex()

    /** 第一次打开显示教程。 */
    var tutorialSeen by mutableStateOf(prefs.getBoolean("tutorial_seen", false)); private set
    fun markTutorialSeen() { tutorialSeen = true; prefs.edit().putBoolean("tutorial_seen", true).apply() }

    /** 生成靶标：下一次默认从上次最后一号往后接，避免重号。 */
    var targetPrefix by mutableStateOf(prefs.getString("target_prefix", "T") ?: "T"); private set
    fun nextTargetStart(prefix: String) = prefs.getInt("target_next_$prefix", 1)
    fun makeTargets(prefix: String, start: Int, count: Int, digits: Int, onReady: (File) -> Unit) = viewModelScope.launch {
        val codes = com.fieldalign.camera.data.CodeMap.codes(prefix, start, count, digits)
        val f = withContext(Dispatchers.IO) {
            runCatching {
                val dir = File(getApplication<Application>().cacheDir, "exports").apply { mkdirs() }
                com.fieldalign.camera.data.TargetSheet.pdf(codes, File(dir, "靶标_${codes.first()}-${codes.last()}.pdf"))
            }.getOrNull()
        }
        if (f == null) { toast = "生成失败"; return@launch }
        repo.codeMap.register(codes)
        targetPrefix = prefix
        prefs.edit().putString("target_prefix", prefix).putInt("target_next_$prefix", maxOf(nextTargetStart(prefix), start + count)).apply()
        onReady(f)
    }

    /** 编号对应表改动后：已建档的测点按表改名。 */
    fun applyCodeMap() = viewModelScope.launch {
        val n = repo.applyCodeMap()
        toast = if (n > 0) "已按对应表更新 $n 个测点名称" else "已保存"
        codeMapVersion++
    }
    /** 对应表内容版本号（界面刷新用）。 */
    var codeMapVersion by mutableStateOf(0)

    fun importCodeMap(uri: android.net.Uri) = viewModelScope.launch {
        val n = withContext(Dispatchers.IO) {
            runCatching {
                val text = getApplication<Application>().contentResolver.openInputStream(uri)!!.use { it.readBytes() }.let { b ->
                    val u = String(b, Charsets.UTF_8)
                    if ('\uFFFD' in u) String(b, charset("GBK")) else u          // Excel 另存的 CSV 可能是 GBK
                }
                repo.codeMap.importCsv(text)
            }.getOrNull()
        }
        if (n == null) { toast = "读取失败：请选择 CSV 文件"; return@launch }
        val r = repo.applyCodeMap()
        codeMapVersion++
        toast = "导入 $n 条" + if (r > 0) "，更新了 $r 个测点名称" else ""
    }

    fun resetBatch() { batchCreated = 0; batchMatched = 0; batchPending = 0; batchLast = null; batchLastZone = null }

    /** 批量拍摄：照片排队在后台逐张处理，相机可以接着拍。 */
    fun onBatchCaptured(file: File, mode: com.fieldalign.camera.data.CaptureMode, loc: com.fieldalign.camera.data.GeoFix?,
                        pose: com.fieldalign.camera.data.Pose?, forcedZoneId: Long? = null, newQr: String? = null) = viewModelScope.launch {
        batchQueued++
        val r = batchLock.withLock { repo.capture(file, mode, loc, pose, batchKinds, forcedZoneId, newQr) }
        batchQueued--
        when (r) {
            is com.fieldalign.camera.data.Batch.Created -> { batchCreated++; batchLast = "已建档：${r.name}"; batchLastZone = r.zoneId }
            is com.fieldalign.camera.data.Batch.Matched -> { batchMatched++; batchLastZone = r.zoneId
                val ph = repo.db.photos().get(r.photoId)
                val lv = ph?.conclusion?.let { Regex("^【(.+?)】").find(it)?.groupValues?.get(1) }
                // 二维码相同但画面和参考对不上：多半是同一个码贴到了别处（或拍的不是同一处），提醒复核
                val warn = if (ph?.status == "conflict" || ph?.status == "failed") " ⚠ 画面和参考不一致，请复核" else ""
                batchLast = "已归入：${r.name}（${r.how}）" + (lv?.let { " · $it" } ?: "") + warn }
            is com.fieldalign.camera.data.Batch.Pending -> { batchPending++; batchLast = "待确认：${r.reason}"; batchLastZone = null }
            is com.fieldalign.camera.data.Batch.Error -> { batchLast = r.message; batchLastZone = null }
        }
    }

    /** 打开一张待确认照片：给出按 图像 + 距离 排好序的候选，由人选择归入或新建。 */
    fun openPending(p: com.fieldalign.camera.data.Photo) = viewModelScope.launch {
        processing = true
        val loc = if (p.lat != null && p.lon != null) com.fieldalign.camera.data.GeoFix(p.lat, p.lon, p.locAcc ?: 30.0, p.takenAt) else null
        val pose = p.pitch?.let { com.fieldalign.camera.data.Pose(p.azimuth, it) }
        val out = repo.process(File(p.path), loc = loc, pose = pose, autoAssign = false)
        processing = false
        route(out)
    }

    /** 导出全部测点（电脑端后处理用）。 */
    var exporting by mutableStateOf(false); private set
    fun exportAll(onReady: (File) -> Unit) = viewModelScope.launch {
        exporting = true
        val f = runCatching { repo.exportAll() }.getOrNull()
        exporting = false
        if (f == null) toast = "没有可导出的测点" else onReady(f)
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
     * 供相机页按画面特征自动识别测点的取景签名（低分辨率）。
     * 有定位时只取附近的测点（以及还没有位置记录的老测点），远处背景相似的测点不参与，识别更快更准。
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

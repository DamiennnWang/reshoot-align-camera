package com.fieldalign.camera.data

import android.content.Context
import com.fieldalign.camera.align.AlignEngine
import com.fieldalign.camera.align.AlignEngine.Dot
import com.fieldalign.camera.align.AlignEngine.Markers
import com.fieldalign.camera.align.ImageIO
import com.fieldalign.camera.align.Measure
import com.fieldalign.camera.align.P
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.Mat
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/** 拍照后的处理结果，决定界面下一步去哪。 */
sealed class Outcome {
    /** 已对齐对比完成。 */
    data class Aligned(val photoId: Long) : Outcome()
    /** 新二维码或用户要新建系列：需要确认圆点、起名。 */
    data class NeedRegister(val file: File, val markers: Markers, val autoDots: List<Dot>, val qrText: String? = markers.qrText) : Outcome()
    /** 无码且无法唯一识别：让用户手选系列或新建。 */
    data class Unclassified(val file: File, val markers: Markers, val message: String,
                            val ranking: List<Pair<Zone, Int>>, val distances: Map<Long, Double> = emptyMap(),
                            val poseOk: Set<Long> = emptySet()) : Outcome()
    data class Error(val message: String) : Outcome()
}

class Repository(private val ctx: Context, val db: AppDb = AppDb.get(ctx)) {
    private val sigCache = ConcurrentHashMap<Long, AlignEngine.Signature>()   // zoneId → 参考照片识别签名

    fun newPhotoFile(): File {
        val dir = File(ctx.filesDir, "photos").apply { mkdirs() }
        return File(dir, "IMG_${System.currentTimeMillis()}.jpg")
    }

    /** 照片的拍摄时间：优先读 EXIF（导入的旧照片），读不到用文件时间。 */
    fun takenAt(file: File): Long = runCatching {
        // EXIF 时间是拍摄地的本地时间，通常不带时区；有时区偏移就用，没有就按手机当前时区解析
        val ex = androidx.exifinterface.media.ExifInterface(file.absolutePath)
        val t = ex.getAttribute(androidx.exifinterface.media.ExifInterface.TAG_DATETIME_ORIGINAL)
            ?: ex.getAttribute(androidx.exifinterface.media.ExifInterface.TAG_DATETIME) ?: return@runCatching null
        val off = ex.getAttribute(androidx.exifinterface.media.ExifInterface.TAG_OFFSET_TIME_ORIGINAL)
        val fmt = java.text.SimpleDateFormat("yyyy:MM:dd HH:mm:ss", java.util.Locale.US)
        if (off != null && off.matches(Regex("[+-]\\d{2}:\\d{2}"))) fmt.timeZone = java.util.TimeZone.getTimeZone("GMT$off")
        fmt.parse(t)?.time
    }.getOrNull() ?: file.lastModified()

    private fun resultDir(zoneId: Long, photoId: Long) = File(ctx.filesDir, "results/$zoneId/$photoId").apply { mkdirs() }

    // ------------------------------------------------------------- JSON
    fun markersToJson(m: Markers, dots: List<Dot>?): String = JSONObject().apply {
        put("qr_text", m.qrText ?: JSONObject.NULL)
        put("qr_corners", m.qrCorners?.let { c -> JSONArray(c.map { JSONArray(listOf(it.x, it.y)) }) } ?: JSONObject.NULL)
        put("work_w", m.workW); put("work_h", m.workH)
        put("dots", dots?.let { JSONArray(it.map { d -> JSONArray(listOf(d.x, d.y, d.d)) }) } ?: JSONObject.NULL)
        put("candidates", JSONArray(m.dotCandidates.take(30).map { JSONArray(listOf(it.x, it.y, it.d, it.score)) }))
    }.toString()

    fun markersFromJson(s: String): Pair<Markers, List<Dot>> {
        val o = JSONObject(s)
        val corners = if (o.isNull("qr_corners")) null else o.getJSONArray("qr_corners").let { a ->
            (0 until a.length()).map { P(a.getJSONArray(it).getDouble(0), a.getJSONArray(it).getDouble(1)) } }
        fun dots(key: String) = if (o.isNull(key)) emptyList() else o.getJSONArray(key).let { a ->
            (0 until a.length()).map { val e = a.getJSONArray(it)
                Dot(e.getDouble(0), e.getDouble(1), e.getDouble(2), if (e.length() > 3) e.getDouble(3) else 1.0) } }
        val m = Markers(if (o.isNull("qr_text")) null else o.getString("qr_text"), corners,
            dots("candidates"), o.getInt("work_w"), o.getInt("work_h"))
        return m to dots("dots")
    }

    private fun alignToJson(r: AlignEngine.AlignResult) = JSONObject().apply {
        put("status", r.status); put("message", r.message)
        put("H", r.h?.let { JSONArray(it.toList()) } ?: JSONObject.NULL)
        put("marker_rms", r.markerRms ?: JSONObject.NULL)
        put("feature_inliers", r.featureInliers); put("feature_coverage", r.featureCoverage)
        put("consistency_px", r.consistencyPx ?: JSONObject.NULL)
        put("dots_moved", JSONArray(r.dotsMoved))
        put("roi", r.roi?.let { JSONArray(it.map { p -> JSONArray(listOf(p.x, p.y)) }) } ?: JSONObject.NULL)
        put("k_new", r.kNew); put("k_ref", r.kRef)
    }.toString()

    // ------------------------------------------------------------- 处理流程
    /** 照片拍摄位置：相机页传入的 GPS 优先，否则读 EXIF。新建系列/手动归类时还会用到，按文件暂存。 */
    private val pendingLoc = ConcurrentHashMap<String, GeoFix>()
    private fun locOf(file: File, given: GeoFix?): GeoFix? =
        (given ?: pendingLoc[file.absolutePath] ?: Geo.fromExif(file))?.also { pendingLoc[file.absolutePath] = it }
    private val pendingPose = ConcurrentHashMap<String, Pose>()
    private fun poseOf(file: File, given: Pose?): Pose? = (given ?: pendingPose[file.absolutePath])?.also { pendingPose[file.absolutePath] = it }

    /**
     * 拍照后调用：识别 → 查库 → 对齐 → 对比 → 存档。
     * @param forcedZoneId 用户手动选的、或扫码后锁定的系列（大场景：先扫码锁定，再后退拍全景，照片里可以没有码）
     * @param newQr 扫到了库里没有的码并锁定：这张（全景）照片作为该码新系列的参考
     * @param loc 拍照时的手机定位
     */
    suspend fun process(file: File, forcedZoneId: Long? = null, hintZoneId: Long? = null,
                        newQr: String? = null, loc: GeoFix? = null, pose: Pose? = null): Outcome = withContext(Dispatchers.Default) {
        try {
            val work = ImageIO.loadWorkMat(file.absolutePath)
            val m = com.fieldalign.camera.align.QrFallback.detectMarkers(work)
            val here = locOf(file, loc)
            val ps = poseOf(file, pose)
            if (forcedZoneId != null) {
                val z = db.zones().get(forcedZoneId) ?: return@withContext Outcome.Error("系列不存在")
                return@withContext Outcome.Aligned(alignAndStore(z, file, work, m, if (hintZoneId == forcedZoneId) "扫码锁定" else "手动选择", here, ps))
            }
            // 照片里读到的码优先；读不到时用取景时扫码锁定的新码
            val qr = m.qrText ?: newQr
            if (qr != null) {
                val z = db.zones().byQr(qr)
                    ?: return@withContext Outcome.NeedRegister(file, m,
                        if (m.qrText == qr) AlignEngine.selectReferenceDots(m) else emptyList(), qr)
                return@withContext Outcome.Aligned(alignAndStore(z, file, work, m,
                    if (m.qrText != null) "二维码 $qr" else "扫码锁定 $qr", here, ps))
            }
            // 无码：按 位置 + 图像特征 识别
            val zones = db.zones().all().filter { it.refPhotoId != null }
            if (zones.isEmpty()) return@withContext Outcome.NeedRegister(file, m, emptyList())
            val dist = if (here == null) emptyMap() else
                zones.filter { it.lat != null && it.lon != null }.associate { it.id to Geo.distance(here.lat, here.lon, it.lat!!, it.lon!!) }
            val refs = zones.mapNotNull { z -> refSignature(z)?.let { z to it } }
            val sig = AlignEngine.signature(work)
            val idr = AlignEngine.identify(sig, refs)
            val ranking = idr.ranking.map { it.key to it.inliers }
            fun distNote(z: Zone) = dist[z.id]?.let { "，距 ${Geo.fmtDistance(it)}" } ?: ""
            if (idr.decision == "unique" && idr.best != null)
                return@withContext Outcome.Aligned(alignAndStore(idr.best, file, work, m, idr.message.dropLast(1) + distNote(idr.best) + "）", here, ps))
            // 位置 + 朝向辅助：只在定位半径内、镜头朝向也吻合的系列里再判断一次
            // （远处背景相似的系列、同一点位其他方向的系列都被排除），并放宽特征门槛
            if (here != null) {
                val r = Geo.nearRadius(here.acc)
                val near = refs.filter { (dist[it.first.id] ?: Double.MAX_VALUE) <= r &&
                    Geo.poseCompatible(ps?.azimuth, ps?.pitch, it.first.azimuth, it.first.pitch) }
                if (near.isNotEmpty()) {
                    val nearId = AlignEngine.identify(sig, near, minInliers = 20, minCoverage = 0.03)
                    if (nearId.decision == "unique" && nearId.best != null) {
                        val how = if (ps?.azimuth != null && nearId.best.azimuth != null) "位置 + 朝向 + 图像特征" else "位置 + 图像特征"
                        return@withContext Outcome.Aligned(alignAndStore(nearId.best, file, work, m,
                            "按${how}识别（附近 ${near.size} 个系列中唯一吻合，${nearId.ranking[0].inliers} 个匹配点${distNote(nearId.best)}）", here, ps))
                    }
                }
            }
            // 取景时已识别出系列：按放宽的覆盖率再确认一次，仍唯一指向它才采纳
            if (hintZoneId != null && idr.decision != "ambiguous") {
                val relaxed = AlignEngine.identify(sig, refs, minCoverage = 0.03)
                if (relaxed.decision == "unique" && relaxed.best?.id == hintZoneId)
                    return@withContext Outcome.Aligned(alignAndStore(relaxed.best, file, work, m, "按图像特征识别（取景时已匹配，${relaxed.ranking[0].inliers} 个匹配点）", here, ps))
            }
            val poseOk = zones.filter { Geo.poseCompatible(ps?.azimuth, ps?.pitch, it.azimuth, it.pitch) }.map { it.id }.toSet()
            Outcome.Unclassified(file, m, idr.message, ranking, dist, if (ps == null) emptySet() else poseOk)
        } catch (e: Throwable) {
            Outcome.Error("处理失败：${e.message ?: e.javaClass.simpleName}")
        }
    }

    private suspend fun refSignature(z: Zone): AlignEngine.Signature? {
        sigCache[z.id]?.let { return it }
        val ref = db.photos().get(z.refPhotoId ?: return null) ?: return null
        val sig = AlignEngine.signature(refWorkOf(ref.path))
        sigCache[z.id] = sig
        return sig
    }

    /** 实时引导用的低分辨率参考签名。 */
    private val guideCache = ConcurrentHashMap<Long, Pair<AlignEngine.Signature, String>>()
    suspend fun guideSignature(zoneId: Long): Pair<AlignEngine.Signature, String>? = withContext(Dispatchers.Default) {
        guideCache[zoneId]?.let { return@withContext it }
        val z = db.zones().get(zoneId) ?: return@withContext null
        val ref = db.photos().get(z.refPhotoId ?: return@withContext null) ?: return@withContext null
        val work = refWorkOf(ref.path)
        val sig = AlignEngine.signature(work, side = 640, nFeatures = 1500)
        (sig to ref.path).also { guideCache[zoneId] = it }
    }

    /** 参考照片解码后的工作图缓存（只读使用）：连续拍同一系列时不用每张都重新解码 4000px 原图。 */
    private val refWorkCache = object : LinkedHashMap<String, Mat>(4, 0.75f, true) {
        override fun removeEldestEntry(e: MutableMap.MutableEntry<String, Mat>?) = size > 2
    }
    private fun refWorkOf(path: String): Mat = synchronized(refWorkCache) {
        refWorkCache.getOrPut(path) { ImageIO.loadWorkMat(path) }
    }

    private suspend fun alignAndStore(z: Zone, file: File, work: Mat, m0: Markers, how: String, loc: GeoFix?, pose: Pose? = null): Long {
        val ref = db.photos().get(z.refPhotoId!!)!!
        val refWork = refWorkOf(ref.path)
        val (rm, refDots) = markersFromJson(ref.markersJson)
        val refKey = "${ref.id}:${ref.path.hashCode()}:${ref.markersJson.hashCode()}"
        // 参考有码、新照片没读出码：按特征摆正放大后再试一次，读出就走“靶标+特征”的完整流程
        val m = if (m0.qrText == null && rm.qrText != null) com.fieldalign.camera.align.QrFallback.recover(refWork, rm, work) ?: m0 else m0
        val useMarkers = rm.qrCorners != null && m.qrCorners != null && rm.qrText == m.qrText
        val res = if (useMarkers) AlignEngine.align(refWork, rm, refDots, work, m, refKey)
                  else AlignEngine.alignFeaturesOnly(refWork, work, refKey)
        val id = db.photos().insert(Photo(zoneId = z.id, path = file.absolutePath, takenAt = takenAt(file),
            status = res.status, message = res.message, markersJson = markersToJson(m, null),
            alignJson = alignToJson(res), refPhotoId = ref.id, how = how,
            lat = loc?.lat, lon = loc?.lon, locAcc = loc?.acc, azimuth = pose?.azimuth, pitch = pose?.pitch))
        pendingLoc.remove(file.absolutePath); pendingPose.remove(file.absolutePath)
        // 老系列没有位置/朝向：第一次带数据的成功复拍补上
        if (res.h != null && ((z.lat == null && loc != null) || (z.azimuth == null && pose?.azimuth != null)))
            db.zones().update(z.copy(lat = z.lat ?: loc?.lat, lon = z.lon ?: loc?.lon,
                azimuth = z.azimuth ?: pose?.azimuth, pitch = z.pitch ?: pose?.pitch))
        val h = res.h ?: return id
        val (aligned, valid) = AlignEngine.warpToRef(work, h, refWork.cols(), refWork.rows(), res.kNew, res.kRef)
        val excl = AlignEngine.comparisonMask(refWork.cols(), refWork.rows(), res, if (useMarkers) rm.qrCorners else null, refDots)
        val c = AlignEngine.compare(refWork, aligned, valid, excl, resRatio = AlignEngine.resolutionRatio(res))
        val dir = resultDir(z.id, id)
        val fa = File(dir, "aligned.jpg"); val fc = File(dir, "changes.jpg"); val fk = File(dir, "cracks.jpg")
        ImageIO.saveJpeg(aligned, fa)
        ImageIO.saveJpeg(AlignEngine.changesView(refWork, c), fc)
        // 裂缝参数 + 文字结论（比例尺来自参考照片里的二维码）
        val cr = measureCracks(z, rm, refWork, aligned, valid, excl, c, fk)
        db.photos().update(db.photos().get(id)!!.copy(changedRatio = c.changedRatio, changeRegions = c.regions.size,
            mmPerPx = cr.mmpp, crackCount = cr.ch.new.cracks.size, crackLenPx = cr.ch.new.totalPx, crackMaxWPx = cr.ch.new.maxWPx,
            crackGrowPx = cr.ch.growPx, crackJson = Measure.toJson(cr.ch, cr.mmpp, cr.level, fk.absolutePath),
            conclusion = "【${cr.level}】${cr.text}",
            similarity = c.similarity, lightRatio = c.lightRatio, lightRegions = c.lightRegions.size,
            glareRatio = c.glareRatio, quality = AlignEngine.qualityWarnings(c, work).joinToString("\n").ifEmpty { null },
            alignedPath = fa.absolutePath, changesPath = fc.absolutePath))
        return id
    }

    /** 参考照片的范围在新照片中占多大（面积比）。远小于 1：新照片比参考拍得更大范围。 */
    fun refAreaInNew(p: Photo, ref: Photo): Double? = runCatching {
        val o = JSONObject(p.alignJson ?: return null)
        if (o.isNull("H")) return null
        val h = com.fieldalign.camera.align.Homography(o.getJSONArray("H").let { a -> DoubleArray(9) { a.getDouble(it) } }).inverse()
        val (nm, _) = markersFromJson(p.markersJson); val (rm, _) = markersFromJson(ref.markersJson)
        val w = rm.workW.toDouble(); val hh = rm.workH.toDouble()
        val q = listOf(P(0.0, 0.0), P(w, 0.0), P(w, hh), P(0.0, hh)).map { h.apply(it) }
        com.fieldalign.camera.align.convexHullArea(q) / (nm.workW.toDouble() * nm.workH)
    }.getOrNull()

    class CrackResult(val ch: Measure.Change, val mmpp: Double?, val level: String, val text: String)

    /** 在对比区（去掉靶标、反光、无效区）里测裂缝、算比例、出结论，标注图写到 out。 */
    private fun measureCracks(z: Zone, rm: Markers, refWork: Mat, aligned: Mat, valid: Mat, excl: Mat?,
                              c: AlignEngine.CompareResult, out: File): CrackResult {
        val w = refWork.cols(); val h = refWork.rows()
        val m = Mat(); org.opencv.core.Core.compare(c.changeMask, org.opencv.core.Scalar(0x40.toDouble()), m, org.opencv.core.Core.CMP_NE)
        org.opencv.core.Core.bitwise_and(m, valid, m)
        if (excl != null) org.opencv.core.Core.bitwise_and(m, excl, m)
        val ch = Measure.compare(AlignEngine.gray(refWork), AlignEngine.gray(aligned), m)
        val sc = Measure.scaleFromQr(rm.qrCorners, z.qrMm ?: Measure.qrSizeMm(rm.qrText ?: z.qrText))
        val mmpp = sc?.let { s ->
            val ks = ch.new.cracks.ifEmpty { ch.ref.cracks }
            val at = if (ks.isEmpty()) P(rm.qrCorners!!.sumOf { it.x } / 4, rm.qrCorners.sumOf { it.y } / 4)
                     else P(ks.sumOf { it.cx * it.lengthPx } / ks.sumOf { it.lengthPx }, ks.sumOf { it.cy * it.lengthPx } / ks.sumOf { it.lengthPx })
            s.mmPerPx(at)
        }
        val (level, text) = Measure.conclude(ch, mmpp, c.lightRegions.size, c.lightRatio)
        ImageIO.saveJpeg(Measure.draw(aligned, ch, mmpp), out)
        return CrackResult(ch, mmpp, level, text)
    }

    /** 新建系列：照片成为参考照片。 */
    suspend fun createZone(name: String, file: File, m: Markers, dots: List<Dot>, qrText: String? = m.qrText,
                           given: GeoFix? = null, givenPose: Pose? = null): Long = withContext(Dispatchers.IO) {
        val loc = locOf(file, given); pendingLoc.remove(file.absolutePath)
        val pose = poseOf(file, givenPose); pendingPose.remove(file.absolutePath)
        val zid = db.zones().insert(Zone(qrText = qrText, name = name, createdAt = System.currentTimeMillis(), lat = loc?.lat, lon = loc?.lon,
            azimuth = pose?.azimuth, pitch = pose?.pitch, qrMm = qrText?.let { Measure.qrSizeMm(it) }))
        val pid = db.photos().insert(Photo(zoneId = zid, path = file.absolutePath, takenAt = takenAt(file),
            status = "reference", message = "参考照片", markersJson = markersToJson(m, dots),
            how = if (qrText != null && m.qrText == null) "扫码后拍全景" else "新建", lat = loc?.lat, lon = loc?.lon, locAcc = loc?.acc,
            azimuth = pose?.azimuth, pitch = pose?.pitch))
        db.zones().update(db.zones().get(zid)!!.copy(refPhotoId = pid))
        zid
    }

    /** 把某张照片设为参考（重新检测靶标，圆点由用户确认）。之后的照片都和它比。 */
    suspend fun setReference(photoId: Long, dots: List<Dot>) = withContext(Dispatchers.IO) {
        val p = db.photos().get(photoId) ?: return@withContext
        val (m, _) = markersFromJson(p.markersJson)
        db.photos().update(p.copy(status = "reference", message = "参考照片", markersJson = markersToJson(m, dots)))
        db.zones().get(p.zoneId)?.let { db.zones().update(it.copy(refPhotoId = photoId, lat = p.lat ?: it.lat, lon = p.lon ?: it.lon,
            azimuth = p.azimuth ?: it.azimuth, pitch = p.pitch ?: it.pitch)) }
        sigCache.remove(p.zoneId); guideCache.remove(p.zoneId)
    }

    /** 修改二维码打印尺寸：比例与尺寸成正比，已有照片的毫米值和文字结论一并更新。 */
    suspend fun setQrMm(id: Long, mm: Double) = withContext(Dispatchers.IO) {
        val z = db.zones().get(id) ?: return@withContext
        val old = z.qrMm ?: Measure.qrSizeMm(z.qrText)
        db.zones().update(z.copy(qrMm = mm))
        for (p in db.photos().listForZone(id)) {
            val mmpp = p.mmPerPx?.let { it * mm / old } ?: continue
            val st = p.crackJson?.let { Measure.stats(it) }
            val text = st?.let { s -> Measure.conclude(s, mmpp, p.lightRegions ?: 0, p.lightRatio ?: 0.0) }
            db.photos().update(p.copy(mmPerPx = mmpp, conclusion = text?.let { "【${it.first}】${it.second}" } ?: p.conclusion))
        }
    }

    suspend fun renameZone(id: Long, name: String) = withContext(Dispatchers.IO) {
        db.zones().get(id)?.let { db.zones().update(it.copy(name = name)) }
    }

    data class PairResult(val baseImagePath: String, val changesPath: String, val changedRatio: Double,
                          val regions: Int, val similarity: Double, val lightRatio: Double = 0.0, val lightRegions: Int = 0,
                          val warnings: List<String> = emptyList(), val crackPath: String? = null,
                          val conclusion: String? = null, val mmPerPx: Double? = null)

    /**
     * 任选两张对比：同一系列的照片都已变换到参考照片坐标，直接比较两张对齐图即可
     * （例如本月对上月，而不总是对最早的参考）。结果缓存在 results/<系列>/<照片>/vs_<基准>.jpg。
     */
    suspend fun compareTwo(baseId: Long, photoId: Long): PairResult? = withContext(Dispatchers.Default) {
        val p = db.photos().get(photoId) ?: return@withContext null
        val b = db.photos().get(baseId) ?: return@withContext null
        val ref = db.photos().get(p.refPhotoId ?: return@withContext null) ?: return@withContext null
        if (b.id != ref.id && b.refPhotoId != ref.id) return@withContext null   // 不在同一参考坐标下
        val refWork = refWorkOf(ref.path)
        val w = refWork.cols(); val h = refWork.rows()
        class Frame(val img: Mat, val valid: Mat, val roi: List<P>?, val path: String)
        fun frameOf(x: Photo): Frame? {
            if (x.id == ref.id) return Frame(refWork, Mat(h, w, org.opencv.core.CvType.CV_8U, org.opencv.core.Scalar(255.0)), null, ref.path)
            val ap = x.alignedPath ?: return null
            val o = JSONObject(x.alignJson ?: return null)
            if (o.isNull("H")) return null
            val hm = com.fieldalign.camera.align.Homography(o.getJSONArray("H").let { a -> DoubleArray(9) { a.getDouble(it) } })
            val roi = if (o.isNull("roi")) null else o.getJSONArray("roi").let { a ->
                (0 until a.length()).map { P(a.getJSONArray(it).getDouble(0), a.getJSONArray(it).getDouble(1)) } }
            val (m, _) = markersFromJson(x.markersJson)
            return Frame(ImageIO.loadWorkMat(ap),
                AlignEngine.validMask(m.workW, m.workH, hm, w, h, o.optDouble("k_new", 0.0), o.optDouble("k_ref", 0.0)), roi, ap)
        }
        val fb = frameOf(b) ?: return@withContext null
        val fp = frameOf(p) ?: return@withContext null
        val valid = Mat(); org.opencv.core.Core.bitwise_and(fb.valid, fp.valid, valid)
        val mask = AlignEngine.roiMask(w, h, fb.roi)
        org.opencv.core.Core.bitwise_and(mask, AlignEngine.roiMask(w, h, fp.roi), mask)
        val (rm, refDots) = markersFromJson(ref.markersJson)
        if (rm.qrCorners != null) org.opencv.core.Core.bitwise_and(mask, AlignEngine.markerMask(w, h, rm.qrCorners, refDots), mask)
        val c = AlignEngine.compare(fb.img, fp.img, valid, mask)
        val out = File(resultDir(p.zoneId, p.id), "vs_${b.id}.jpg")
        ImageIO.saveJpeg(AlignEngine.changesView(fb.img, c), out)
        val z = db.zones().get(p.zoneId)!!
        val kOut = File(resultDir(p.zoneId, p.id), "cracks_vs_${b.id}.jpg")
        val cr = measureCracks(z, rm, fb.img, fp.img, valid, mask, c, kOut)
        PairResult(fb.path, out.absolutePath, c.changedRatio, c.regions.size, c.similarity, c.lightRatio, c.lightRegions.size,
            AlignEngine.qualityWarnings(c), kOut.absolutePath, "【${cr.level}】${cr.text}", cr.mmpp)
    }

    /**
     * 导出整个系列为 ZIP：report.csv（Excel 可直接打开）+ 每张照片的原图、对齐图、变化图。
     * 返回 ZIP 文件（在应用私有目录 exports/ 下，通过分享发出）。
     */
    suspend fun exportZone(zoneId: Long): File? = withContext(Dispatchers.IO) {
        val z = db.zones().get(zoneId) ?: return@withContext null
        val photos = db.photos().listForZone(zoneId).sortedBy { it.takenAt }
        val dir = File(ctx.filesDir, "exports").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() }          // 只保留最新一次导出
        val safe = z.name.replace(Regex("[\\\\/:*?\"<>|\\s]"), "_")
        val out = File(dir, "${safe}_${java.text.SimpleDateFormat("yyyyMMdd_HHmm", java.util.Locale.US).format(java.util.Date())}.zip")
        val fmt = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.CHINA)
        java.util.zip.ZipOutputStream(out.outputStream().buffered()).use { zip ->
            fun put(name: String, f: File?) {
                if (f == null || !f.exists()) return
                zip.putNextEntry(java.util.zip.ZipEntry(name)); f.inputStream().use { it.copyTo(zip) }; zip.closeEntry()
            }
            val csv = StringBuilder("\uFEFF")   // BOM：Excel 识别 UTF-8 中文
            csv.append("序号,拍摄时间,是否参考,对齐状态,说明,归类方式,结构变化处数,结构变化占比%,明暗变化处数,明暗变化占比%,反光未比较%,整体相似度,拍摄质量提示,特征匹配点,靶标残差px,畸变k新,畸变k参考,纬度,经度,定位精度m,镜头朝向°,俯仰°,毫米/像素,裂缝条数,裂缝总长,最大宽度,新增长度,单位,文字结论,原图,对齐图,变化图,裂缝图\n")
            photos.forEachIndexed { i, p ->
                val n = "%03d".format(i + 1)
                val isRef = p.id == z.refPhotoId
                val a = p.alignJson?.let { runCatching { JSONObject(it) }.getOrNull() }
                fun q(v: String) = "\"" + v.replace("\"", "\"\"") + "\""
                csv.append(listOf(n, fmt.format(java.util.Date(p.takenAt)), if (isRef) "是" else "", if (isRef) "reference" else p.status,
                    q(p.message), q(p.how ?: ""), p.changeRegions?.toString() ?: "", p.changedRatio?.let { "%.3f".format(it * 100) } ?: "",
                    p.lightRegions?.toString() ?: "", p.lightRatio?.let { "%.3f".format(it * 100) } ?: "",
                    p.glareRatio?.let { "%.2f".format(it * 100) } ?: "",
                    p.similarity?.let { "%.3f".format(it) } ?: "", q(p.quality?.replace("\n", "；") ?: ""), a?.optInt("feature_inliers")?.toString() ?: "",
                    a?.let { if (it.isNull("marker_rms")) "" else "%.2f".format(it.getDouble("marker_rms")) } ?: "",
                    a?.optDouble("k_new", 0.0)?.let { "%.3f".format(it) } ?: "", a?.optDouble("k_ref", 0.0)?.let { "%.3f".format(it) } ?: "",
                    p.lat?.let { "%.6f".format(it) } ?: "", p.lon?.let { "%.6f".format(it) } ?: "", p.locAcc?.let { "%.0f".format(it) } ?: "",
                    p.azimuth?.let { "%.0f".format(it) } ?: "", p.pitch?.let { "%.0f".format(it) } ?: "",
                    p.mmPerPx?.let { "%.4f".format(it) } ?: "", p.crackCount?.toString() ?: "",
                    p.crackLenPx?.let { "%.1f".format(it * (p.mmPerPx ?: 1.0)) } ?: "",
                    p.crackMaxWPx?.let { "%.2f".format(it * (p.mmPerPx ?: 1.0)) } ?: "",
                    p.crackGrowPx?.let { "%.1f".format(it * (p.mmPerPx ?: 1.0)) } ?: "",
                    if (p.crackLenPx == null) "" else if (p.mmPerPx != null) "mm" else "px", q(p.conclusion ?: ""),
                    "photos/${n}_original.jpg", if (p.alignedPath != null) "photos/${n}_aligned.jpg" else "",
                    if (p.changesPath != null) "photos/${n}_changes.jpg" else "",
                    if (p.crackJson != null) "photos/${n}_cracks.jpg" else "").joinToString(",")).append("\n")
                put("photos/${n}_original.jpg", File(p.path))
                put("photos/${n}_aligned.jpg", p.alignedPath?.let { File(it) })
                put("photos/${n}_changes.jpg", p.changesPath?.let { File(it) })
                put("photos/${n}_cracks.jpg", p.crackJson?.let { runCatching { JSONObject(it).optString("image") }.getOrNull() }?.takeIf { it.isNotEmpty() }?.let { File(it) })
            }
            zip.putNextEntry(java.util.zip.ZipEntry("report.csv")); zip.write(csv.toString().toByteArray()); zip.closeEntry()
        }
        out
    }

    suspend fun deletePhoto(id: Long) = withContext(Dispatchers.IO) {
        val p = db.photos().get(id) ?: return@withContext
        val z = db.zones().get(p.zoneId)
        if (z?.refPhotoId == id) return@withContext   // 参考照片不能直接删，先换参考
        File(p.path).delete(); p.alignedPath?.let { File(it).parentFile?.deleteRecursively() }
        db.photos().delete(id)
    }

    suspend fun deleteZone(id: Long) = withContext(Dispatchers.IO) {
        db.photos().listForZone(id).forEach { File(it.path).delete() }
        File(ctx.filesDir, "results/$id").deleteRecursively()
        db.photos().listForZone(id).forEach { db.photos().delete(it.id) }
        db.zones().delete(id)
        sigCache.remove(id); guideCache.remove(id)
    }

    fun discard(file: File) { pendingLoc.remove(file.absolutePath); pendingPose.remove(file.absolutePath); file.delete() }
}

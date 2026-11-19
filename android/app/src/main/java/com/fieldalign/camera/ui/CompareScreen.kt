package com.fieldalign.camera.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.fieldalign.camera.AppViewModel
import com.fieldalign.camera.data.Photo
import kotlinx.coroutines.launch
import org.json.JSONObject

private val MODES = listOf("滑动", "叠加", "闪烁", "变化", "裂缝", "原图")

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CompareScreen(vm: AppViewModel, photoId: Long) {
    var photo by remember { mutableStateOf<Photo?>(null) }
    var refPhoto by remember { mutableStateOf<Photo?>(null) }
    LaunchedEffect(photoId) {
        photo = vm.repo.db.photos().get(photoId)
        refPhoto = photo?.refPhotoId?.let { vm.repo.db.photos().get(it) }
    }
    val p = photo
    // 对比基准：默认是参考照片，也可以选同系列的其他照片（如上一次）
    var siblings by remember { mutableStateOf<List<Photo>>(emptyList()) }
    var baseId by remember { mutableStateOf<Long?>(null) }
    var pair by remember { mutableStateOf<com.fieldalign.camera.data.Repository.PairResult?>(null) }
    var pairBusy by remember { mutableStateOf(false) }
    var baseMenu by remember { mutableStateOf(false) }
    LaunchedEffect(p?.id) {
        val pp = p ?: return@LaunchedEffect
        siblings = vm.repo.db.photos().listForZone(pp.zoneId)
            .filter { it.id != pp.id && (it.id == pp.refPhotoId || (it.refPhotoId == pp.refPhotoId && it.alignedPath != null)) }
            .sortedByDescending { it.takenAt }
        baseId = pp.refPhotoId
    }
    LaunchedEffect(baseId) {
        val pp = p ?: return@LaunchedEffect
        val b = baseId ?: return@LaunchedEffect
        if (b == pp.refPhotoId) { pair = null; return@LaunchedEffect }
        pairBusy = true
        pair = vm.repo.compareTwo(b, pp.id)
        pairBusy = false
    }
    val basePath = pair?.baseImagePath ?: refPhoto?.path
    val changesPath = pair?.changesPath ?: p?.changesPath
    val refBmp = rememberBitmap(basePath) { basePath?.let { vm.loadBitmap(it) } }
    val alignedBmp = rememberBitmap(p?.alignedPath) { p?.alignedPath?.let { vm.loadBitmap(it) } }
    val changesBmp = rememberBitmap(changesPath) { changesPath?.let { vm.loadBitmap(it) } }
    val ctx = androidx.compose.ui.platform.LocalContext.current
    val rawBmp = rememberBitmap(p?.path) { p?.path?.let { vm.loadBitmap(it) } }
    val crackPath = pair?.crackPath ?: p?.crackJson?.let { runCatching { JSONObject(it).optString("image") }.getOrNull() }?.takeIf { it.isNotEmpty() }
    val crackBmp = rememberBitmap(crackPath) { crackPath?.let { vm.loadBitmap(it) } }
    val conclusion = pair?.conclusion ?: p?.conclusion
    val mmpp = pair?.mmPerPx ?: p?.mmPerPx

    var qualityDismissed by remember { mutableStateOf(false) }
    var wideDismissed by remember { mutableStateOf(false) }
    val scope = androidx.compose.runtime.rememberCoroutineScope()
    var mode by remember { mutableIntStateOf(0) }
    var split by remember { mutableFloatStateOf(0.5f) }
    var blend by remember { mutableFloatStateOf(0.5f) }
    var blinkNew by remember { mutableStateOf(false) }
    LaunchedEffect(mode) { while (mode == 2) { kotlinx.coroutines.delay(600); blinkNew = !blinkNew } }

    Scaffold(containerColor = Page, topBar = {
        AppTopBar("对比结果", p?.let { fmtTime(it.takenAt) + (it.how?.let { h -> " · $h" } ?: "") }, onBack = { vm.back() },
            actions = {
                if (p?.alignedPath != null) IconButton(onClick = {
                    shareResults(ctx, listOfNotNull(changesPath, p.alignedPath, basePath),
                        "对比结果：${p.message}\n拍摄 ${fmtTime(p.takenAt)}" +
                            ((pair?.changedRatio ?: p.changedRatio)?.let { "\n变化 %.2f%%".format(it * 100) } ?: ""))
                }) { Icon(Icons.Default.Share, "分享") }
            })
    }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).verticalScroll(rememberScrollState())) {
            if (p == null) return@Column
            if (p.alignedPath == null) {
                AppCard(Modifier.fillMaxWidth().padding(12.dp)) {
                    Column(Modifier.padding(16.dp)) {
                        StatusChip(p.status)
                        Text(p.message, modifier = Modifier.padding(top = 8.dp))
                        Text("可以换个角度重拍，或在系列里把这张设为新的参考照片。", color = Color.Gray, fontSize = 13.sp,
                            modifier = Modifier.padding(top = 8.dp))
                        Row(Modifier.padding(top = 10.dp)) {
                            androidx.compose.material3.Button(onClick = {
                                scope.launch { vm.repo.deletePhoto(p.id); vm.replaceTop(com.fieldalign.camera.Screen.Camera(p.zoneId)) }
                            }, colors = androidx.compose.material3.ButtonDefaults.buttonColors(containerColor = Amber, contentColor = Ink)) { Text("删除并重拍") }
                            Spacer(Modifier.width(8.dp))
                            androidx.compose.material3.TextButton(onClick = { scope.launch { vm.repo.deletePhoto(p.id); vm.back() } }) { Text("删除这张") }
                        }
                    }
                }
                rawBmp?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxWidth(), contentScale = ContentScale.FillWidth) }
                return@Column
            }
            // 拍摄质量有问题（反光、模糊、偏暗）：放在最上面，现场就能决定要不要重拍
            if (!p.quality.isNullOrBlank() && !qualityDismissed) {
                Card(Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, top = 8.dp),
                    colors = androidx.compose.material3.CardDefaults.cardColors(containerColor = Color(0xFFFFF3E0))) {
                    Column(Modifier.padding(14.dp)) {
                        Text("拍摄质量提示", fontWeight = FontWeight.SemiBold, color = Color(0xFFB45309))
                        p.quality.split("\n").forEach { Text("• $it", fontSize = 13.sp, modifier = Modifier.padding(top = 4.dp)) }
                        Row(Modifier.padding(top = 8.dp)) {
                            androidx.compose.material3.Button(onClick = {
                                scope.launch { vm.repo.deletePhoto(p.id); vm.replaceTop(com.fieldalign.camera.Screen.Camera(p.zoneId)) }
                            }, colors = androidx.compose.material3.ButtonDefaults.buttonColors(containerColor = Amber)) { Text("删除并重拍") }
                            Spacer(Modifier.width(8.dp))
                            androidx.compose.material3.TextButton(onClick = { qualityDismissed = true }) { Text("保留这张") }
                        }
                    }
                }
            }
            // 比参考拍得更大范围（参考是特写、这次后退拍了全景）：建议换参考，以后都按全景比
            val refShare = remember(p.id, refPhoto?.id) { refPhoto?.let { vm.repo.refAreaInNew(p, it) } }
            if (refShare != null && refShare < 0.5 && !wideDismissed) {
                AppCard(Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, top = 8.dp)) {
                    Column(Modifier.padding(14.dp)) {
                        Text("这张比参考照片拍得更大范围", fontWeight = FontWeight.SemiBold, color = Info)
                        Text("参考照片只占这张画面的 ${(refShare * 100).toInt()}%，对比只在参考范围内进行。如果以后都按这个范围拍，建议把这张设为新的参考照片。",
                            fontSize = 13.sp, modifier = Modifier.padding(top = 4.dp))
                        Row(Modifier.padding(top = 8.dp)) {
                            androidx.compose.material3.Button(onClick = { vm.startSetReference(p.id) },
                                colors = androidx.compose.material3.ButtonDefaults.buttonColors(containerColor = Ink)) { Text("设为参考照片") }
                            Spacer(Modifier.width(8.dp))
                            androidx.compose.material3.TextButton(onClick = { wideDismissed = true }) { Text("暂不") }
                        }
                    }
                }
            }
            conclusion?.let { ConclusionCard(it) }
            HeadlineCard(pair?.regions ?: p.changeRegions ?: 0, pair?.changedRatio ?: p.changedRatio ?: 0.0,
                pair?.lightRegions ?: p.lightRegions ?: 0, p.status)
            SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp)) {
                MODES.forEachIndexed { i, label ->
                    SegmentedButton(selected = mode == i, onClick = { mode = i },
                        shape = SegmentedButtonDefaults.itemShape(i, MODES.size)) { Text(label, fontSize = 13.sp) }
                }
            }
            // 对比基准选择
            Box(Modifier.padding(horizontal = 12.dp)) {
                val base = siblings.firstOrNull { it.id == baseId }
                androidx.compose.material3.AssistChip(onClick = { baseMenu = true },
                    label = { Text("对比基准：" + (if (baseId == p.refPhotoId) "参考照片" else "") + " " + fmtTime(base?.takenAt) +
                        if (pairBusy) "（计算中…）" else "") })
                androidx.compose.material3.DropdownMenu(baseMenu, { baseMenu = false }) {
                    siblings.forEach { s ->
                        androidx.compose.material3.DropdownMenuItem(
                            text = { Text((if (s.id == p.refPhotoId) "参考照片  " else "") + fmtTimeSec(s.takenAt)) },
                            onClick = { baseMenu = false; baseId = s.id })
                    }
                }
            }
            val ratio = refBmp?.let { it.width.toFloat() / it.height } ?: 0.75f
            ZoomBox(Modifier.fillMaxWidth().aspectRatio(ratio).background(Color.Black)) {
                Box(Modifier.fillMaxWidth().aspectRatio(ratio)) {
                    when (mode) {
                        0 -> { // 滑动：左边参考，右边新照片
                            refBmp?.let { Img(it) }
                            alignedBmp?.let { b ->
                                Image(b.asImageBitmap(), null, Modifier.fillMaxSize().drawWithContent {
                                    clipRect(left = size.width * split) { this@drawWithContent.drawContent() }
                                }, contentScale = ContentScale.FillBounds)
                            }
                            Canvas(Modifier.fillMaxSize()) {
                                drawLine(Amber, Offset(size.width * split, 0f), Offset(size.width * split, size.height), 4f)
                            }
                        }
                        1 -> { refBmp?.let { Img(it) }; alignedBmp?.let { Img(it, blend) } }
                        2 -> { (if (blinkNew) alignedBmp else refBmp)?.let { Img(it) } }
                        3 -> changesBmp?.let { Img(it) }
                        4 -> crackBmp?.let { Img(it) }
                        else -> rawBmp?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Fit) }
                    }
                }
            }
            when (mode) {
                0 -> LabeledSlider("左：参考  右：新照片", split) { split = it }
                1 -> LabeledSlider("新照片透明度", blend) { blend = it }
                2 -> Text(if (blinkNew) "新照片" else "参考照片", Modifier.padding(12.dp), color = Color.Gray)
                4 -> Text("黄色 = 原有裂缝；红色 = 比基准新增的部分；蓝色 = 基准里有、这次没检出（修补/遮挡/不清）。标注：编号、长度 L、最大宽度 W" +
                    (mmpp?.let { "。比例 1 像素 ≈ %.3f mm（由参考照片中的二维码换算）".format(it) } ?: "。参考照片里没有二维码，以像素表示"),
                    Modifier.padding(12.dp), color = Color.Gray, fontSize = 13.sp)
                3 -> Text("红色/黄框 = 结构变化（裂缝、剥落等）；浅蓝/蓝框 = 明暗变化（阴影、水渍、光照），不计入变化率；灰白 = 反光/过曝，未比较。只比较重叠且在同一平面上的区域", Modifier.padding(12.dp), color = Color.Gray, fontSize = 13.sp)
                else -> Text("拍摄的原始照片（未对齐）", Modifier.padding(12.dp), color = Color.Gray, fontSize = 13.sp)
            }
            Text("双指缩放、拖动查看细节，双击复位", Modifier.padding(horizontal = 12.dp), color = Color.Gray, fontSize = 12.sp)
            pair?.let { pr ->
                AppCard(Modifier.fillMaxWidth().padding(horizontal = 12.dp)) {
                    Column(Modifier.padding(16.dp)) {
                        Text("与所选基准对比", fontWeight = FontWeight.SemiBold)
                        Line("结构变化", "${pr.regions} 处，占对比区 %.2f%%".format(pr.changedRatio * 100))
                        if (pr.lightRegions > 0) Line("明暗变化", "${pr.lightRegions} 处，%.2f%%".format(pr.lightRatio * 100))
                        Line("整体相似度", "%.2f".format(pr.similarity))
                        pr.warnings.forEach { Text("⚠ $it", color = Bad, fontSize = 13.sp, modifier = Modifier.padding(top = 4.dp)) }
                    }
                }
            }
            InfoCard(p, refPhoto)
        }
    }
}

/** 用系统分享面板发送结果图片和说明文字。 */
private fun shareResults(ctx: android.content.Context, paths: List<String>, text: String) {
    val uris = ArrayList(paths.distinct().map { java.io.File(it) }.filter { it.exists() }.map {
        androidx.core.content.FileProvider.getUriForFile(ctx, ctx.packageName + ".files", it) })
    val i = android.content.Intent(android.content.Intent.ACTION_SEND_MULTIPLE).apply {
        type = "image/jpeg"
        putParcelableArrayListExtra(android.content.Intent.EXTRA_STREAM, uris)
        putExtra(android.content.Intent.EXTRA_TEXT, text)
        // 放进 ClipData，读取授权才会同时传给系统分享面板（预览）和接收方
        if (uris.isNotEmpty()) clipData = android.content.ClipData.newRawUri("", uris[0]).also { c ->
            uris.drop(1).forEach { c.addItem(android.content.ClipData.Item(it)) } }
        addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    ctx.startActivity(android.content.Intent.createChooser(i, "分享对比结果"))
}

@Composable
private fun Img(b: Bitmap, alpha: Float = 1f) =
    Image(b.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.FillBounds, alpha = alpha)

@Composable
private fun LabeledSlider(label: String, v: Float, on: (Float) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, fontSize = 12.sp, color = Color.Gray)
        Spacer(Modifier.width(8.dp))
        Slider(v, on, Modifier.weight(1f))
    }
}

@Composable
private fun InfoCard(p: Photo, ref: Photo?) {
    val a = p.alignJson?.let { runCatching { JSONObject(it) }.getOrNull() }
    AppCard(Modifier.fillMaxWidth().padding(12.dp)) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                StatusChip(p.status); Spacer(Modifier.width(8.dp))
                Text(p.how ?: "", fontSize = 13.sp, color = Color.Gray)
            }
            Text(p.message, Modifier.padding(top = 8.dp), fontSize = 14.sp)
            Spacer(Modifier.padding(4.dp))
            Line("拍摄时间", fmtTime(p.takenAt))
            Line("参考照片", fmtTime(ref?.takenAt))
            p.changedRatio?.let { Line("结构变化", "${p.changeRegions ?: 0} 处，占对比区 %.2f%%".format(it * 100)) }
            p.lightRatio?.let { if ((p.lightRegions ?: 0) > 0) Line("明暗变化", "${p.lightRegions} 处，%.2f%%（阴影、水渍等）".format(it * 100)) }
            p.similarity?.let {
                Line("整体相似度", "%.2f".format(it))
                if (it < 0.5) Text("两张照片整体差异很大：可能拍的不是同一位置，或表面整体发生了变化（翻新、积水、遮挡等）。",
                    color = Bad, fontSize = 13.sp, modifier = Modifier.padding(top = 4.dp))
            }
            p.glareRatio?.let { if (it > 0.005) Line("反光/过曝", "%.1f%%（未参与比较）".format(it * 100)) }
            if (p.lat != null && p.lon != null) {
                val d = if (ref?.lat != null && ref.lon != null)
                    "，距参考拍摄点 " + com.fieldalign.camera.data.Geo.fmtDistance(com.fieldalign.camera.data.Geo.distance(p.lat, p.lon, ref.lat, ref.lon)) else ""
                Line("拍摄位置", "%.5f, %.5f（±%.0f 米）%s".format(p.lat, p.lon, p.locAcc ?: 0.0, d))
            }
            a?.let {
                if (!it.isNull("marker_rms")) Line("靶标残差", "%.2f 像素".format(it.getDouble("marker_rms")))
                Line("特征匹配点", "${it.optInt("feature_inliers")}（覆盖 %.0f%%）".format(it.optDouble("feature_coverage") * 100))
                if (!it.isNull("consistency_px")) Line("靶标/特征偏差", "%.1f 像素".format(it.getDouble("consistency_px")))
            }
        }
    }
}

@Composable
private fun Line(k: String, v: String) {
    Row(Modifier.padding(vertical = 2.dp)) {
        Text(k, Modifier.width(96.dp), color = Color.Gray, fontSize = 13.sp)
        Text(v, fontSize = 13.sp, fontWeight = FontWeight.Medium)
    }
}

/** 结论卡：一眼看到“有没有变化、变化多大”。 */
@Composable
private fun HeadlineCard(regions: Int, ratio: Double, light: Int, status: String) {
    val col = if (regions > 0) Warn else Good
    AppCard(Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, top = 10.dp)) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(52.dp).background(col.copy(alpha = 0.12f), androidx.compose.foundation.shape.CircleShape),
                contentAlignment = Alignment.Center) {
                Text(if (regions > 0) "$regions" else "✓", color = col, fontSize = 22.sp, fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Text(if (regions > 0) "发现 $regions 处结构变化" else "未发现结构变化", fontWeight = FontWeight.SemiBold, fontSize = 17.sp)
                Text("变化面积占对比区 %.2f%%".format(ratio * 100) + if (light > 0) " · 明暗变化 $light 处（不计入）" else "",
                    color = Muted, fontSize = 13.sp)
            }
            StatusChip(status)
        }
    }
}

/** 文字结论卡（裂缝参数 + 判定）。文本格式：【等级】正文。 */
@Composable
private fun ConclusionCard(text: String) {
    val level = Regex("^【(.+?)】").find(text)?.groupValues?.get(1) ?: ""
    val body = text.removePrefix("【$level】")
    val col = when (level) { "稳定", "无裂缝" -> Good; "延长", "变宽" -> Warn; else -> Bad }
    AppCard(Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, top = 10.dp)) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("裂缝结论", fontWeight = FontWeight.SemiBold, fontSize = 16.sp)
                Spacer(Modifier.width(8.dp))
                if (level.isNotEmpty()) Pill(level, col, filled = true)
            }
            Text(body, fontSize = 14.sp, lineHeight = 21.sp, modifier = Modifier.padding(top = 6.dp))
        }
    }
}

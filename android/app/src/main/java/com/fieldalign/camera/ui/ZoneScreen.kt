package com.fieldalign.camera.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.fieldalign.camera.AppViewModel
import com.fieldalign.camera.Screen
import com.fieldalign.camera.data.Photo
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ZoneScreen(vm: AppViewModel, zoneId: Long) {
    val zone by vm.repo.db.zones().observe(zoneId).collectAsState(initial = null)
    val photos by vm.repo.db.photos().forZone(zoneId).collectAsState(initial = emptyList())
    val scope = rememberCoroutineScope()
    val ctx = androidx.compose.ui.platform.LocalContext.current
    var menu by remember { mutableStateOf(false) }
    var renaming by remember { mutableStateOf(false) }
    var sizing by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }
    var editKinds by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            AppTopBar(zone?.name ?: "", zone?.qrText?.let { "二维码 $it" } ?: "无码 · 按位置 + 图像特征识别", onBack = { vm.back() },
                actions = {
                    IconButton(onClick = { menu = true }) { Icon(Icons.Default.MoreVert, null) }
                    DropdownMenu(menu, { menu = false }) {
                        DropdownMenuItem(text = { Text("重命名") }, leadingIcon = { Icon(Icons.Default.Edit, null) },
                            onClick = { menu = false; renaming = true })
                        DropdownMenuItem(text = { Text("病害类型") }, leadingIcon = { Icon(Icons.Default.Edit, null) },
                            onClick = { menu = false; editKinds = true })
                        DropdownMenuItem(text = { Text("二维码尺寸（毫米换算）") }, leadingIcon = { Icon(Icons.Default.Edit, null) },
                            onClick = { menu = false; sizing = true })
                        DropdownMenuItem(text = { Text("导出全部（ZIP + 表格）") }, leadingIcon = { Icon(Icons.Default.Share, null) },
                            onClick = {
                                menu = false
                                scope.launch {
                                    val f = vm.repo.exportZone(zoneId) ?: return@launch
                                    val uri = androidx.core.content.FileProvider.getUriForFile(ctx, ctx.packageName + ".files", f)
                                    val i = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                                        type = "application/zip"
                                        putExtra(android.content.Intent.EXTRA_STREAM, uri)
                                        putExtra(android.content.Intent.EXTRA_SUBJECT, f.name)
                                        clipData = android.content.ClipData.newRawUri("", uri)
                                        addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                    }
                                    ctx.startActivity(android.content.Intent.createChooser(i, "导出系列"))
                                }
                            })
                        DropdownMenuItem(text = { Text("删除整个系列") }, leadingIcon = { Icon(Icons.Default.Delete, null) },
                            onClick = { menu = false; confirmDelete = true })
                    }
                },
            )
        },
        containerColor = Page,
        floatingActionButton = {
            ExtendedFloatingActionButton(onClick = { vm.go(Screen.Camera(zoneId)) }, containerColor = Amber,
                icon = { Icon(Icons.Default.PhotoCamera, null) }, text = { Text("复拍") })
        },
    ) { pad ->
        LazyColumn(Modifier.fillMaxSize().padding(pad), contentPadding = PaddingValues(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)) {
            val trend = photos.filter { it.id != zone?.refPhotoId && it.changedRatio != null }.sortedBy { it.takenAt }
            vm.reanalyzing?.let { msg -> item(key = "busy") {
                Column(Modifier.fillMaxWidth()) {
                    androidx.compose.material3.LinearProgressIndicator(Modifier.fillMaxWidth(), color = Amber)
                    Text(msg, fontSize = 12.sp, color = Muted, modifier = Modifier.padding(top = 4.dp))
                }
            } }
            item(key = "summary") { zone?.let { SummaryCard(it, photos.size, trend.lastOrNull(), seriesConclusion(photos, it.refPhotoId)) } }
            if (trend.size >= 2) item(key = "trend") { TrendCard(trend, photos.firstOrNull { it.id == zone?.refPhotoId }) }
            items(photos, key = { it.id }) { p ->
                PhotoRow(vm, p, isRef = p.id == zone?.refPhotoId,
                    onSetRef = { vm.startSetReference(p.id) },
                    onDelete = { scope.launch { vm.repo.deletePhoto(p.id) } })
            }
        }
    }

    if (renaming) {
        var name by remember { mutableStateOf(zone?.name ?: "") }
        AlertDialog(onDismissRequest = { renaming = false },
            title = { Text("重命名系列") },
            text = { OutlinedTextField(name, { name = it }, singleLine = true) },
            confirmButton = { TextButton(onClick = { scope.launch { vm.repo.renameZone(zoneId, name) }; renaming = false }) { Text("保存") } },
            dismissButton = { TextButton(onClick = { renaming = false }) { Text("取消") } })
    }
    if (sizing) {
        var txt by remember { mutableStateOf("%.0f".format(zone?.qrMm ?: com.fieldalign.camera.align.Measure.qrSizeMm(zone?.qrText))) }
        AlertDialog(onDismissRequest = { sizing = false },
            title = { Text("二维码打印尺寸") },
            text = {
                Column {
                    Text("用尺子量二维码黑色方块外边长（不含白边），单位毫米。用于把裂缝长度、宽度从像素换算成毫米；修改后已有照片的数值和结论会一并更新。",
                        fontSize = 13.sp, color = Muted)
                    Spacer(Modifier.height(8.dp))
                    OutlinedTextField(txt, { txt = it }, singleLine = true, suffix = { Text("mm") })
                }
            },
            confirmButton = { TextButton(onClick = {
                txt.toDoubleOrNull()?.takeIf { it in 5.0..500.0 }?.let { v -> vm.changeQrMm(zoneId, v) }
                sizing = false
            }) { Text("保存") } },
            dismissButton = { TextButton(onClick = { sizing = false }) { Text("取消") } })
    }
    if (editKinds) {
        var sel by remember { mutableStateOf(com.fieldalign.camera.align.Defects.parse(zone?.kinds)) }
        AlertDialog(onDismissRequest = { editKinds = false },
            title = { Text("病害类型") },
            text = {
                Column {
                    Text("决定这个测点用哪些检测：裂缝量长度宽度；面状看剥落、锈蚀、渗水、泛碱；位移看缝的张开和错动；螺栓看防松标记线。修改后已有照片会重新分析。",
                        fontSize = 13.sp, color = Muted)
                    Spacer(Modifier.height(8.dp))
                    KindPicker(sel) { sel = it }
                }
            },
            confirmButton = { TextButton(onClick = { editKinds = false; vm.changeKinds(zoneId, sel) }) { Text("保存并重新分析") } },
            dismissButton = { TextButton(onClick = { editKinds = false }) { Text("取消") } })
    }
    if (confirmDelete) {
        AlertDialog(onDismissRequest = { confirmDelete = false },
            title = { Text("删除整个系列？") },
            text = { Text("会删除这个系列的全部 ${photos.size} 张照片和对比结果，无法恢复。") },
            confirmButton = { TextButton(onClick = {
                confirmDelete = false
                scope.launch { vm.repo.deleteZone(zoneId); vm.back() }
            }) { Text("删除", color = Bad) } },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("取消") } })
    }
}

@Composable
private fun PhotoRow(vm: AppViewModel, p: Photo, isRef: Boolean, onSetRef: () -> Unit, onDelete: () -> Unit) {
    val thumb = rememberBitmap(p.path) { vm.loadBitmap(p.path, 240) }
    var menu by remember { mutableStateOf(false) }
    AppCard(Modifier.fillMaxWidth(), onClick = { if (!isRef) vm.go(Screen.Compare(p.id)) }) {
        Row(Modifier.padding(10.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(76.dp).clip(RoundedCornerShape(12.dp)).background(Color(0xFFDDE1E6))) {
                thumb?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
            }
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    StatusChip(if (isRef) "reference" else p.status)
                    Spacer(Modifier.width(8.dp))
                    Text(fmtTime(p.takenAt), fontSize = 13.sp, color = Color.Gray)
                }
                Text(if (isRef) "后续照片都与这张对比" else p.message, fontSize = 13.sp, maxLines = 2,
                    overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(top = 4.dp))
                val lv = p.conclusion?.let { Regex("^【(.+?)】").find(it)?.groupValues?.get(1) }
                if (!isRef && lv != null && p.findingsJson != null) {          // 病害结论：等级 + 第一条明细
                    val its = com.fieldalign.camera.align.Analysis.itemsOf(p.findingsJson)
                    val first = (its.firstOrNull { "正常" !in it } ?: its.firstOrNull())
                        ?: p.conclusion.removePrefix("【$lv】").lineSequence().first().substringBefore("。")
                    Text("$lv · $first", fontSize = 13.sp, fontWeight = FontWeight.Medium, color = levelColor(lv),
                        maxLines = 2, overflow = TextOverflow.Ellipsis)
                } else if (!isRef && p.changedRatio != null)
                    Text("结构变化 ${p.changeRegions ?: 0} 处 · %.2f%%".format(p.changedRatio * 100) +
                        if ((p.lightRegions ?: 0) > 0) "  · 明暗 ${p.lightRegions} 处" else "",
                        fontSize = 13.sp, fontWeight = FontWeight.Medium,
                        color = if ((p.changeRegions ?: 0) > 0) Warn else Good)
                if (!isRef && p.crackLenPx != null && (p.crackCount ?: 0) > 0)
                    Text("裂缝 ${p.crackCount} 条 · 总长 ${com.fieldalign.camera.align.Measure.fmtLen(p.crackLenPx, p.mmPerPx)}" +
                        (p.conclusion?.let { Regex("^【(.+?)】").find(it)?.groupValues?.get(1)?.let { l -> " · $l" } } ?: ""),
                        fontSize = 12.sp, color = Muted)
                if (!isRef && !p.quality.isNullOrBlank())
                    Text("⚠ " + p.quality.lineSequence().first(), fontSize = 12.sp, color = Color(0xFFB45309),
                        maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
            Box {
                IconButton(onClick = { menu = true }) { Icon(Icons.Default.MoreVert, null) }
                DropdownMenu(menu, { menu = false }) {
                    if (!isRef) {
                        DropdownMenuItem(text = { Text("设为参考照片") }, onClick = { menu = false; onSetRef() })
                        DropdownMenuItem(text = { Text("删除") }, onClick = { menu = false; onDelete() })
                    } else DropdownMenuItem(text = { Text("参考照片不能删除，请先把别的照片设为参考") }, onClick = { menu = false })
                }
            }
        }
    }
}

/** 走势图：结构变化占比，或裂缝总长（参考照片为起点）。一眼看出病害是否在发展。 */
@Composable
private fun TrendCard(trend: List<Photo>, ref: Photo?) {
    val hasCrack = trend.any { it.crackLenPx != null }
    var metric by remember { mutableStateOf(if (hasCrack) 1 else 0) }
    val mm = trend.lastOrNull { it.mmPerPx != null }?.mmPerPx
    // 裂缝：起点用参考照片自身的裂缝长度（存在每张复拍的明细里）
    val refLen = trend.firstNotNullOfOrNull { p -> p.crackJson?.let { runCatching { org.json.JSONObject(it).getDouble("ref_total_px") }.getOrNull() } }
    val pts: List<Pair<Long, Double>> = if (metric == 0) trend.map { it.takenAt to (it.changedRatio ?: 0.0) * 100 }
        else (listOfNotNull(if (ref != null && refLen != null) ref.takenAt to refLen * (mm ?: 1.0) else null) +
              trend.filter { it.crackLenPx != null }.map { it.takenAt to it.crackLenPx!! * (it.mmPerPx ?: mm ?: 1.0) })
    val unit = if (metric == 0) "%" else if (mm != null) " mm" else " px"
    AppCard(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("走势", fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.width(10.dp))
                listOf("结构变化", "裂缝总长").forEachIndexed { i, l ->
                    if (i == 1 && !hasCrack) return@forEachIndexed
                    Pill(l, if (metric == i) Ink else Muted, filled = metric == i,
                        modifier = Modifier.padding(end = 6.dp).clip(RoundedCornerShape(50)).clickable { metric = i })
                }
                Spacer(Modifier.weight(1f))
                if (pts.isNotEmpty()) Text("最新 %.2f%s".format(pts.last().second, unit), fontSize = 12.sp, color = Color.Gray)
            }
            if (pts.size < 2) { Text("数据不足", fontSize = 12.sp, color = Muted, modifier = Modifier.padding(top = 8.dp)); return@Column }
            val vals = pts.map { it.second }
            val maxV = maxOf(vals.max(), if (metric == 0) 0.05 else 1.0)
            androidx.compose.foundation.Canvas(Modifier.fillMaxWidth().height(110.dp).padding(top = 10.dp)) {
                val n = vals.size
                val left = 8f; val right = size.width - 8f; val top = 6f; val bottom = size.height - 6f
                val t0 = pts.first().first.toDouble(); val t1 = maxOf(pts.last().first.toDouble(), t0 + 1)
                fun x(i: Int) = (left + (right - left) * ((pts[i].first - t0) / (t1 - t0))).toFloat()
                fun y(v: Double) = (bottom - (bottom - top) * (v / maxV)).toFloat()
                drawLine(Color(0xFFDDE1E6), androidx.compose.ui.geometry.Offset(left, bottom),
                    androidx.compose.ui.geometry.Offset(right, bottom), 2f)
                val path = androidx.compose.ui.graphics.Path()
                vals.forEachIndexed { i, v -> if (i == 0) path.moveTo(x(i), y(v)) else path.lineTo(x(i), y(v)) }
                drawPath(path, Ink, style = androidx.compose.ui.graphics.drawscope.Stroke(width = 4f))
                vals.forEachIndexed { i, v ->
                    val up = i > 0 && v > vals[i - 1] * 1.05 + 1e-9
                    drawCircle(if (up) Warn else Good, 7f, androidx.compose.ui.geometry.Offset(x(i), y(v)))
                }
            }
            Row(Modifier.fillMaxWidth().padding(top = 4.dp)) {
                Text(fmtTime(pts.first().first), fontSize = 11.sp, color = Color.Gray)
                Spacer(Modifier.weight(1f))
                Text(fmtTime(pts.last().first), fontSize = 11.sp, color = Color.Gray)
            }
        }
    }
}

/** 系列结论：从参考到最近一次，裂缝总长、最大宽度的变化与月均增长。 */
private fun seriesConclusion(photos: List<Photo>, refId: Long?): String? {
    val ref = photos.firstOrNull { it.id == refId } ?: return null
    val seq = photos.filter { it.id != refId && it.crackLenPx != null }.sortedBy { it.takenAt }
    if (seq.isEmpty()) return null
    val last = seq.last()
    val o = runCatching { org.json.JSONObject(last.crackJson!!) }.getOrNull() ?: return null
    val mm = last.mmPerPx
    val l0 = o.getDouble("ref_total_px"); val l1 = last.crackLenPx!!
    val w0 = o.getDouble("ref_max_w_px"); val w1 = last.crackMaxWPx ?: 0.0
    if (l0 == 0.0 && l1 == 0.0) return "共 ${seq.size} 次复拍，均未检出裂缝。"
    val days = maxOf((last.takenAt - ref.takenAt) / 86_400_000.0, 1.0)
    val per30 = (l1 - l0) / days * 30
    val f = com.fieldalign.camera.align.Measure
    val trend = when {
        l1 - l0 >= maxOf(10.0, 0.05 * l0) -> if (seq.size >= 3 && seq.zipWithNext().all { (a, b) -> b.crackLenPx!! >= a.crackLenPx!! - 5 }) "持续发展" else "有所发展"
        else -> "基本稳定"
    }
    return "共 ${seq.size} 次复拍（${days.toInt()} 天）：裂缝总长 ${f.fmtLen(l0, mm)} → ${f.fmtLen(l1, mm)}" +
        (if (days >= 7) "，平均每 30 天 ${if (per30 >= 0) "+" else ""}${f.fmtLen(per30, mm)}" else "") +
        "；最大宽度 ${f.fmtW(w0, mm)} → ${f.fmtW(w1, mm)}。判定：$trend。"
}

/** 系列概况：照片数、最近一次结果、位置（可在地图中打开）。 */
@Composable
private fun SummaryCard(z: com.fieldalign.camera.data.Zone, count: Int, last: Photo?, series: String?) {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    AppCard(Modifier.fillMaxWidth()) {
        Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("$count 张照片", fontWeight = FontWeight.SemiBold, fontSize = 16.sp)
                Text(when {
                    last == null -> "还没有复拍，点「复拍」开始"
                    last.conclusion != null -> "最近一次（${fmtTime(last.takenAt)}）：" + (Regex("^【(.+?)】").find(last.conclusion)?.groupValues?.get(1) ?: "")
                    (last.changeRegions ?: 0) > 0 -> "最近一次（${fmtTime(last.takenAt)}）：${last.changeRegions} 处结构变化"
                    else -> "最近一次（${fmtTime(last.takenAt)}）：无结构变化"
                }, fontSize = 13.sp, color = last?.conclusion?.let { c -> Regex("^【(.+?)】").find(c)?.groupValues?.get(1)?.let { levelColor(it) } }
                    ?: if ((last?.changeRegions ?: 0) > 0) Warn else Muted)
            }
            if (z.lat != null && z.lon != null) {
                Pill("位置", Info, icon = androidx.compose.material.icons.Icons.Default.LocationOn,
                    modifier = Modifier.clip(RoundedCornerShape(50)).clickable {
                        val uri = android.net.Uri.parse("geo:${z.lat},${z.lon}?q=${z.lat},${z.lon}(${android.net.Uri.encode(z.name)})")
                        runCatching { ctx.startActivity(android.content.Intent(android.content.Intent.ACTION_VIEW, uri)) }
                            .onFailure { android.widget.Toast.makeText(ctx, "%.6f, %.6f".format(z.lat, z.lon), android.widget.Toast.LENGTH_LONG).show() }
                    })
            } else Pill("无位置", Muted)
        }
        val info = listOfNotNull(
            z.azimuth?.let { "朝向 " + com.fieldalign.camera.data.Geo.fmtAzimuth(it) },
            z.pitch?.let { "俯仰 ${it.toInt()}°" },
            "二维码 %.0f mm".format(z.qrMm ?: com.fieldalign.camera.align.Measure.qrSizeMm(z.qrText)).takeIf { z.qrText != null },
            "类型 " + com.fieldalign.camera.align.Defects.labels(com.fieldalign.camera.align.Defects.parse(z.kinds)))
        if (info.isNotEmpty()) Text(info.joinToString(" · "), fontSize = 12.sp, color = Muted,
            modifier = Modifier.padding(start = 14.dp, end = 14.dp, bottom = if (series == null) 12.dp else 4.dp))
        if (series != null) Text(series, fontSize = 13.sp, lineHeight = 19.sp,
            modifier = Modifier.padding(start = 14.dp, end = 14.dp, bottom = 14.dp))
    }
}

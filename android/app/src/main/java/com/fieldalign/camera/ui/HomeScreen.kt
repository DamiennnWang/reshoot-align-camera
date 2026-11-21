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
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddAPhoto
import androidx.compose.material.icons.filled.Assignment
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.CloudSync
import androidx.compose.material.icons.filled.Computer
import androidx.compose.material.icons.filled.Inbox
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Map
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.QrCode2
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.fieldalign.camera.AppViewModel
import com.fieldalign.camera.Screen
import com.fieldalign.camera.data.CaptureMode
import com.fieldalign.camera.data.ZoneSummary

/** 需要关注的等级（裂缝等级 + 通用四级）。 */
private val ATTENTION = setOf("明显", "建议重点关注", "延长", "变宽", "延长且变宽", "新增")
private fun levelOf(z: ZoneSummary) = z.lastConclusion?.let { Regex("^【(.+?)】").find(it)?.groupValues?.get(1) }

/**
 * 首页：
 *  上 — 两个主入口：批量建档（每张一个新测点）、批量复拍（按码 / 位置 + 朝向 + 图像自动匹配）；
 *  中 — 功能格：待确认、导入、导出全部、电脑端处理、地图、巡检任务、云同步、设置（规划中的先占位）；
 *  下 — 需要关注 + 全部测点（搜索、筛选）。
 */
@Composable
fun HomeScreen(vm: AppViewModel) {
    val ctx = LocalContext.current
    val zones by vm.repo.db.zones().summaries().collectAsState(initial = emptyList())
    val pending by vm.repo.db.photos().pending().collectAsState(initial = emptyList())
    var query by remember { mutableStateOf("") }
    var onlyAttention by remember { mutableStateOf(false) }
    val pick = androidx.activity.compose.rememberLauncherForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.PickMultipleVisualMedia()) { uris ->
        if (uris.isNotEmpty()) vm.importUris(uris)
    }
    val importPhotos = {
        pick.launch(androidx.activity.result.PickVisualMediaRequest(
            androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia.ImageOnly))
    }
    val share = { f: java.io.File ->
        val uri = androidx.core.content.FileProvider.getUriForFile(ctx, ctx.packageName + ".files", f)
        val i = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
            type = "application/zip"
            putExtra(android.content.Intent.EXTRA_STREAM, uri)
            putExtra(android.content.Intent.EXTRA_SUBJECT, f.name)
            clipData = android.content.ClipData.newRawUri("", uri)
            addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        ctx.startActivity(android.content.Intent.createChooser(i, "导出全部测点"))
    }
    val attention = zones.filter { levelOf(it) in ATTENTION }
    val shown = zones.filter { z ->
        (query.isBlank() || z.name.contains(query, true) || (z.qrText?.contains(query, true) ?: false)) &&
            (!onlyAttention || levelOf(z) in ATTENTION)
    }

    LazyColumn(Modifier.fillMaxSize().background(Page), contentPadding = PaddingValues(bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item(key = "header") {
            Column(Modifier.fillMaxWidth()
                .background(Brush.verticalGradient(listOf(Ink, InkSoft)), RoundedCornerShape(bottomStart = 26.dp, bottomEnd = 26.dp))
                .statusBarsPadding().padding(horizontal = 18.dp, vertical = 18.dp)) {
                Text("复拍对齐相机", color = Color.White, fontSize = 24.sp, fontWeight = FontWeight.Bold)
                Text(if (zones.isEmpty()) "批量建档 · 复拍自动匹配 · 变化对比"
                     else "${zones.size} 个测点 · ${zones.sumOf { it.photoCount }} 张照片" +
                        (if (attention.isNotEmpty()) " · ${attention.size} 个需关注" else ""),
                    color = Color(0xFFAAB6C3), fontSize = 13.sp, modifier = Modifier.padding(top = 2.dp))
                Spacer(Modifier.height(16.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    ActionTile("批量建档", "每张一个新测点", Icons.Default.AddAPhoto, Amber, Ink, Modifier.weight(1f)) {
                        vm.go(Screen.Camera(mode = CaptureMode.NEW))
                    }
                    ActionTile("批量复拍", "按码/位置自动匹配", Icons.Default.PhotoCamera, Color(0x26FFFFFF), Color.White, Modifier.weight(1f)) {
                        vm.go(Screen.Camera(mode = CaptureMode.RESHOOT))
                    }
                }
            }
        }
        item(key = "grid") {
            Column(Modifier.padding(horizontal = 14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    FnTile("待确认", Icons.Default.Inbox, Modifier.weight(1f), badge = pending.size.takeIf { it > 0 }) { vm.go(Screen.Pending) }
                    FnTile("导入照片", Icons.Default.PhotoLibrary, Modifier.weight(1f)) { importPhotos() }
                    FnTile(if (vm.exporting) "打包中…" else "导出全部", Icons.Default.Share, Modifier.weight(1f)) { if (!vm.exporting) vm.exportAll(share) }
                    FnTile("电脑端处理", Icons.Default.Computer, Modifier.weight(1f)) {
                        vm.go(Screen.Planned("电脑端后处理", PC_HELP))
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    FnTile("测点地图", Icons.Default.Map, Modifier.weight(1f)) { vm.go(Screen.Map) }
                    FnTile("巡检任务", Icons.Default.Assignment, Modifier.weight(1f), planned = true) {
                        vm.go(Screen.Planned("巡检任务", TASK_HELP))
                    }
                    FnTile("云端同步", Icons.Default.CloudSync, Modifier.weight(1f), planned = true) {
                        vm.go(Screen.Planned("云端同步", CLOUD_HELP))
                    }
                    FnTile("设置", Icons.Default.Settings, Modifier.weight(1f)) { vm.go(Screen.Settings) }
                }
            }
        }
        if (zones.isEmpty()) {
            item(key = "guide") { GuideCard() }
        } else {
            if (attention.isNotEmpty() && !onlyAttention && query.isBlank()) {
                item(key = "att_title") { SectionTitle("需要关注（${attention.size}）", Bad) }
                items(attention.take(5), key = { "a${it.id}" }) { z -> ZoneRow(vm, z, Modifier.padding(horizontal = 14.dp)) }
            }
            item(key = "title") { SectionTitle("全部测点（${zones.size}）", Muted) }
            if (zones.size > 4) item(key = "search") {
                Column(Modifier.padding(horizontal = 14.dp)) {
                    OutlinedTextField(query, { query = it }, Modifier.fillMaxWidth(),
                        placeholder = { Text("搜索测点名称或编号") }, singleLine = true,
                        leadingIcon = { Icon(Icons.Default.Search, null) }, shape = RoundedCornerShape(14.dp),
                        colors = OutlinedTextFieldDefaults.colors(unfocusedContainerColor = Color.White, focusedContainerColor = Color.White,
                            unfocusedBorderColor = Hairline))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        FilterChip(selected = !onlyAttention, onClick = { onlyAttention = false }, label = { Text("全部") })
                        FilterChip(selected = onlyAttention, onClick = { onlyAttention = true }, label = { Text("只看需关注") })
                    }
                }
            }
            items(shown, key = { it.id }) { z -> ZoneRow(vm, z, Modifier.padding(horizontal = 14.dp)) }
        }
    }
}

@Composable
private fun SectionTitle(s: String, c: Color) {
    Text(s, fontWeight = FontWeight.SemiBold, fontSize = 15.sp, color = c, modifier = Modifier.padding(start = 18.dp, top = 6.dp))
}

@Composable
private fun ActionTile(title: String, sub: String, icon: ImageVector, bg: Color, fg: Color, modifier: Modifier, onClick: () -> Unit) {
    Row(modifier.clip(RoundedCornerShape(16.dp)).background(bg).clickable(onClick = onClick).padding(14.dp),
        verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(38.dp).background(fg.copy(alpha = 0.14f), CircleShape), contentAlignment = Alignment.Center) {
            Icon(icon, null, tint = fg)
        }
        Spacer(Modifier.width(10.dp))
        Column {
            Text(title, color = fg, fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
            Text(sub, color = fg.copy(alpha = 0.7f), fontSize = 11.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    }
}

/** 功能格：图标 + 名称；badge = 角标数字；planned = 规划中（灰色小字）。 */
@Composable
private fun FnTile(label: String, icon: ImageVector, modifier: Modifier, badge: Int? = null, planned: Boolean = false, onClick: () -> Unit) {
    Column(modifier.clip(RoundedCornerShape(14.dp)).background(Color.White).clickable(onClick = onClick).padding(vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally) {
        Box {
            Icon(icon, null, tint = if (planned) Color(0xFFA0AAB4) else Ink, modifier = Modifier.size(26.dp))
            if (badge != null) Box(Modifier.align(Alignment.TopEnd).padding(start = 18.dp).size(18.dp).background(Bad, CircleShape),
                contentAlignment = Alignment.Center) {
                Text(if (badge > 99) "99" else "$badge", color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            }
        }
        Spacer(Modifier.height(6.dp))
        Text(label, fontSize = 12.sp, color = if (planned) Muted else Ink, textAlign = TextAlign.Center, maxLines = 1)
        Text(if (planned) "规划中" else " ", fontSize = 9.sp, color = Color(0xFFA0AAB4))     // 占一行，格子等高
    }
}

/** 首次使用：怎么用。 */
@Composable
private fun GuideCard() {
    AppCard(Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 6.dp)) {
        Column(Modifier.padding(18.dp)) {
            Text("怎么用", fontWeight = FontWeight.SemiBold, fontSize = 17.sp)
            Spacer(Modifier.height(12.dp))
            Step(1, "第一次：批量建档", "点「批量建档」，一个测点拍一张，每张自动建一个测点（记下位置和镜头朝向）。有二维码的按码命名。")
            Step(2, "下次：批量复拍", "点「批量复拍」，按半透明参考对准拍。有码按码归档；无码按 位置 + 朝向 缩小范围，再用图像确认。")
            Step(3, "认不出的进「待确认」", "附近有多个相似测点、或没有定位时不会乱归，回来在「待确认」里点一下归入或新建。", last = true)
        }
    }
}

@Composable
private fun Step(n: Int, title: String, body: String, last: Boolean = false) {
    Row(Modifier.padding(bottom = if (last) 0.dp else 14.dp)) {
        Box(Modifier.size(26.dp).background(Amber, CircleShape), contentAlignment = Alignment.Center) {
            Text("$n", color = Ink, fontWeight = FontWeight.Bold, fontSize = 13.sp)
        }
        Spacer(Modifier.width(12.dp))
        Column {
            Text(title, fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
            Text(body, color = Muted, fontSize = 13.sp, lineHeight = 19.sp)
        }
    }
}

@Composable
fun ZoneRow(vm: AppViewModel, z: ZoneSummary, modifier: Modifier) {
    val thumb = rememberBitmap(z.refPath) { z.refPath?.let { vm.loadBitmap(it, 300) } }
    AppCard(modifier.fillMaxWidth(), onClick = { vm.go(Screen.ZoneDetail(z.id)) }) {
        Row(Modifier.padding(10.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(78.dp).clip(RoundedCornerShape(12.dp)).background(Color(0xFFDDE1E6))) {
                thumb?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
            }
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(z.name, fontWeight = FontWeight.SemiBold, fontSize = 16.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Spacer(Modifier.height(4.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    if (z.qrText != null) Pill(z.qrText, Info, icon = Icons.Default.QrCode2)
                    else if (z.lat != null) Pill("无码 · 有位置", Good, icon = Icons.Default.LocationOn)
                    else Pill("无码", Muted)
                    Pill(com.fieldalign.camera.align.Defects.labels(com.fieldalign.camera.align.Defects.parse(z.kinds)), Muted)
                }
                Spacer(Modifier.height(4.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("${z.photoCount} 张 · ${fmtTime(z.lastTaken)}", color = Muted, fontSize = 12.sp)
                    val lv = levelOf(z)
                    if (lv != null) {
                        Spacer(Modifier.width(8.dp))
                        Box(Modifier.size(7.dp).background(levelColor(lv), CircleShape))
                        Text(" 最近：$lv", color = levelColor(lv), fontSize = 12.sp, fontWeight = FontWeight.Medium)
                    } else if (z.photoCount <= 1) {
                        Spacer(Modifier.width(8.dp))
                        Text("待复拍", color = Muted, fontSize = 12.sp)
                    }
                }
            }
            Icon(Icons.Default.ChevronRight, null, tint = Color(0xFFB0B8C1))
        }
    }
}

const val PC_HELP = """手机上只给大致判断；需要精细分析、出正式报告时，在电脑上处理：

1. 首页「导出全部」（或测点页菜单「导出」），把 ZIP 发到电脑（微信、数据线、网盘都行）。
2. 用浏览器打开项目里的 web/index.html，把 ZIP 拖进去：按测点查看前后对比（滑动对照）、标注图、病害明细，可以筛选需关注的测点、打印成 PDF 报告。不上传任何数据，离线可用。
3. 需要重新分析（更高分辨率、调阈值）时，在电脑上运行：
   python python/report.py 导出包.zip
   会生成带统计的 HTML 报告。

以后可以把这个网页部署到服务器或云上，手机直接上传、团队共享（规划中）。"""

const val TASK_HELP = """规划中：

· 按路线/区段建巡检任务，列出这次要复拍的测点，现场逐个打勾；
· 到点自动提醒（按定位），漏拍的测点在任务结束时提示；
· 任务完成后一键导出这次的全部结果。

目前可以先用「批量复拍」+ 首页的「需要关注」「待复拍」完成同样的工作。"""

const val CLOUD_HELP = """规划中：

· 测点档案和照片同步到服务器（自建或云），多人、多台手机共用同一批测点；
· 电脑端网页直接读取，不用再手动导出 ZIP；
· 照片只上传压缩后的对比图和明细数据，原图按需上传，节省流量。

目前请用首页「导出全部」+ 电脑端网页处理。"""

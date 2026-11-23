package com.fieldalign.camera.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.material.icons.automirrored.filled.HelpOutline
import androidx.compose.material.icons.automirrored.filled.ListAlt
import androidx.compose.material.icons.filled.Layers
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
import androidx.compose.material3.IconButton
import androidx.compose.material3.TextButton
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
 *  顶部 — 标题 + 右上角「教程」「设置」；一个主按钮「拍照」（一张或多张都行，自动认测点 / 自动建档），
 *         两个次要入口「复拍指定测点」「只建新测点」；
 *  中部 — 概况（测点 / 需关注 / 待确认）+ 工具格（生成靶标、编号对应、导入、导出、电脑端、地图）+ 一行「即将推出」；
 *  下部 — 需要关注 + 全部测点（搜索、筛选）。
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
    // 第一次打开：先看教程
    androidx.compose.runtime.LaunchedEffect(Unit) { if (!vm.tutorialSeen) vm.go(Screen.Tutorial) }
    val attention = zones.filter { levelOf(it) in ATTENTION }
    val shown = zones.filter { z ->
        (query.isBlank() || z.name.contains(query, true) || (z.qrText?.contains(query, true) ?: false)) &&
            (!onlyAttention || levelOf(z) in ATTENTION)
    }

    LazyColumn(Modifier.fillMaxSize().background(Page), contentPadding = PaddingValues(bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item(key = "header") {
            Box(Modifier.fillMaxWidth()
                .clip(RoundedCornerShape(bottomStart = 28.dp, bottomEnd = 28.dp))
                .background(Brush.verticalGradient(listOf(Ink, InkSoft)))) {
                HeroArt(Modifier.matchParentSize())
                Column(Modifier.fillMaxWidth().statusBarsPadding().padding(start = 18.dp, end = 8.dp, top = 6.dp, bottom = 18.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f).padding(top = 8.dp)) {
                            Text("复拍对齐相机", color = Color.White, fontSize = 24.sp, fontWeight = FontWeight.Bold)
                            Text("同一处，隔段时间再拍，自动找出变化", color = Color(0xFFAAB6C3), fontSize = 13.sp)
                        }
                        IconButton(onClick = { vm.go(Screen.Tutorial) }) {
                            Icon(Icons.AutoMirrored.Filled.HelpOutline, "教程", tint = Color.White)
                        }
                        IconButton(onClick = { vm.go(Screen.Settings) }) { Icon(Icons.Default.Settings, "设置", tint = Color.White) }
                    }
                    Spacer(Modifier.height(18.dp))
                    // 主按钮：拍照（不管一张还是一批）
                    Row(Modifier.padding(end = 10.dp).fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(Amber)
                        .clickable { vm.go(Screen.Camera(mode = CaptureMode.AUTO)) }.padding(horizontal = 16.dp, vertical = 16.dp),
                        verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(50.dp).background(Ink, CircleShape), contentAlignment = Alignment.Center) {
                            Icon(Icons.Default.PhotoCamera, null, tint = Amber, modifier = Modifier.size(28.dp))
                        }
                        Spacer(Modifier.width(14.dp))
                        Column(Modifier.weight(1f)) {
                            Text("拍照", color = Ink, fontWeight = FontWeight.Bold, fontSize = 20.sp)
                            Text("一张或多张 · 自动认测点、自动建档", color = Ink.copy(alpha = 0.75f), fontSize = 12.sp, maxLines = 1)
                        }
                        Icon(Icons.Default.ChevronRight, null, tint = Ink)
                    }
                    Spacer(Modifier.height(10.dp))
                    Row(Modifier.padding(end = 10.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        GhostButton("复拍指定测点", Icons.Default.Layers, Modifier.weight(1f)) { vm.go(Screen.PickZone) }
                        GhostButton("只建新测点", Icons.Default.AddAPhoto, Modifier.weight(1f)) { vm.go(Screen.Camera(mode = CaptureMode.NEW)) }
                    }
                }
            }
        }
        item(key = "stats") {
            AppCard(Modifier.fillMaxWidth().padding(horizontal = 14.dp)) {
                Row(Modifier.padding(vertical = 12.dp)) {
                    Stat("${zones.size}", "测点", Ink, Modifier.weight(1f))
                    Stat("${attention.size}", "需关注", if (attention.isNotEmpty()) Bad else Muted, Modifier.weight(1f)) {
                        if (attention.isNotEmpty()) onlyAttention = true
                    }
                    Stat("${pending.size}", "待确认", if (pending.isNotEmpty()) Warn else Muted, Modifier.weight(1f)) { vm.go(Screen.Pending) }
                }
            }
        }
        item(key = "grid") {
            Column(Modifier.padding(horizontal = 14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    FnTile("生成靶标", Icons.Default.QrCode2, Modifier.weight(1f)) { vm.go(Screen.Targets) }
                    FnTile("编号对应", Icons.AutoMirrored.Filled.ListAlt, Modifier.weight(1f)) { vm.go(Screen.CodeMap) }
                    FnTile("导入照片", Icons.Default.PhotoLibrary, Modifier.weight(1f)) { importPhotos() }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    FnTile(if (vm.exporting) "打包中…" else "导出全部", Icons.Default.Share, Modifier.weight(1f)) { if (!vm.exporting) vm.exportAll(share) }
                    FnTile("电脑端处理", Icons.Default.Computer, Modifier.weight(1f)) { vm.go(Screen.Planned("电脑端处理", PC_HELP)) }
                    FnTile("测点地图", Icons.Default.Map, Modifier.weight(1f)) { vm.go(Screen.Map) }
                }
                // 规划中的功能：收在一行里，不占格子
                Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).clickable { vm.go(Screen.Planned("即将推出", SOON_HELP)) }
                    .padding(horizontal = 4.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text("即将推出", color = Muted, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                    Spacer(Modifier.width(8.dp))
                    listOf("巡检任务", "云端同步", "RTK 高精度定位").forEach {
                        Text(it, color = Muted, fontSize = 12.sp, modifier = Modifier.padding(end = 6.dp)
                            .border(1.dp, Hairline, RoundedCornerShape(50)).padding(horizontal = 8.dp, vertical = 2.dp))
                    }
                    Spacer(Modifier.weight(1f))
                    Icon(Icons.Default.ChevronRight, null, tint = Color(0xFFB0B8C1), modifier = Modifier.size(18.dp))
                }
            }
        }
        if (zones.isEmpty()) {
            item(key = "guide") { GuideCard(vm) }
        } else {
            if (attention.isNotEmpty() && !onlyAttention && query.isBlank()) {
                item(key = "att_title") { SectionTitle("需要关注（${attention.size}）", Bad) }
                items(attention.take(5), key = { "a${it.id}" }) { z -> ZoneRow(vm, z, Modifier.padding(horizontal = 14.dp)) }
            }
            item(key = "title") { SectionTitle(if (onlyAttention) "需要关注（${attention.size}）" else "全部测点（${zones.size}）", Muted) }
            if (zones.size > 4 || onlyAttention) item(key = "search") {
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
private fun GhostButton(title: String, icon: ImageVector, modifier: Modifier, onClick: () -> Unit) {
    Row(modifier.clip(RoundedCornerShape(14.dp)).background(Color(0x22FFFFFF)).border(1.dp, Color(0x33FFFFFF), RoundedCornerShape(14.dp))
        .clickable(onClick = onClick).padding(horizontal = 12.dp, vertical = 11.dp), verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center) {
        Icon(icon, null, tint = Color.White, modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(6.dp))
        Text(title, color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Medium, maxLines = 1)
    }
}

@Composable
private fun Stat(value: String, label: String, color: Color, modifier: Modifier, onClick: (() -> Unit)? = null) {
    Column(modifier.clip(RoundedCornerShape(10.dp)).let { if (onClick != null) it.clickable(onClick = onClick) else it }.padding(vertical = 2.dp),
        horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, color = color, fontSize = 22.sp, fontWeight = FontWeight.Bold)
        Text(label, color = Muted, fontSize = 12.sp)
    }
}

/** 功能格：图标 + 名称（一行，格子等高）；badge = 角标数字。 */
@Composable
private fun FnTile(label: String, icon: ImageVector, modifier: Modifier, badge: Int? = null, onClick: () -> Unit) {
    Column(modifier.clip(RoundedCornerShape(14.dp)).background(Color.White).clickable(onClick = onClick).padding(vertical = 14.dp),
        horizontalAlignment = Alignment.CenterHorizontally) {
        Box {
            Box(Modifier.size(40.dp).background(Page, CircleShape), contentAlignment = Alignment.Center) {
                Icon(icon, null, tint = Ink, modifier = Modifier.size(22.dp))
            }
            if (badge != null) Box(Modifier.align(Alignment.TopEnd).size(18.dp).background(Bad, CircleShape),
                contentAlignment = Alignment.Center) {
                Text(if (badge > 99) "99" else "$badge", color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            }
        }
        Spacer(Modifier.height(6.dp))
        Text(label, fontSize = 13.sp, color = Ink, textAlign = TextAlign.Center, maxLines = 1)
    }
}

/** 还没有测点：三步上手。 */
@Composable
private fun GuideCard(vm: AppViewModel) {
    AppCard(Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 6.dp)) {
        Column(Modifier.padding(18.dp)) {
            Text("三步上手", fontWeight = FontWeight.SemiBold, fontSize = 17.sp)
            Spacer(Modifier.height(12.dp))
            Step(1, "贴靶标（可选）", "「生成靶标」打印编号二维码，贴在裂缝旁。有编号最准，还能换算毫米。")
            Step(2, "点「拍照」", "对准拍一张就是一个测点；一处接一处连着拍也行。")
            Step(3, "下次再点「拍照」", "自动认出是哪个测点，叠加上次的照片帮你对准，拍完自动对比、标出变化。", last = true)
            TextButton(onClick = { vm.go(Screen.Tutorial) }, modifier = Modifier.padding(top = 4.dp)) { Text("看图文教程") }
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
fun ZoneRow(vm: AppViewModel, z: ZoneSummary, modifier: Modifier, onClick: (() -> Unit)? = null) {
    val thumb = rememberBitmap(z.refPath) { z.refPath?.let { vm.loadBitmap(it, 300) } }
    AppCard(modifier.fillMaxWidth(), onClick = onClick ?: { vm.go(Screen.ZoneDetail(z.id)) }) {
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

const val PC_HELP = """手机上给现场判断；需要复核、出正式报告时，在电脑上处理：

1. 首页「导出全部」（或测点页菜单「导出」），把 ZIP 发到电脑（微信、数据线、网盘都行）。
2. 电脑浏览器打开配套网站的「导出包查看器」，把 ZIP 拖进去：前后对比滑块、标注图、病害明细、筛选需关注的测点、打印 PDF 报告。ZIP 只在浏览器里打开，不上传。
3. 需要重新分析（调阈值、批量出报告）：网站「在线处理」上传 ZIP；或下载电脑端工具包，运行
   python report.py 导出包.zip
4. 网站上还能批量生成靶标、查看测试图片。

没有网络时：工具包里的「查看器.html」双击打开即可，功能和网站上的查看器一样。"""

const val SOON_HELP = """以下功能正在规划：

巡检任务
· 按路线 / 区段建任务，列出这次要复拍的测点，现场逐个打勾；
· 到点按定位提醒，漏拍的测点在任务结束时提示；
· 任务完成后一键导出这次的全部结果。

云端同步
· 测点档案和照片同步到服务器（自建或云），多人、多台手机共用同一批测点；
· 电脑端网站直接读取，不用再手动导出 ZIP；
· 只上传压缩后的对比图和明细数据，原图按需上传，节省流量。

RTK 高精度定位
· 蓝牙连接北斗 RTK 接收机 + 网络差分（CORS），测点位置精确到 1–2 厘米；
· 适合测点密集、外观相似、没法贴二维码的场景。
普通场景用二维码 + 手机定位就够了。"""

const val TASK_HELP = SOON_HELP
const val CLOUD_HELP = SOON_HELP

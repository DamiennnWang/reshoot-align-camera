package com.fieldalign.camera.ui

import android.content.ClipData
import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.ExperimentalFoundationApi
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
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.FileDownload
import androidx.compose.material.icons.filled.FileUpload
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import com.fieldalign.camera.AppViewModel
import com.fieldalign.camera.Screen
import com.fieldalign.camera.data.CodeMap
import com.fieldalign.camera.data.TargetSheet
import kotlinx.coroutines.launch
import java.io.File
import kotlin.random.Random

// ====================================================================== 插图（纯绘制，不依赖图片资源）
/** 一条蜿蜒的裂缝（带分叉），坐标按 0..1 归一化，固定种子，每次画出来一样。 */
private fun crackPath(seed: Int, x0: Float, y0: Float, x1: Float, y1: Float, n: Int = 14, amp: Float = 0.04f): List<Offset> {
    val r = Random(seed)
    return (0..n).map { i ->
        val t = i / n.toFloat()
        val jitter = if (i == 0 || i == n) 0f else (r.nextFloat() - 0.5f) * 2 * amp
        Offset(x0 + (x1 - x0) * t + jitter * 0.6f, y0 + (y1 - y0) * t + jitter)
    }
}

private fun DrawScope.polyline(pts: List<Offset>, color: Color, width: Float) {
    if (pts.size < 2) return
    val p = Path().apply { moveTo(pts[0].x * size.width, pts[0].y * size.height); pts.drop(1).forEach { lineTo(it.x * size.width, it.y * size.height) } }
    drawPath(p, color, style = Stroke(width, cap = StrokeCap.Round, join = StrokeJoin.Round))
}

/** 取景框四角。 */
private fun DrawScope.viewfinder(l: Float, t: Float, r: Float, b: Float, color: Color, w: Float) {
    val L = l * size.width; val T = t * size.height; val R = r * size.width; val B = b * size.height
    val k = minOf(R - L, B - T) * 0.16f
    listOf(Triple(L, T, 1 to 1), Triple(R, T, -1 to 1), Triple(L, B, 1 to -1), Triple(R, B, -1 to -1)).forEach { (x, y, d) ->
        drawLine(color, Offset(x, y), Offset(x + k * d.first, y), w, StrokeCap.Round)
        drawLine(color, Offset(x, y), Offset(x, y + k * d.second), w, StrokeCap.Round)
    }
}

/** 小靶标：二维码样式方块 + 三个圆点。 */
private fun DrawScope.miniTarget(cx: Float, cy: Float, side: Float, ink: Color, paper: Color) {
    val s = side * size.minDimension; val x = cx * size.width - s / 2; val y = cy * size.height - s / 2
    drawRoundRect(paper, Offset(x - s * 0.12f, y - s * 0.12f), Size(s * 1.24f, s * 1.24f), CornerRadius(s * 0.08f))
    val m = s / 7
    val r = Random(7)
    for (i in 0 until 7) for (j in 0 until 7) {
        val finder = (i < 3 && j < 3) || (i > 3 && j < 3) || (i < 3 && j > 3)
        val on = if (finder) !(i == 1 && j == 1) && !(i == 5 && j == 1) && !(i == 1 && j == 5) else r.nextFloat() > 0.5f
        if (on) drawRect(ink, Offset(x + i * m, y + j * m), Size(m, m))
    }
}

/** 首页顶部的背景插画：混凝土斑点 + 裂缝 + 取景框 + 测量标注。画在深色底上。 */
@Composable
fun HeroArt(modifier: Modifier) {
    Canvas(modifier) {
        val r = Random(3)
        repeat(160) {
            val o = Offset(r.nextFloat() * size.width, r.nextFloat() * size.height)
            drawCircle(Color.White.copy(alpha = 0.03f + r.nextFloat() * 0.05f), 1f + r.nextFloat() * 3.5f, o)
        }
        // 一条从右上斜贯的裂缝（带两条分叉），只作底纹，不和按钮抢视线
        val main = crackPath(11, 0.66f, -0.05f, 0.98f, 1.05f, 22, 0.035f)
        polyline(main, Color.White.copy(alpha = 0.06f), 10f)
        polyline(main, Color.White.copy(alpha = 0.20f), 2.6f)
        polyline(crackPath(5, main[7].x, main[7].y, 1.05f, 0.42f, 8, 0.025f), Color.White.copy(alpha = 0.14f), 1.8f)
        polyline(crackPath(9, main[15].x, main[15].y, 0.78f, 1.05f, 7, 0.025f), Color.White.copy(alpha = 0.12f), 1.5f)
    }
}

/** 教程每一步的插图。 */
@Composable
private fun StepArt(step: Int, modifier: Modifier) {
    Canvas(modifier.clip(RoundedCornerShape(18.dp)).background(Brush.linearGradient(listOf(Color(0xFF8E959C), Color(0xFF6E767E))))) {
        val r = Random(step + 20)
        repeat(260) {
            val o = Offset(r.nextFloat() * size.width, r.nextFloat() * size.height)
            drawCircle((if (r.nextBoolean()) Color.White else Color.Black).copy(alpha = 0.06f + r.nextFloat() * 0.08f), 1f + r.nextFloat() * 4f, o)
        }
        val crack = crackPath(4, 0.16f, 0.1f, 0.56f, 0.95f, 24, 0.06f)
        val dark = Color(0xFF2A2F35)
        when (step) {
            0 -> { // 贴靶标：二维码在裂缝旁，3 个圆点分散
                polyline(crack, dark, 6f)
                miniTarget(0.74f, 0.3f, 0.2f, dark, Color(0xFFF4F1EA))
                listOf(Offset(0.12f, 0.75f), Offset(0.48f, 0.12f), Offset(0.85f, 0.82f)).forEach { o ->
                    drawCircle(Color(0xFFF4F1EA), size.minDimension * 0.06f, Offset(o.x * size.width, o.y * size.height))
                    drawCircle(dark, size.minDimension * 0.042f, Offset(o.x * size.width, o.y * size.height))
                }
            }
            1 -> { // 拍照：取景框
                polyline(crack, dark, 6f)
                miniTarget(0.74f, 0.3f, 0.2f, dark, Color(0xFFF4F1EA))
                drawRect(Color.Black.copy(alpha = 0.25f))
                viewfinder(0.08f, 0.08f, 0.92f, 0.92f, Amber, 7f)
                drawCircle(Color.White, size.minDimension * 0.07f, Offset(size.width / 2, size.height * 0.86f), style = Stroke(5f))
            }
            2 -> { // 复拍：半透明参考叠加
                polyline(crack, dark, 6f)
                val ghost = crack.map { Offset(it.x + 0.05f, it.y - 0.03f) }
                polyline(ghost, Color.White.copy(alpha = 0.55f), 6f)
                miniTarget(0.74f, 0.3f, 0.2f, dark, Color(0xFFF4F1EA))
                viewfinder(0.08f, 0.08f, 0.92f, 0.92f, Good, 7f)
            }
            3 -> { // 结果：原有黄、新增红，标长度宽度；剥落框出面积
                polyline(crack.take(12), Color(0xFFFFD34D), 7f)
                polyline(crack.drop(11), Color(0xFFFF4D4D), 8f)
                val tag = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply { textSize = size.minDimension * 0.05f; color = android.graphics.Color.WHITE }
                fun label(text: String, x0: Float, y: Float, col: Color) {
                    val h = tag.textSize * 1.7f; val pad = h * 0.45f
                    val w = pad + h * 0.35f + 10f + tag.measureText(text) + pad
                    val x = x0.coerceAtMost(size.width - w - 14f)
                    drawRoundRect(Color(0xE6111820), Offset(x, y - h / 2), Size(w, h), CornerRadius(h / 2))
                    drawCircle(col, h * 0.17f, Offset(x + pad + h * 0.17f, y))
                    drawContext.canvas.nativeCanvas.drawText(text, x + pad + h * 0.35f + 10f, y + tag.textSize * 0.36f, tag)
                }
                val p = crack[18]
                label("新增 96 mm · 宽 0.4 mm", p.x * size.width + 30f, p.y * size.height, Color(0xFFFF4D4D))
                // 剥落区域
                val sp = Path().apply {
                    val cx = size.width * 0.74f; val cy = size.height * 0.3f; val r = size.minDimension * 0.13f
                    val rr = Random(2)
                    for (k in 0..11) { val a = k / 12f * 6.283f; val d = r * (0.75f + rr.nextFloat() * 0.45f)
                        val x = cx + d * kotlin.math.cos(a); val y = cy + d * kotlin.math.sin(a); if (k == 0) moveTo(x, y) else lineTo(x, y) }
                    close()
                }
                drawPath(sp, Color(0x55222222))
                drawPath(sp, Color(0xFFFF4D4D), style = Stroke(6f, join = StrokeJoin.Round))
                label("剥落 约 32 cm²", size.width * 0.55f, size.height * 0.5f, Color(0xFFFF4D4D))
            }
            else -> { // 电脑端：屏幕
                drawRect(Color(0xFF1B2530))
                drawRoundRect(Color(0xFF26323F), Offset(size.width * 0.1f, size.height * 0.14f), Size(size.width * 0.8f, size.height * 0.56f), CornerRadius(14f))
                val c2 = crackPath(4, 0.25f, 0.2f, 0.55f, 0.64f, 10, 0.04f)
                polyline(c2, Color(0xFFFF4D4D), 5f)
                drawLine(Amber, Offset(size.width * 0.5f, size.height * 0.14f), Offset(size.width * 0.5f, size.height * 0.7f), 4f)
                drawRect(Color(0xFF26323F), Offset(size.width * 0.38f, size.height * 0.72f), Size(size.width * 0.24f, size.height * 0.08f))
                drawRect(Color(0xFF34424F), Offset(size.width * 0.28f, size.height * 0.8f), Size(size.width * 0.44f, size.height * 0.03f))
            }
        }
    }
}

// ====================================================================== 教程
private val TUTORIAL = listOf(
    "贴靶标" to "二维码贴在裂缝旁边，3 个圆点分散贴在周围（不挡裂缝、不排成一条线）。没有靶标也能用，只是没有编号和毫米尺寸。\n靶标可以在首页「生成靶标」批量打印：T0001、T0002……一处一个编号。",
    "拍照建档" to "首页点「拍照」，对准拍一张就行，也可以一处接一处连着拍。读到新编号、或到了新位置，自动建一个测点。",
    "下次复拍" to "过一段时间再来，还是点「拍照」。按编号（或位置 + 朝向 + 图像）自动认出是哪个测点，画面上会叠加上次的照片，对准了再按快门。认不准的放进「待确认」，不会乱归。",
    "看结果" to "复拍后自动对齐、对比：默认显示「标注」——黄色是原有裂缝，红色是新增部分，框出剥落、渗水、锈蚀等，写明长度、宽度、面积。",
    "电脑上复核" to "首页「导出全部」得到一个 ZIP，在电脑浏览器里打开网站（或工具包里的查看器）查看、重新分析、打印报告。编号可以在「编号对应」里对上项目编号。",
)

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun TutorialScreen(vm: AppViewModel) {
    val pager = rememberPagerState { TUTORIAL.size }
    val scope = rememberCoroutineScope()
    val done = { vm.markTutorialSeen(); vm.back(); Unit }
    Column(Modifier.fillMaxSize().background(Page).statusBarsPadding().navigationBarsPadding()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("使用教程", fontWeight = FontWeight.SemiBold, fontSize = 17.sp, modifier = Modifier.padding(start = 10.dp).weight(1f))
            TextButton(onClick = done) { Text("跳过", color = Muted) }
        }
        HorizontalPager(pager, Modifier.weight(1f)) { i ->
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 22.dp, vertical = 8.dp)) {
                StepArt(i, Modifier.fillMaxWidth().aspectRatio(1.25f))
                Spacer(Modifier.height(20.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(28.dp).background(Amber, CircleShape), contentAlignment = Alignment.Center) {
                        Text("${i + 1}", color = Ink, fontWeight = FontWeight.Bold)
                    }
                    Spacer(Modifier.width(10.dp))
                    Text(TUTORIAL[i].first, fontSize = 22.sp, fontWeight = FontWeight.Bold)
                }
                Spacer(Modifier.height(10.dp))
                Text(TUTORIAL[i].second, fontSize = 15.sp, lineHeight = 23.sp, color = Color(0xFF3A4652))
            }
        }
        Row(Modifier.fillMaxWidth().padding(horizontal = 22.dp, vertical = 14.dp), verticalAlignment = Alignment.CenterVertically) {
            Row(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                repeat(TUTORIAL.size) { k ->
                    Box(Modifier.size(if (k == pager.currentPage) 22.dp else 8.dp, 8.dp).clip(CircleShape)
                        .background(if (k == pager.currentPage) Ink else Hairline))
                }
            }
            val last = pager.currentPage == TUTORIAL.size - 1
            Button(onClick = { if (last) done() else scope.launch { pager.animateScrollToPage(pager.currentPage + 1) } },
                colors = ButtonDefaults.buttonColors(containerColor = Amber, contentColor = Ink)) {
                Text(if (last) "开始使用" else "下一步", fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

// ====================================================================== 生成靶标
@Composable
fun TargetsScreen(vm: AppViewModel) {
    val ctx = LocalContext.current
    var prefix by remember { mutableStateOf(vm.targetPrefix) }
    var start by remember { mutableStateOf(vm.nextTargetStart(vm.targetPrefix).toString()) }
    var count by remember { mutableStateOf("12") }
    var digits by remember { mutableStateOf("4") }
    var busy by remember { mutableStateOf(false) }
    val zones by vm.repo.db.zones().summaries().collectAsState(initial = emptyList())
    val p = prefix.trim()
    val st = start.toIntOrNull(); val n = count.toIntOrNull(); val d = digits.toIntOrNull()
    val ok = Regex("[A-Za-z0-9_\\-]{0,12}").matches(p) && st != null && st >= 0 && n != null && n in 1..300 && d != null && d in 1..8
    val codes = if (ok) CodeMap.codes(p, st!!, n!!, d!!) else emptyList()
    val used = remember(codes, zones) { val qs = zones.mapNotNull { it.qrText?.let(CodeMap::base) }.toSet(); codes.filter { it in qs } }
    val preview = remember(codes.firstOrNull()) { codes.firstOrNull()?.let { TargetSheet.preview(it, 560) } }

    Scaffold(containerColor = Page, topBar = { AppTopBar("生成靶标", "一处一个编号，打印后随手贴", onBack = { vm.back() }) }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).verticalScroll(rememberScrollState()).padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)) {
            AppCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(prefix, { prefix = it.take(12); start = vm.nextTargetStart(it.trim()).toString() }, Modifier.weight(1f),
                            label = { Text("前缀") }, singleLine = true)
                        OutlinedTextField(start, { start = it.filter(Char::isDigit).take(8) }, Modifier.weight(1f), label = { Text("起始号") },
                            singleLine = true, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number))
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(count, { count = it.filter(Char::isDigit).take(3) }, Modifier.weight(1f), label = { Text("数量（≤300）") },
                            singleLine = true, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number))
                        OutlinedTextField(digits, { digits = it.filter(Char::isDigit).take(1) }, Modifier.weight(1f), label = { Text("数字位数") },
                            singleLine = true, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number))
                    }
                    Text(if (!ok) "前缀只能用字母、数字、- 和 _；数量 1–300"
                         else "${codes.first()} … ${codes.last()}，共 ${codes.size} 个，${(codes.size + 5) / 6} 页 A4",
                        color = if (ok) Muted else Bad, fontSize = 13.sp)
                    if (used.isNotEmpty()) Text("注意：${used.take(3).joinToString("、")}${if (used.size > 3) " 等 ${used.size} 个" else ""} 已经建档，重复贴会把不同裂缝当成同一个测点。",
                        color = Warn, fontSize = 13.sp)
                    Button(onClick = {
                        busy = true
                        vm.makeTargets(p, st!!, n!!, d!!) { f -> busy = false; start = (st + n).toString(); sharePdf(ctx, f) }
                    }, enabled = ok && !busy, modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = Amber, contentColor = Ink)) {
                        Text(if (busy) "生成中…" else "生成 PDF 并分享（打印 / 发到电脑）", fontWeight = FontWeight.SemiBold)
                    }
                }
            }
            preview?.let {
                AppCard(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(14.dp)) {
                        Text("标签样式（实际 95 × 92 mm）", fontSize = 13.sp, color = Muted)
                        Spacer(Modifier.height(8.dp))
                        Image(it.asImageBitmap(), null, Modifier.fillMaxWidth().aspectRatio(it.width.toFloat() / it.height).border(1.dp, Hairline, RoundedCornerShape(8.dp)))
                    }
                }
            }
            AppCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp)) {
                    Text("怎么用", fontWeight = FontWeight.SemiBold)
                    Text("1. 打印时选「实际大小 / 100%」，二维码边长应为 40 mm（App 靠它换算毫米）。\n" +
                        "2. 二维码贴在裂缝旁，3 个圆点分散贴在裂缝周围；同一个编号只贴一处。\n" +
                        "3. 拍照时读到编号自动建测点，下次按编号自动归入。\n" +
                        "4. 编号和项目编号的对应：首页「编号对应」里逐个填写，或导入 CSV（电脑上填好表格），测点会自动改名。",
                        fontSize = 13.sp, color = Color(0xFF3A4652), lineHeight = 20.sp, modifier = Modifier.padding(top = 6.dp))
                    TextButton(onClick = { vm.go(Screen.CodeMap) }) { Text("打开编号对应表") }
                }
            }
        }
    }
}

private fun shareFile(ctx: android.content.Context, f: File, mime: String, title: String) {
    val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".files", f)
    val i = Intent(Intent.ACTION_SEND).apply {
        type = mime
        putExtra(Intent.EXTRA_STREAM, uri); putExtra(Intent.EXTRA_SUBJECT, f.name)
        clipData = ClipData.newRawUri("", uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    ctx.startActivity(Intent.createChooser(i, title))
}

private fun sharePdf(ctx: android.content.Context, f: File) = shareFile(ctx, f, "application/pdf", "打印或发送靶标")

// ====================================================================== 编号对应表
@Composable
fun CodeMapScreen(vm: AppViewModel) {
    val ctx = LocalContext.current
    val zones by vm.repo.db.zones().summaries().collectAsState(initial = emptyList())
    val ver = vm.codeMapVersion
    val entries = remember(ver) { vm.repo.codeMap.all() }
    val zoneByCode = zones.filter { it.qrText != null }.associateBy { CodeMap.base(it.qrText!!) }
    // 已建档但还没进表的编号也列出来，方便直接填
    val rows = remember(entries, zoneByCode) {
        (entries.map { it.code } + zoneByCode.keys).distinct().sorted().map { c -> vm.repo.codeMap.get(c) ?: CodeMap.Entry(c) }
    }
    var query by remember { mutableStateOf("") }
    var editing by remember { mutableStateOf<CodeMap.Entry?>(null) }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { u -> u?.let { vm.importCodeMap(it) } }
    val shown = rows.filter { query.isBlank() || listOf(it.code, it.project, it.name, it.note, zoneByCode[it.code]?.name.orEmpty()).any { s -> s.contains(query, true) } }

    Scaffold(containerColor = Page, topBar = {
        AppTopBar("编号对应", "${rows.size} 个编号 · ${rows.count { it.display != null }} 个已对应", onBack = { vm.back() }, actions = {
            IconButton(onClick = { pick.launch(arrayOf("text/*", "text/csv", "text/comma-separated-values", "application/vnd.ms-excel", "application/octet-stream")) }) {
                Icon(Icons.Default.FileUpload, "导入 CSV")
            }
            IconButton(onClick = {
                val dir = File(ctx.cacheDir, "exports").apply { mkdirs() }
                val f = File(dir, "编号对应表.csv")
                f.writeText(vm.repo.codeMap.toCsv(rows) { c -> zoneByCode[c]?.name })
                shareFile(ctx, f, "text/csv", "导出编号对应表")
            }) { Icon(Icons.Default.FileDownload, "导出 CSV") }
        })
    }) { pad ->
        LazyColumn(Modifier.fillMaxSize().padding(pad), contentPadding = PaddingValues(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            item {
                Text("靶标编号（二维码里的 T0001 等）对应到你的项目编号。填好后已建档的测点自动改名；还没拍的编号，第一次拍到时直接用对应的名称建档。" +
                    "也可以导出 CSV 在电脑上填好再导入（右上角）：列依次为 编号、项目编号、名称、备注。",
                    color = Muted, fontSize = 13.sp, lineHeight = 19.sp)
            }
            if (rows.size > 6) item {
                OutlinedTextField(query, { query = it }, Modifier.fillMaxWidth(), placeholder = { Text("搜索编号、项目编号、名称") },
                    singleLine = true, leadingIcon = { Icon(Icons.Default.Search, null) }, shape = RoundedCornerShape(14.dp))
            }
            if (rows.isEmpty()) item {
                AppCard(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("还没有编号", fontWeight = FontWeight.SemiBold)
                        Text("先「生成靶标」，或导入一份 CSV", color = Muted, fontSize = 13.sp)
                        TextButton(onClick = { vm.go(Screen.Targets) }) { Text("去生成靶标") }
                    }
                }
            }
            items(shown, key = { it.code }) { e ->
                val z = zoneByCode[e.code]
                AppCard(Modifier.fillMaxWidth(), onClick = { editing = e }) {
                    Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(e.code, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                                Spacer(Modifier.width(8.dp))
                                if (z != null) Pill("已建档", Good) else Pill("未拍", Muted)
                            }
                            Text(e.display?.let { "→ $it" } ?: "未填写项目编号", color = if (e.display != null) Ink else Muted, fontSize = 14.sp)
                            if (e.note.isNotBlank()) Text(e.note, color = Muted, fontSize = 12.sp)
                        }
                        Icon(Icons.Default.Edit, "编辑", tint = Muted)
                    }
                }
            }
        }
    }
    editing?.let { e ->
        var project by remember(e.code) { mutableStateOf(e.project) }
        var name by remember(e.code) { mutableStateOf(e.name) }
        var note by remember(e.code) { mutableStateOf(e.note) }
        AlertDialog(onDismissRequest = { editing = null },
            title = { Text("编号 ${e.code}") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    OutlinedTextField(project, { project = it }, label = { Text("项目编号") }, singleLine = true)
                    OutlinedTextField(name, { name = it }, label = { Text("名称（如 2#墩 西侧）") }, singleLine = true)
                    OutlinedTextField(note, { note = it }, label = { Text("备注") }, singleLine = true)
                    zoneByCode[e.code]?.let { Text("已建档测点：${it.name}（保存后改名）", color = Muted, fontSize = 12.sp) }
                }
            },
            confirmButton = { TextButton(onClick = {
                vm.repo.codeMap.put(e.copy(project = project.trim(), name = name.trim(), note = note.trim())); editing = null; vm.applyCodeMap()
            }) { Text("保存") } },
            dismissButton = { TextButton(onClick = { editing = null }) { Text("取消") } })
    }
}

// ====================================================================== 复拍指定测点：选测点
@Composable
fun PickZoneScreen(vm: AppViewModel) {
    val zones by vm.repo.db.zones().summaries().collectAsState(initial = emptyList())
    var query by remember { mutableStateOf("") }
    val shown = zones.filter { query.isBlank() || it.name.contains(query, true) || (it.qrText?.contains(query, true) ?: false) }
    Scaffold(containerColor = Page, topBar = { AppTopBar("复拍哪个测点？", "选好后按上次的照片对准拍", onBack = { vm.back() }) }) { pad ->
        LazyColumn(Modifier.fillMaxSize().padding(pad), contentPadding = PaddingValues(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            item {
                OutlinedTextField(query, { query = it }, Modifier.fillMaxWidth(), placeholder = { Text("搜索名称或编号") },
                    singleLine = true, leadingIcon = { Icon(Icons.Default.Search, null) }, shape = RoundedCornerShape(14.dp))
            }
            if (zones.isEmpty()) item { Text("还没有测点：先用「拍照」建档。", color = Muted, modifier = Modifier.padding(8.dp)) }
            items(shown, key = { it.id }) { z ->
                ZoneRow(vm, z, Modifier, onClick = { vm.replaceTop(Screen.Camera(z.id)) })
            }
        }
    }
}

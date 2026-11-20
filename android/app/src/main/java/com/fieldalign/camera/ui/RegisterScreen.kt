package com.fieldalign.camera.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.fieldalign.camera.AppViewModel
import com.fieldalign.camera.Screen
import kotlin.math.hypot

/** 新建系列 / 设为参考：确认靶标圆点（点按切换选中），填写系列名称。 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RegisterScreen(vm: AppViewModel, r: Screen.Register) {
    val bmp = rememberBitmap(r.file.absolutePath) { vm.loadBitmap(r.file.absolutePath) }
    val m = r.markers
    val cands = m.dotCandidates.take(20)
    val selected = remember { mutableStateListOf<Int>().apply {
        addAll(r.autoDots.mapNotNull { d -> cands.indexOfFirst { it.x == d.x && it.y == d.y }.takeIf { it >= 0 } })
    } }
    var name by remember { mutableStateOf(r.qrText ?: "系列 " + java.text.SimpleDateFormat("MM-dd HH:mm", java.util.Locale.CHINA).format(java.util.Date())) }
    val isNew = r.existingPhotoId == null
    var kinds by remember { mutableStateOf(com.fieldalign.camera.align.Defects.fromQr(r.qrText)) }
    val hasQr = m.qrText != null

    Scaffold(containerColor = Page, topBar = {
        AppTopBar(if (isNew) "新建系列" else "设为参考照片", r.qrText?.let { "编号 $it" },
            onBack = { if (isNew) vm.discard(r.file) else vm.back() })
    }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).verticalScroll(rememberScrollState())) {
            Text(
                when {
                    !hasQr && r.qrText != null -> "编号「${r.qrText}」（取景时扫码锁定）。这张全景照片将作为参考：以后扫这个码锁定系列后，后退按参考取景拍照即可，照片里二维码可以很小或不入镜。"
                    hasQr -> "识别到二维码「${m.qrText}」。黄圈是检测到的圆点，绿色为选中的靶标圆点（点按切换，建议选 3 个，分散、不共线）。"
                    else -> "这张照片没有二维码。以后同一处的照片将按图像特征自动归入此系列；背景相似的多个位置建议贴二维码区分。"
                }, Modifier.padding(12.dp), fontSize = 14.sp)
            val ratio = bmp?.let { it.width.toFloat() / it.height } ?: 0.75f
            // 竖拍照片限制高度，保证名称和按钮不被挤出屏幕
            Box(Modifier.fillMaxWidth().background(Color.Black), contentAlignment = androidx.compose.ui.Alignment.Center) {
            Box(Modifier.heightIn(max = 420.dp).aspectRatio(ratio)) {
                bmp?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.FillBounds) }
                if (hasQr) Canvas(Modifier.fillMaxSize().pointerInput(cands) {
                    detectTapGestures { tap ->
                        val sx = size.width.toFloat() / m.workW; val sy = size.height.toFloat() / m.workH
                        val hit = cands.indices.minByOrNull { hypot(cands[it].x * sx - tap.x, cands[it].y * sy - tap.y) }
                        if (hit != null && hypot(cands[hit].x * sx - tap.x, cands[hit].y * sy - tap.y) < 60) {
                            if (hit in selected) selected.remove(hit) else selected.add(hit)
                        }
                    }
                }) {
                    val sx = size.width / m.workW; val sy = size.height / m.workH
                    m.qrCorners?.let { q ->
                        for (i in 0..3) drawLine(Color(0xFF2F80ED),
                            Offset(q[i].x.toFloat() * sx, q[i].y.toFloat() * sy),
                            Offset(q[(i + 1) % 4].x.toFloat() * sx, q[(i + 1) % 4].y.toFloat() * sy), 5f)
                    }
                    cands.forEachIndexed { i, c ->
                        val sel = i in selected
                        drawCircle(if (sel) Good else Amber, (c.d / 2 * sx).toFloat() + 10f,
                            Offset(c.x.toFloat() * sx, c.y.toFloat() * sy), style = Stroke(if (sel) 7f else 3f))
                    }
                }
            }
            }
            if (hasQr) Text("已选 ${selected.size} 个圆点" + if (selected.size < 3) "（少于 3 个时主要靠二维码和图像特征对齐）" else "",
                Modifier.padding(12.dp), color = if (selected.size >= 3) Good else Warn, fontSize = 13.sp)
            if (isNew) OutlinedTextField(name, { name = it }, label = { Text("系列名称") }, singleLine = true,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp))
            if (isNew) Column(Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, top = 12.dp)) {
                Text("这个测点拍什么？（可多选，以后在系列菜单里可改）", fontSize = 14.sp)
                if (com.fieldalign.camera.align.Defects.fromQr(r.qrText).isNotEmpty())
                    Text("已按二维码里写的类型自动选好", fontSize = 12.sp, color = Good)
                KindPicker(kinds) { kinds = it }
            }
            Row(Modifier.fillMaxWidth().padding(12.dp)) {
                OutlinedButton(onClick = { if (isNew) vm.discard(r.file) else vm.back() }, Modifier.weight(1f)) {
                    Text(if (isNew) "丢弃照片" else "取消")
                }
                Spacer(Modifier.width(12.dp))
                Button(onClick = { vm.saveRegister(r, name, selected.sorted().map { cands[it] }, kinds) }, Modifier.weight(1f),
                    enabled = !isNew || name.isNotBlank(),
                    colors = ButtonDefaults.buttonColors(containerColor = Ink)) {
                    Text(if (isNew) "建立系列" else "设为参考")
                }
            }
        }
    }
}

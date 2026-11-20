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
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.QrCode2
import androidx.compose.material.icons.filled.Search
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.fieldalign.camera.AppViewModel
import com.fieldalign.camera.Screen
import com.fieldalign.camera.data.ZoneSummary

@Composable
fun HomeScreen(vm: AppViewModel) {
    val zones by vm.repo.db.zones().summaries().collectAsState(initial = emptyList())
    var query by remember { mutableStateOf("") }
    val pick = androidx.activity.compose.rememberLauncherForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.PickMultipleVisualMedia()) { uris ->
        if (uris.isNotEmpty()) vm.importUris(uris)
    }
    val importPhotos = {
        pick.launch(androidx.activity.result.PickVisualMediaRequest(
            androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia.ImageOnly))
    }
    val shown = if (query.isBlank()) zones else zones.filter {
        it.name.contains(query, true) || (it.qrText?.contains(query, true) ?: false)
    }

    LazyColumn(Modifier.fillMaxSize().background(Page), contentPadding = PaddingValues(bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item(key = "header") {
            Column(Modifier.fillMaxWidth()
                .background(Brush.verticalGradient(listOf(Ink, InkSoft)), RoundedCornerShape(bottomStart = 26.dp, bottomEnd = 26.dp))
                .statusBarsPadding().padding(horizontal = 18.dp, vertical = 18.dp)) {
                Text("复拍对齐相机", color = Color.White, fontSize = 24.sp, fontWeight = FontWeight.Bold)
                Text(if (zones.isEmpty()) "扫码定位 · 自动对齐 · 变化对比"
                     else "${zones.size} 个系列 · ${zones.sumOf { it.photoCount }} 张照片",
                    color = Color(0xFFAAB6C3), fontSize = 13.sp, modifier = Modifier.padding(top = 2.dp))
                Spacer(Modifier.height(16.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    ActionTile("扫码 / 拍照", "自动识别系列", Icons.Default.PhotoCamera, Amber, Ink, Modifier.weight(1f)) {
                        vm.go(Screen.Camera())
                    }
                    ActionTile("导入照片", "相册批量归类", Icons.Default.PhotoLibrary, Color(0x26FFFFFF), Color.White, Modifier.weight(1f)) {
                        importPhotos()
                    }
                }
            }
        }
        if (zones.isEmpty()) {
            item(key = "guide") { GuideCard() }
        } else {
            if (zones.size > 4) item(key = "search") {
                OutlinedTextField(query, { query = it }, Modifier.fillMaxWidth().padding(horizontal = 14.dp),
                    placeholder = { Text("搜索系列名称或编号") }, singleLine = true,
                    leadingIcon = { Icon(Icons.Default.Search, null) }, shape = RoundedCornerShape(14.dp),
                    colors = OutlinedTextFieldDefaults.colors(unfocusedContainerColor = Color.White, focusedContainerColor = Color.White,
                        unfocusedBorderColor = Hairline))
            }
            item(key = "title") {
                Text("我的系列", fontWeight = FontWeight.SemiBold, fontSize = 15.sp, color = Muted,
                    modifier = Modifier.padding(start = 18.dp, top = 6.dp))
            }
            items(shown, key = { it.id }) { z -> ZoneRow(vm, z, Modifier.padding(horizontal = 14.dp)) }
        }
    }
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
            Text(sub, color = fg.copy(alpha = 0.7f), fontSize = 11.sp)
        }
    }
}

/** 首次使用：三步说明（大场景先扫码再后退拍）。 */
@Composable
private fun GuideCard() {
    AppCard(Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 6.dp)) {
        Column(Modifier.padding(18.dp)) {
            Text("怎么用", fontWeight = FontWeight.SemiBold, fontSize = 17.sp)
            Spacer(Modifier.height(12.dp))
            Step(1, "扫码定编号", "靠近二维码扫一下，系列就锁定了。没有二维码也行：按位置和画面特征自动识别。")
            Step(2, "后退拍全景", "锁定后后退，把要监测的整个场景拍进画面，二维码很小甚至不入镜都可以。第一张就是参考照片。")
            Step(3, "复拍自动对比", "以后到现场再扫码，按半透明参考对准拍照，App 自动对齐并标出变化。", last = true)
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
private fun ZoneRow(vm: AppViewModel, z: ZoneSummary, modifier: Modifier) {
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
                    else Pill("无码", Muted)
                    if (z.kinds != null) Pill(com.fieldalign.camera.align.Defects.labels(com.fieldalign.camera.align.Defects.parse(z.kinds)), Muted)
                    else if (z.lat != null) Pill("有位置", Good, icon = Icons.Default.LocationOn)
                }
                Spacer(Modifier.height(4.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("${z.photoCount} 张 · ${fmtTime(z.lastTaken)}", color = Muted, fontSize = 12.sp)
                    val lv = z.lastConclusion?.let { Regex("^【(.+?)】").find(it)?.groupValues?.get(1) }
                    if (lv != null) {                                   // 最近一次的判定等级
                        Spacer(Modifier.width(8.dp))
                        Box(Modifier.size(7.dp).background(levelColor(lv), CircleShape))
                        Text(" 最近：$lv", color = levelColor(lv), fontSize = 12.sp, fontWeight = FontWeight.Medium)
                    } else if (z.lastChanges != null) {
                        Spacer(Modifier.width(8.dp))
                        Box(Modifier.size(7.dp).background(if (z.lastChanges > 0) Warn else Good, CircleShape))
                        Text(if (z.lastChanges > 0) " 最近 ${z.lastChanges} 处变化" else " 最近无变化",
                            color = if (z.lastChanges > 0) Warn else Good, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                    }
                }
            }
            Icon(Icons.Default.ChevronRight, null, tint = Color(0xFFB0B8C1))
        }
    }
}

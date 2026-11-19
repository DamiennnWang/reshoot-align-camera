package com.fieldalign.camera.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
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
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.fieldalign.camera.AppViewModel
import com.fieldalign.camera.data.Outcome

/** 无码且无法唯一识别的照片：手动选系列、新建系列或丢弃。 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun UnclassifiedScreen(vm: AppViewModel, u: Outcome.Unclassified) {
    val bmp = rememberBitmap(u.file.absolutePath) { vm.loadBitmap(u.file.absolutePath, 1200) }
    val all by vm.repo.db.zones().summaries().collectAsState(initial = emptyList())
    val ranked = u.ranking.associate { it.first.id to it.second }
    val dist = u.distances
    Scaffold(containerColor = Page, topBar = { AppTopBar("未能自动归类", "请选择系列，或新建", onBack = { vm.discard(u.file) }) }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).verticalScroll(rememberScrollState()).padding(12.dp)) {
            Text(u.message, fontWeight = FontWeight.Medium)
            Spacer(Modifier.padding(4.dp))
            bmp?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxWidth().heightIn(max = 320.dp), contentScale = ContentScale.Fit) }
            Spacer(Modifier.padding(6.dp))
            Text(if (dist.isEmpty()) "归入已有系列（按特征相似度排序）：" else "归入已有系列（按特征相似度、距离排序）：",
                fontSize = 14.sp, color = Muted)
            all.sortedWith(compareByDescending<com.fieldalign.camera.data.ZoneSummary> { (ranked[it.id] ?: 0) / 10 }
                .thenBy { dist[it.id] ?: Double.MAX_VALUE }).forEach { z ->
                AppCard(Modifier.fillMaxWidth().padding(vertical = 4.dp), onClick = { vm.assignToZone(u.file, z.id) }) {
                    Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(z.name, fontWeight = FontWeight.SemiBold)
                            Text(listOfNotNull(z.qrText?.let { "二维码 $it" } ?: "无码",
                                dist[z.id]?.let { "距 " + com.fieldalign.camera.data.Geo.fmtDistance(it) }).joinToString(" · "),
                                fontSize = 12.sp, color = Muted)
                        }
                        Text("${ranked[z.id] ?: 0} 个匹配点", fontSize = 13.sp,
                            color = if ((ranked[z.id] ?: 0) >= 40) Good else Color.Gray)
                    }
                }
            }
            Spacer(Modifier.padding(8.dp))
            Row {
                OutlinedButton(onClick = { vm.discard(u.file) }, Modifier.weight(1f)) { Text("丢弃") }
                Spacer(Modifier.width(12.dp))
                Button(onClick = { vm.newSeriesFrom(u) }, Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = Ink)) { Text("新建系列") }
            }
        }
    }
}

package com.fieldalign.camera.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material3.Icon
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.fieldalign.camera.AppViewModel
import com.fieldalign.camera.Screen
import com.fieldalign.camera.data.Geo
import com.fieldalign.camera.data.GeoFix
import com.fieldalign.camera.data.GeoTracker
import kotlinx.coroutines.launch
import kotlin.math.cos
import kotlin.math.max

// ====================================================================== 待确认
/** 批量复拍时没能归类的照片：点一下给出候选（图像相似度 + 距离），归入或新建；也可以全部建为新测点。 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PendingScreen(vm: AppViewModel) {
    val list by vm.repo.db.photos().pending().collectAsState(initial = emptyList())
    val scope = rememberCoroutineScope()
    var confirmAll by remember { mutableStateOf(false) }
    Scaffold(containerColor = Page, topBar = {
        AppTopBar("待确认", if (list.isEmpty()) "没有待确认的照片" else "${list.size} 张：点一下选择归入或新建", onBack = { vm.back() }) {
            if (list.isNotEmpty()) TextButton(onClick = { confirmAll = true }) { Text("全部新建") }
        }
    }) { pad ->
        if (list.isEmpty()) Box(Modifier.fillMaxSize().padding(pad), contentAlignment = Alignment.Center) {
            Text("批量复拍时认不出的照片会放在这里，不会乱归。", color = Muted, modifier = Modifier.padding(32.dp))
        } else LazyColumn(Modifier.fillMaxSize().padding(pad).padding(horizontal = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(list, key = { it.id }) { p ->
                val bmp = rememberBitmap(p.path) { vm.loadBitmap(p.path, 300) }
                AppCard(Modifier.fillMaxWidth(), onClick = { vm.openPending(p) }) {
                    Row(Modifier.padding(10.dp), verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(84.dp).clip(RoundedCornerShape(10.dp)).background(Color(0xFFDDE1E6))) {
                            bmp?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
                        }
                        Spacer(Modifier.width(12.dp))
                        Column(Modifier.weight(1f)) {
                            Text(fmtTime(p.takenAt), fontWeight = FontWeight.SemiBold)
                            Text(p.message, color = Muted, fontSize = 12.sp, maxLines = 2, overflow = TextOverflow.Ellipsis)
                            Text(listOfNotNull(p.locAcc?.let { "定位 ±${it.toInt()} 米" } ?: "无定位",
                                p.azimuth?.let { "朝向 ${Geo.fmtAzimuth(it)}" }).joinToString(" · "), color = Muted, fontSize = 12.sp)
                        }
                    }
                }
            }
        }
    }
    if (confirmAll) androidx.compose.material3.AlertDialog(onDismissRequest = { confirmAll = false },
        title = { Text("全部建为新测点？") },
        text = { Text("${list.size} 张照片各建一个新测点（名称“测点 月日-序号”，类型用批量建档的默认类型）。之后可以在测点页改名、合并。") },
        confirmButton = { TextButton(onClick = { confirmAll = false; scope.launch {
            for (p in list) vm.repo.pendingToNewZone(p, vm.batchKinds)
            vm.toast = "已新建 ${list.size} 个测点"
        } }) { Text("全部新建") } },
        dismissButton = { TextButton(onClick = { confirmAll = false }) { Text("取消") } })
}

// ====================================================================== 设置
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(vm: AppViewModel) {
    val zones by vm.repo.db.zones().summaries().collectAsState(initial = emptyList())
    val ctx = LocalContext.current
    val ver = remember { runCatching { ctx.packageManager.getPackageInfo(ctx.packageName, 0).versionName }.getOrNull() ?: "" }
    Scaffold(containerColor = Page, topBar = { AppTopBar("设置", onBack = { vm.back() }) }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).verticalScroll(rememberScrollState()).padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Section("帮助") {
                NavRow("使用教程", "图文说明：贴靶标、拍照、复拍、看结果") { vm.go(Screen.Tutorial) }
                Spacer(Modifier.height(6.dp))
                NavRow("电脑端处理", "导出包查看器、在线重新分析、批量生成靶标") { vm.go(Screen.Planned("电脑端处理", PC_HELP)) }
            }
            Section("靶标与编号") {
                NavRow("生成靶标", "批量打印编号二维码（T0001、T0002…）和圆点") { vm.go(Screen.Targets) }
                Spacer(Modifier.height(6.dp))
                NavRow("编号对应", "靶标编号 → 项目编号 / 名称，可导入导出 CSV") { vm.go(Screen.CodeMap) }
            }
            Section("新测点的默认病害类型") {
                Text("新测点默认用这个类型；二维码里写了类型的按码（如 B01-Z023;40mm;螺栓）。每个测点之后都能单独改。", color = Muted, fontSize = 13.sp)
                Spacer(Modifier.height(6.dp))
                KindPicker(vm.batchKinds) { vm.updateBatchKinds(it) }
            }
            Section("定位与匹配") {
                Text("复拍无码时：先按定位（手机 GPS/北斗，通常 3~10 米）和镜头朝向挑出附近的测点，再用图像确认唯一吻合才自动归入，" +
                    "否则进「待确认」。定位只用来缩小范围，最终以图像为准。", color = Muted, fontSize = 13.sp, lineHeight = 19.sp)
                Spacer(Modifier.height(8.dp))
                PlannedRow("高精度定位（外接 RTK / 北斗接收机）", "蓝牙连接厘米级接收机 + 网络差分（CORS），测点位置精确到厘米，适合测点密集、外观相似的场景") {
                    vm.go(Screen.Planned("高精度定位", RTK_HELP))
                }
            }
            Section("数据") {
                Text("${zones.size} 个测点 · ${zones.sumOf { it.photoCount }} 张照片", fontSize = 14.sp)
                Spacer(Modifier.height(6.dp))
                PlannedRow("自动备份 / 云端同步", "规划中") { vm.go(Screen.Planned("云端同步", CLOUD_HELP)) }
                PlannedRow("报告模板（单位、工程名、签名）", "规划中：电脑端网页已可打印报告") {
                    vm.go(Screen.Planned("报告模板", "规划中：在设置里填写单位、工程名称、检查人，导出和电脑端报告自动带上。\n\n目前电脑端网页（web/index.html）打开导出包即可打印报告。"))
                }
            }
            Section("关于") {
                Text("复拍对齐相机 $ver", fontSize = 14.sp)
                Text("手机端给大致判断；精细分析请导出后在电脑上处理。", color = Muted, fontSize = 13.sp)
            }
        }
    }
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    AppCard(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp)) {
            Text(title, fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
            Spacer(Modifier.height(8.dp))
            content()
        }
    }
}

@Composable
private fun NavRow(title: String, sub: String, onClick: () -> Unit) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(Page).clickable(onClick = onClick).padding(12.dp),
        verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, fontSize = 14.sp, fontWeight = FontWeight.Medium)
            Text(sub, fontSize = 12.sp, color = Muted)
        }
        Icon(Icons.Default.ChevronRight, null, tint = Muted)
    }
}

@Composable
private fun PlannedRow(title: String, sub: String, onClick: () -> Unit) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(Page).padding(10.dp)
        .let { m -> m }, verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, fontSize = 14.sp, fontWeight = FontWeight.Medium)
            Text(sub, fontSize = 12.sp, color = Muted)
        }
        TextButton(onClick = onClick) { Text("了解") }
    }
}

// ====================================================================== 地图（简易）
/** 测点分布：按经纬度画点（不含底图，不依赖地图服务），蓝点是当前位置；下面按距离列出。点一下进测点。 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MapScreen(vm: AppViewModel) {
    val ctx = LocalContext.current
    val zones by vm.repo.db.zones().summaries().collectAsState(initial = emptyList())
    var here by remember { mutableStateOf<GeoFix?>(null) }
    val tracker = remember { GeoTracker(ctx) { here = it } }
    DisposableEffect(Unit) { tracker.start(); onDispose { tracker.stop() } }
    val pts = zones.filter { it.lat != null && it.lon != null }
    Scaffold(containerColor = Page, topBar = {
        AppTopBar("测点分布", "${pts.size}/${zones.size} 个测点有位置" + (here?.let { " · 定位 ±${it.acc.toInt()} 米" } ?: ""), onBack = { vm.back() })
    }) { pad ->
        LazyColumn(Modifier.fillMaxSize().padding(pad).padding(horizontal = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            item {
                if (pts.isEmpty()) Text("还没有带位置的测点（拍照时打开定位即可记录）。", color = Muted, modifier = Modifier.padding(16.dp))
                else AppCard(Modifier.fillMaxWidth()) {
                    val all = pts.map { it.lat!! to it.lon!! } + listOfNotNull(here?.let { it.lat to it.lon })
                    val lat0 = all.minOf { it.first }; val lat1 = all.maxOf { it.first }
                    val lon0 = all.minOf { it.second }; val lon1 = all.maxOf { it.second }
                    val k = cos(Math.toRadians((lat0 + lat1) / 2))
                    val spanM = max(max((lat1 - lat0) * 111_000, (lon1 - lon0) * 111_000 * k), 20.0)
                    fun proj(lat: Double, lon: Double, w: Float, h: Float): Offset {
                        val s = (minOf(w, h) * 0.85f) / spanM.toFloat()
                        val cx = ((lon0 + lon1) / 2); val cy = ((lat0 + lat1) / 2)
                        return Offset(w / 2 + ((lon - cx) * 111_000 * k).toFloat() * s, h / 2 - ((lat - cy) * 111_000).toFloat() * s)
                    }
                    Column(Modifier.padding(10.dp)) {
                        Box(Modifier.fillMaxWidth().aspectRatio(1.2f).clip(RoundedCornerShape(10.dp)).background(Color(0xFFE9EDF1))) {
                            Canvas(Modifier.fillMaxSize().pointerInput(pts) {
                                detectTapGestures { t ->
                                    val near = pts.minByOrNull { (proj(it.lat!!, it.lon!!, size.width.toFloat(), size.height.toFloat()) - t).getDistance() }
                                    if (near != null && (proj(near.lat!!, near.lon!!, size.width.toFloat(), size.height.toFloat()) - t).getDistance() < 60f)
                                        vm.go(Screen.ZoneDetail(near.id))
                                }
                            }) {
                                pts.forEach { z ->
                                    val p = proj(z.lat!!, z.lon!!, size.width, size.height)
                                    val lv = z.lastConclusion?.let { Regex("^【(.+?)】").find(it)?.groupValues?.get(1) }
                                    drawCircle(lv?.let { levelColor(it) } ?: Ink, 11f, p)
                                    drawCircle(Color.White, 11f, p, style = Stroke(3f))
                                }
                                here?.let {
                                    val p = proj(it.lat, it.lon, size.width, size.height)
                                    drawCircle(Info.copy(alpha = 0.25f), 26f, p); drawCircle(Info, 9f, p)
                                }
                            }
                            Text("范围约 ${Geo.fmtDistance(spanM)} · 颜色 = 最近一次判定", fontSize = 11.sp, color = Muted,
                                modifier = Modifier.align(Alignment.BottomStart).padding(8.dp))
                        }
                        Text("简易分布图（不含底图）。接入地图底图、按路线导航到测点：规划中。", fontSize = 12.sp, color = Muted, modifier = Modifier.padding(top = 6.dp))
                    }
                }
            }
            val sorted = zones.sortedBy { z -> here?.let { h -> if (z.lat != null && z.lon != null) Geo.distance(h.lat, h.lon, z.lat, z.lon) else null } ?: Double.MAX_VALUE }
            items(sorted, key = { it.id }) { z -> ZoneRow(vm, z, Modifier) }
        }
    }
}

// ====================================================================== 规划中 / 说明
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlannedScreen(vm: AppViewModel, title: String, body: String) {
    Scaffold(containerColor = Page, topBar = { AppTopBar(title, onBack = { vm.back() }) }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).verticalScroll(rememberScrollState()).padding(16.dp)) {
            AppCard(Modifier.fillMaxWidth()) {
                Text(body, fontSize = 14.sp, lineHeight = 22.sp, modifier = Modifier.padding(16.dp))
            }
            Spacer(Modifier.height(16.dp))
            Button(onClick = { vm.back() }, colors = ButtonDefaults.buttonColors(containerColor = Ink), modifier = Modifier.fillMaxWidth()) { Text("知道了") }
        }
    }
}

const val RTK_HELP = """手机自带的 GPS/北斗定位一般 3~10 米，城市峡谷、隧道口、桥下更差；隧道里没有信号。

要做到厘米级，需要 RTK（实时差分）：
1. 外接 RTK 接收机（几千元的手持/背包式，支持北斗三号、双频），通过蓝牙把 NMEA 定位数据发给手机；
2. 接收机或手机连网络差分服务（CORS，如千寻位置、中国移动高精度定位，按年付费），用 NTRIP 协议取改正数；
3. 固定解时平面精度 1~2 厘米。App 读取蓝牙 NMEA，把测点位置存成厘米级。

IMU（手机里的陀螺仪、加速度计）：
· 适合测镜头朝向和俯仰（App 已在用），帮助区分同一位置不同方向的测点；
· 不适合单独定位：积分误差几秒钟就漂到米级；只能在 RTK 短暂失锁（桥下、树下）时补几秒。
· 隧道内：没有卫星，靠二维码 + 图像匹配最可靠，可配合里程桩号。

本 App 的建议做法：二维码（最可靠）> 位置 + 朝向缩小范围 + 图像确认 > 人工确认。复拍时半透明参考对准本身就能把相机位置对到厘米级，所以普通测点不必上 RTK；测点很密、外观相似、又不能贴码时再考虑。

外接 RTK 接入：规划中。"""

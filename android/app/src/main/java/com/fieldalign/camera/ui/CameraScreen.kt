package com.fieldalign.camera.ui

import android.Manifest
import android.graphics.Bitmap
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.LayersClear
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.LocationOff
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.QrCode2
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.fieldalign.camera.AppViewModel
import com.fieldalign.camera.align.FrameGuide
import com.fieldalign.camera.data.Geo
import com.fieldalign.camera.data.GeoFix
import com.fieldalign.camera.data.GeoTracker
import java.util.concurrent.Executors
import kotlin.math.roundToInt

/**
 * 相机页。测点的确定方式（优先级从高到低）：
 *  1. 手动选择；
 *  2. 扫码锁定 —— 扫到码后测点就锁定，之后可以后退拍大场景，照片里二维码很小甚至不入镜都可以；
 *     扫到库里没有的码时锁定为“新编号”，下一张照片作为它的参考；
 *  3. 自动识别 —— 无码时按 位置（附近的测点）+ 画面特征 识别。
 */
@Composable
fun CameraScreen(vm: AppViewModel, initialZoneId: Long?, mode: com.fieldalign.camera.data.CaptureMode = com.fieldalign.camera.data.CaptureMode.RESHOOT) {
    val batch = initialZoneId == null                                  // 批量拍摄：拍完不跳页
    val isNew = batch && mode == com.fieldalign.camera.data.CaptureMode.NEW
    var kindsOpen by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { if (batch) vm.resetBatch() }
    val ctx = LocalContext.current
    val lifecycle = LocalLifecycleOwner.current
    val zones by vm.repo.db.zones().summaries().collectAsState(initial = emptyList())

    var live by remember { mutableStateOf(FrameGuide.LiveState()) }
    var lockZone by remember { mutableStateOf(initialZoneId) }          // 锁定的测点（手动或扫码）
    var lockByScan by remember { mutableStateOf(false) }
    var lockNewQr by remember { mutableStateOf<String?>(null) }         // 扫到的新码（库里没有）
    var suggestZone by remember { mutableStateOf<Long?>(null) }         // 按特征/位置自动识别的测点
    val targetZone = lockZone ?: suggestZone
    var ghost by remember { mutableStateOf<Bitmap?>(null) }
    var showGhost by remember { mutableStateOf(true) }
    var ghostAlpha by remember { mutableFloatStateOf(0.35f) }
    var pickerOpen by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    var geo by remember { mutableStateOf<GeoFix?>(null) }
    var geoAllowed by remember { mutableStateOf(Geo.hasPermission(ctx)) }

    val tracker = remember { GeoTracker(ctx) { geo = it } }
    val poseTracker = remember { com.fieldalign.camera.data.PoseTracker(ctx) }
    var pose by remember { mutableStateOf<com.fieldalign.camera.data.Pose?>(null) }
    DisposableEffect(Unit) { poseTracker.start(); onDispose { poseTracker.stop() } }
    LaunchedEffect(Unit) { while (true) { pose = poseTracker.latest; kotlinx.coroutines.delay(250) } }
    val askGeo = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        geoAllowed = Geo.hasPermission(ctx); if (geoAllowed) tracker.start()
    }
    DisposableEffect(Unit) {
        if (geoAllowed) tracker.start()
        else askGeo.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION))
        onDispose { tracker.stop() }
    }

    val previewView = remember {
        PreviewView(ctx).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    val analyzer = remember { FrameGuide { s -> live = s } }
    val analysisExec = remember { Executors.newSingleThreadExecutor() }
    val capture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .setResolutionSelector(ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build())
            .build()
    }

    // 扫到二维码 → 锁定对应测点（手动选择优先，不被覆盖）。离开二维码后锁定保持，方便后退拍全景。
    LaunchedEffect(live.qrText) {
        val q = live.qrText ?: return@LaunchedEffect
        if (lockZone != null && !lockByScan) return@LaunchedEffect
        val id = vm.zoneIdForQr(q)
        if (id != null) { lockZone = id; lockByScan = true; lockNewQr = null }
        else { lockZone = null; lockByScan = false; lockNewQr = q }
        suggestZone = null
    }
    // 无码：附近的测点里按画面特征自动识别（位置每变 ~100 米刷新一次候选）
    val geoKey = geo?.let { (it.lat * 1000).roundToInt() to (it.lon * 1000).roundToInt() }
    LaunchedEffect(geoKey) { analyzer.candidates = if (isNew) emptyList() else vm.guideCandidates(tracker.fresh()) }
    LaunchedEffect(live.suggestedZone) {
        val z = live.suggestedZone ?: return@LaunchedEffect
        if (lockZone == null && lockNewQr == null && !isNew) suggestZone = z
    }
    // 目标测点变化 → 准备引导数据和参考半透明图
    LaunchedEffect(targetZone) {
        val z = targetZone
        if (z == null) { analyzer.target = null; ghost = null; return@LaunchedEffect }
        val t = vm.guideTarget(z)
        analyzer.target = t?.first
        ghost = t?.second?.let { vm.loadBitmap(it, 1000) }
    }

    DisposableEffect(Unit) {
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            val provider = future.get()
            val ratio = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build()
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(ResolutionStrategy(Size(640, 480),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                .build().also { it.setAnalyzer(analysisExec, analyzer) }
            val preview = Preview.Builder().setResolutionSelector(ratio).build()
            preview.setSurfaceProvider(previewView.surfaceProvider)
            provider.unbindAll()
            provider.bindToLifecycle(lifecycle, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, capture)
        }, ContextCompat.getMainExecutor(ctx))
        onDispose {
            runCatching { future.get().unbindAll() }
            analysisExec.shutdown()
        }
    }

    fun distTo(lat: Double?, lon: Double?): Double? {
        val g = geo ?: return null
        return if (lat == null || lon == null) null else Geo.distance(g.lat, g.lon, lat, lon)
    }
    val target = zones.firstOrNull { it.id == targetZone }
    val targetDist = target?.let { distTo(it.lat, it.lon) }
    val unlock = { lockZone = null; lockByScan = false; lockNewQr = null; suggestZone = null }

    // 参考画面在当前取景中的面积（1 = 一样大）。明显小于 1：当前比参考拍得更大范围（后退拍全景、参考是特写）
    val refArea = live.quad?.let { com.fieldalign.camera.align.convexHullArea(it) }
    val wider = refArea != null && refArea < 0.5
    // 有目标却持续找不到参考画面：多半是锁错了测点，或参考与现场差别太大
    var lostSince by remember { mutableStateOf(0L) }
    val lostNow = targetZone != null && live.quad == null && lockNewQr == null
    if (!lostNow) lostSince = 0L else if (lostSince == 0L) lostSince = System.currentTimeMillis()
    val lostLong = lostNow && lostSince > 0 && System.currentTimeMillis() - lostSince > 3000

    // 提示文案：新码 / 比参考范围大 / 长时间对不上 / 引导
    val hint = when {
        lockNewQr != null -> "后退，把整个场景拍进画面\n这张将作为「$lockNewQr」的参考（二维码可以不入镜）"
        isNew && lockZone != null -> "这个二维码已建档：「${target?.name ?: ""}」\n拍照将作为它的复拍"
        isNew -> "对准测点拍照：每张建一个新测点\n有二维码时按码建档，名称可以之后再改"
        wider -> "比参考照片拍得更大范围（参考只占 ${(refArea!! * 100).roundToInt()}%）\n可以直接拍，拍完可设为新的参考"
        lostLong -> "画面和「${target?.name ?: ""}」的参考对不上\n如果锁错了，点上方「解锁」"
        live.hint.isNotEmpty() -> live.hint
        targetZone == null -> when {
            !batch -> "对准二维码，或对准拍摄对象自动识别"
            mode == com.fieldalign.camera.data.CaptureMode.AUTO -> "对准测点拍照，一张或多张都行\n认识的自动归入，新编号、新位置自动建档"
            else -> "对准测点拍照：有二维码按码归档\n无码按 位置 + 朝向 + 图像 匹配，认不出的进“待确认”"
        }
        else -> "正在寻找参考画面…"
    }
    val hintBg by animateColorAsState(if (live.aligned) Good else Color(0xCC111820), label = "hint")

    Column(Modifier.fillMaxSize().background(Color.Black)) {
        // ---------------- 顶栏：返回、测点选择、参考叠加开关
        Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "返回", tint = Color.White) }
            if (isNew) Box(Modifier.weight(1f)) {
                Row(Modifier.clip(RoundedCornerShape(50)).background(Color(0x26FFFFFF)).clickable { kindsOpen = true }
                    .padding(start = 12.dp, end = 6.dp, top = 7.dp, bottom = 7.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text("批量建档 · ${com.fieldalign.camera.align.Defects.labels(vm.batchKinds)}", color = Color.White, fontSize = 15.sp,
                        fontWeight = FontWeight.Medium, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.widthIn(max = 220.dp))
                    Icon(Icons.Default.ArrowDropDown, null, tint = Color.White)
                }
            } else Box(Modifier.weight(1f)) {
                Row(Modifier.clip(RoundedCornerShape(50)).background(Color(0x26FFFFFF)).clickable { pickerOpen = true }
                    .padding(start = 12.dp, end = 6.dp, top = 7.dp, bottom = 7.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(when {
                        lockZone != null -> Icons.Default.Lock
                        lockNewQr != null -> Icons.Default.QrCode2
                        else -> Icons.Default.AutoAwesome
                    }, null, Modifier.size(16.dp), tint = if (lockZone != null || lockNewQr != null) Amber else Color.White)
                    Spacer(Modifier.width(6.dp))
                    Text(when {
                        lockNewQr != null -> "新编号 $lockNewQr"
                        target != null -> target.name
                        else -> "自动识别测点"
                    }, color = Color.White, fontSize = 15.sp, fontWeight = FontWeight.Medium, maxLines = 1,
                        overflow = TextOverflow.Ellipsis, modifier = Modifier.widthIn(max = 190.dp))
                    Icon(Icons.Default.ArrowDropDown, null, tint = Color.White)
                }
                DropdownMenu(expanded = pickerOpen, onDismissRequest = { pickerOpen = false }) {
                    DropdownMenuItem(text = { Text("自动识别（二维码 / 位置 + 图像特征）") },
                        leadingIcon = { Icon(Icons.Default.AutoAwesome, null) },
                        onClick = { unlock(); pickerOpen = false })
                    HorizontalDivider()
                    // 有定位时按距离排序，最近的在前
                    zones.map { it to distTo(it.lat, it.lon) }.sortedBy { it.second ?: Double.MAX_VALUE }.forEach { (z, d) ->
                        DropdownMenuItem(
                            text = {
                                Column {
                                    Text(z.name, fontWeight = FontWeight.Medium)
                                    Text(listOfNotNull(z.qrText?.let { "码 $it" } ?: "无码", d?.let { "距 ${Geo.fmtDistance(it)}" },
                                        z.azimuth?.let { "朝向 ${Geo.fmtAzimuth(it)}" })
                                        .joinToString(" · "), fontSize = 12.sp, color = Muted)
                                }
                            },
                            onClick = { lockZone = z.id; lockByScan = false; lockNewQr = null; pickerOpen = false })
                    }
                }
            }
            if (ghost != null) IconButton(onClick = { showGhost = !showGhost }) {
                Icon(if (showGhost) Icons.Default.Layers else Icons.Default.LayersClear, "参考叠加", tint = Color.White)
            }
        }

        // ---------------- 取景区：3:4，预览 + 参考半透明 + 参考取景框 + 浮层
        Box(Modifier.fillMaxWidth().aspectRatio(3f / 4f)) {
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
            if (showGhost) ghost?.let { g ->
                val q = live.quad
                if (wider && q != null) {
                    // 当前比参考范围大：把参考图贴到它在画面中的实际位置，叠加才对得上
                    Canvas(Modifier.fillMaxSize()) {
                        val src = floatArrayOf(0f, 0f, g.width.toFloat(), 0f, g.width.toFloat(), g.height.toFloat(), 0f, g.height.toFloat())
                        val dst = FloatArray(8) { i -> if (i % 2 == 0) q[i / 2].x.toFloat() * size.width else q[i / 2].y.toFloat() * size.height }
                        val mtx = android.graphics.Matrix().apply { setPolyToPoly(src, 0, dst, 0, 4) }
                        val paint = android.graphics.Paint().apply { alpha = (ghostAlpha * 255).toInt(); isFilterBitmap = true }
                        drawContext.canvas.nativeCanvas.drawBitmap(g, mtx, paint)
                    }
                } else Image(g.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Fit, alpha = ghostAlpha)
            }
            Canvas(Modifier.fillMaxSize()) {
                // 三分线，帮助构图
                val c = Color(0x33FFFFFF)
                for (i in 1..2) {
                    drawLine(c, Offset(size.width * i / 3, 0f), Offset(size.width * i / 3, size.height), 1.5f)
                    drawLine(c, Offset(0f, size.height * i / 3), Offset(size.width, size.height * i / 3), 1.5f)
                }
                val q = live.quad ?: return@Canvas
                val path = Path().apply {
                    moveTo(q[0].x.toFloat() * size.width, q[0].y.toFloat() * size.height)
                    for (i in 1..3) lineTo(q[i].x.toFloat() * size.width, q[i].y.toFloat() * size.height)
                    close()
                }
                val col = if (live.aligned) Good else Amber
                drawPath(path, col, style = Stroke(width = 5f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(24f, 14f))))
                drawCircle(col, 10f, Offset(q[0].x.toFloat() * size.width, q[0].y.toFloat() * size.height))
            }

            // 锁定状态条（顶部）
            if (lockZone != null || lockNewQr != null) Box(Modifier.align(Alignment.TopCenter).padding(10.dp)) {
                Row(Modifier.clip(RoundedCornerShape(14.dp)).background(Color(0xE6111820))
                    .padding(start = 12.dp, end = 4.dp, top = 6.dp, bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Lock, null, Modifier.size(16.dp), tint = Amber)
                    Spacer(Modifier.width(8.dp))
                    Column(Modifier.widthIn(max = 230.dp)) {
                        Text(when {
                            lockNewQr != null -> "已扫码：新编号「$lockNewQr」"
                            lockByScan -> "已扫码锁定：${target?.name ?: ""}"
                            else -> "已选择：${target?.name ?: ""}"
                        }, color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.Medium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text(when {
                            lockNewQr != null -> "后退拍全景作为参考，二维码不必入镜"
                            targetDist != null -> "距参考拍摄点 ${Geo.fmtDistance(targetDist)} · 可后退按参考取景"
                            lockByScan -> "可离开二维码，后退按参考画面取景"
                            else -> "按半透明参考画面取景"
                        }, color = Color(0xFFB8C2CC), fontSize = 11.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                    Text("解锁", color = Amber, fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.clip(RoundedCornerShape(10.dp)).clickable { unlock() }.padding(horizontal = 10.dp, vertical = 6.dp))
                }
            }

            // 引导提示（底部）
            Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                .background(Brush.verticalGradient(listOf(Color.Transparent, Color(0x99000000))))
                .padding(horizontal = 16.dp, vertical = 12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(hint, color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.SemiBold, textAlign = TextAlign.Center,
                    lineHeight = 22.sp,
                    modifier = Modifier.clip(RoundedCornerShape(12.dp)).background(hintBg).padding(horizontal = 14.dp, vertical = 7.dp))
                if (live.source.isNotEmpty())
                    Text("引导依据：${live.source}" + (live.error?.let { "  ·  偏差 %.0f%%".format(it * 100) } ?: ""),
                        color = Color(0xFFCBD2D9), fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp))
            }
        }

        // ---------------- 底部：定位状态、透明度、快门
        Column(Modifier.fillMaxWidth().weight(1f).padding(horizontal = 20.dp), horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.SpaceEvenly) {
            if (batch) Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(listOfNotNull("新建 ${vm.batchCreated}", "归入 ${vm.batchMatched}", "待确认 ${vm.batchPending}",
                    if (vm.batchQueued > 0) "处理中 ${vm.batchQueued}" else null).joinToString("  ·  "),
                    color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.Medium)
                if (mode == com.fieldalign.camera.data.CaptureMode.AUTO)
                    Text("新测点类型：${com.fieldalign.camera.align.Defects.labels(vm.batchKinds)} ▾", color = Color(0xFF9AA5B1), fontSize = 11.sp,
                        modifier = Modifier.padding(top = 2.dp).clip(RoundedCornerShape(8.dp)).clickable { kindsOpen = true }.padding(horizontal = 6.dp, vertical = 2.dp))
                vm.batchLast?.let {
                    Text(it, color = if (it.startsWith("待确认") || it.startsWith("处理失败") || it.contains("⚠")) Amber else Color(0xFFB8C2CC), fontSize = 12.sp,
                        maxLines = 1, overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.padding(top = 2.dp).clickable(enabled = vm.batchLastZone != null) {
                            vm.batchLastZone?.let { z -> vm.go(com.fieldalign.camera.Screen.ZoneDetail(z)) } })
                }
            }
            if (ghost != null && showGhost) Row(verticalAlignment = Alignment.CenterVertically) {
                Text("参考透明度", color = Color(0xFF9AA5B1), fontSize = 12.sp)
                Spacer(Modifier.width(8.dp))
                Slider(ghostAlpha, { ghostAlpha = it }, valueRange = 0f..0.8f, modifier = Modifier.width(200.dp),
                    colors = SliderDefaults.colors(thumbColor = Amber, activeTrackColor = Amber, inactiveTrackColor = Color(0x44FFFFFF)))
            }
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                // 左：定位状态
                Column(Modifier.weight(1f)) {
                    val g = geo
                    Row(Modifier.clip(RoundedCornerShape(50)).background(Color(0x1FFFFFFF))
                        .clickable(enabled = !geoAllowed) {
                            askGeo.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION))
                        }.padding(horizontal = 10.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(if (g != null) Icons.Default.LocationOn else Icons.Default.LocationOff, null, Modifier.size(15.dp),
                            tint = if (g != null) Good else Color(0xFF9AA5B1))
                        Spacer(Modifier.width(4.dp))
                        Text(when {
                            !geoAllowed -> "开启定位"
                            g == null -> "定位中…"
                            else -> "±${g.acc.roundToInt()} 米"
                        }, color = Color(0xFFCBD2D9), fontSize = 12.sp)
                    }
                    // 镜头朝向：同一点位不同方向的照片属于不同测点
                    val ps = pose
                    Text(when {
                        ps == null -> "朝向：无传感器"
                        ps.azimuth == null -> "俯仰 ${ps.pitch.roundToInt()}° · 指南针需校准（手机画 8 字）"
                        else -> "朝向 ${Geo.fmtAzimuth(ps.azimuth)} · 俯仰 ${ps.pitch.roundToInt()}°"
                    }, color = Color(0xFF9AA5B1), fontSize = 11.sp, modifier = Modifier.padding(start = 4.dp, top = 4.dp))
                    val ta = target?.azimuth; val tp = target?.pitch
                    if (ps?.azimuth != null && ta != null) {
                        val d = ((ps.azimuth - ta + 540) % 360) - 180
                        if (kotlin.math.abs(d) > 15) Text(if (d > 0) "比参考偏右 ${d.roundToInt()}°，向左转" else "比参考偏左 ${(-d).roundToInt()}°，向右转",
                            color = Amber, fontSize = 11.sp, modifier = Modifier.padding(start = 4.dp))
                    } else if (ps != null && tp != null && kotlin.math.abs(ps.pitch - tp) > 15) {
                        Text(if (ps.pitch > tp) "比参考仰得多，镜头往下" else "比参考俯得多，镜头往上", color = Amber, fontSize = 11.sp,
                            modifier = Modifier.padding(start = 4.dp))
                    }
                }
                // 中：快门
                val ring = if (live.aligned) Good else Color.White
                Box(
                    Modifier.size(78.dp).border(4.dp, ring, CircleShape).padding(7.dp).clip(CircleShape)
                        .background(if (busy) Color.Gray else if (live.aligned) Good else Color.White)
                        .clickable(enabled = !busy) {
                            busy = true
                            val f = vm.repo.newPhotoFile()
                            capture.takePicture(ImageCapture.OutputFileOptions.Builder(f).build(),
                                ContextCompat.getMainExecutor(ctx), object : ImageCapture.OnImageSavedCallback {
                                    override fun onImageSaved(o: ImageCapture.OutputFileResults) {
                                        busy = false
                                        if (batch) {
                                            vm.onBatchCaptured(f, mode, tracker.fresh(), poseTracker.latest,
                                                forcedZoneId = lockZone, newQr = lockNewQr)
                                            // 扫码锁定只管这一张：下一张多半是另一个测点
                                            if (lockByScan || lockNewQr != null) unlock()
                                        } else vm.onCaptured(f, forcedZoneId = lockZone,
                                            hintZoneId = if (lockByScan) lockZone else suggestZone,
                                            newQr = lockNewQr, loc = tracker.fresh(), pose = poseTracker.latest)
                                    }
                                    override fun onError(e: ImageCaptureException) {
                                        busy = false; vm.toast = "拍照失败：${e.message}"
                                    }
                                })
                        },
                )
                // 右：参考缩略图（有目标时）
                Box(Modifier.weight(1f), contentAlignment = Alignment.CenterEnd) {
                    if (batch) Text("完成", color = Ink, fontWeight = FontWeight.Bold, fontSize = 15.sp,
                        modifier = Modifier.clip(RoundedCornerShape(50)).background(Amber).clickable { vm.back() }
                            .padding(horizontal = 18.dp, vertical = 10.dp))
                    else ghost?.let {
                        Image(it.asImageBitmap(), "参考照片", Modifier.size(52.dp).clip(RoundedCornerShape(10.dp))
                            .border(2.dp, Color(0x66FFFFFF), RoundedCornerShape(10.dp)), contentScale = ContentScale.Crop)
                    }
                }
            }
        }
    }
    if (kindsOpen) androidx.compose.material3.AlertDialog(onDismissRequest = { kindsOpen = false },
        confirmButton = { androidx.compose.material3.TextButton(onClick = { kindsOpen = false }) { Text("好") } },
        title = { Text("新测点的病害类型") },
        text = { Column {
            Text("这一批新建的测点都用这个类型；之后可以在测点页单独修改。二维码里写了类型的按码。", fontSize = 13.sp, color = Muted)
            Spacer(Modifier.width(1.dp).padding(4.dp))
            KindPicker(vm.batchKinds) { vm.updateBatchKinds(it) }
        } })
}

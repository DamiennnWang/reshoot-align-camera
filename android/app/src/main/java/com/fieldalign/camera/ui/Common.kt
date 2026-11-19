package com.fieldalign.camera.ui

import android.graphics.Bitmap
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

val Ink = Color(0xFF16202A)        // 主色：深石板蓝
val InkSoft = Color(0xFF2B3A48)
val Amber = Color(0xFFFFB020)      // 强调：工程黄
val Good = Color(0xFF22A06B)
val Warn = Color(0xFFE08A00)
val Bad = Color(0xFFD64545)
val Info = Color(0xFF2F80ED)
val Muted = Color(0xFF6B7785)
val Page = Color(0xFFF2F4F7)
val Hairline = Color(0xFFE3E7EC)

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(primary = Ink, onPrimary = Color.White, secondary = Amber, onSecondary = Ink,
            background = Page, surface = Color.White, surfaceVariant = Color(0xFFEDF0F3), outline = Hairline),
        shapes = androidx.compose.material3.Shapes(
            small = RoundedCornerShape(10.dp), medium = RoundedCornerShape(16.dp), large = RoundedCornerShape(22.dp)),
        content = content,
    )
}

/** 统一的白底圆角卡片（细描边、无重阴影）。 */
@Composable
fun AppCard(modifier: Modifier = Modifier, onClick: (() -> Unit)? = null, content: @Composable androidx.compose.foundation.layout.ColumnScope.() -> Unit) {
    val shape = RoundedCornerShape(16.dp)
    androidx.compose.material3.Surface(
        modifier = modifier.then(if (onClick != null) Modifier.clip(shape).clickable(onClick = onClick) else Modifier),
        shape = shape, color = Color.White, border = androidx.compose.foundation.BorderStroke(1.dp, Hairline),
    ) { androidx.compose.foundation.layout.Column(content = content) }
}

/** 小标签：图标可选，浅底深字。 */
@Composable
fun Pill(text: String, color: Color = Muted, modifier: Modifier = Modifier, icon: androidx.compose.ui.graphics.vector.ImageVector? = null,
         filled: Boolean = false) {
    androidx.compose.foundation.layout.Row(
        modifier.background(if (filled) color else color.copy(alpha = 0.12f), RoundedCornerShape(50))
            .padding(horizontal = 8.dp, vertical = 3.dp),
        verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
    ) {
        if (icon != null) {
            androidx.compose.material3.Icon(icon, null, Modifier.size(13.dp), tint = if (filled) Color.White else color)
            androidx.compose.foundation.layout.Spacer(Modifier.width(3.dp))
        }
        Text(text, color = if (filled) Color.White else color, fontSize = 12.sp, fontWeight = FontWeight.Medium, maxLines = 1)
    }
}

/** 统一的深色顶栏。 */
@OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
fun AppTopBar(title: String, subtitle: String? = null, onBack: (() -> Unit)? = null,
              actions: @Composable androidx.compose.foundation.layout.RowScope.() -> Unit = {}) {
    androidx.compose.material3.TopAppBar(
        title = {
            androidx.compose.foundation.layout.Column {
                Text(title, maxLines = 1, overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                    fontSize = 19.sp, fontWeight = FontWeight.SemiBold)
                if (subtitle != null) Text(subtitle, fontSize = 12.sp, color = Color(0xFFAAB6C3), maxLines = 1,
                    overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis)
            }
        },
        navigationIcon = {
            if (onBack != null) androidx.compose.material3.IconButton(onClick = onBack) {
                androidx.compose.material3.Icon(androidx.compose.material.icons.Icons.AutoMirrored.Filled.ArrowBack, "返回")
            }
        },
        actions = actions,
        colors = androidx.compose.material3.TopAppBarDefaults.topAppBarColors(containerColor = Ink, titleContentColor = Color.White,
            navigationIconContentColor = Color.White, actionIconContentColor = Color.White),
    )
}

fun statusColor(status: String) = when (status) {
    "confirmed", "features_only" -> Good
    "reference" -> Ink
    "markers_only", "dot_moved" -> Warn
    else -> Bad
}

fun statusShort(status: String) = when (status) {
    "confirmed" -> "已对齐"
    "features_only" -> "特征对齐"
    "reference" -> "参考"
    "markers_only" -> "待确认"
    "dot_moved" -> "圆点异常"
    "conflict" -> "需复核"
    else -> "失败"
}

@Composable
fun StatusChip(status: String, modifier: Modifier = Modifier) {
    Text(
        statusShort(status), color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Medium,
        modifier = modifier.background(statusColor(status), RoundedCornerShape(50)).padding(horizontal = 9.dp, vertical = 2.dp),
    )
}

fun fmtTime(t: Long?): String = if (t == null) "—" else SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.CHINA).format(Date(t))
fun fmtTimeSec(t: Long?): String = if (t == null) "—" else SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.CHINA).format(Date(t))

/**
 * 可双指缩放、拖动的容器；双击复位。所有子内容一起缩放，保证对比时始终对齐。
 * 未放大时单指滑动不拦截，让外层页面可以正常上下滚动。
 */
@Composable
fun ZoomBox(modifier: Modifier = Modifier, content: @Composable BoxScope.() -> Unit) {
    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    Box(
        modifier
            .clipToBounds()
            .pointerInput(Unit) {
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false)
                    do {
                        val event = awaitPointerEvent()
                        val fingers = event.changes.count { it.pressed }
                        if (fingers >= 2 || scale > 1f) {
                            scale = (scale * event.calculateZoom()).coerceIn(1f, 8f)
                            val maxX = (scale - 1f) * size.width / 2f
                            val maxY = (scale - 1f) * size.height / 2f
                            val o = offset + event.calculatePan()
                            offset = Offset(o.x.coerceIn(-maxX, maxX), o.y.coerceIn(-maxY, maxY))
                            event.changes.forEach { if (it.positionChanged()) it.consume() }
                        }
                    } while (event.changes.any { it.pressed })
                }
            }
            .pointerInput(Unit) { detectTapGestures(onDoubleTap = { scale = 1f; offset = Offset.Zero }) },
    ) {
        Box(
            Modifier.graphicsLayer { scaleX = scale; scaleY = scale; translationX = offset.x; translationY = offset.y },
            content = content,
        )
    }
}

/** 异步加载图片。 */
@Composable
fun rememberBitmap(key: Any?, loader: suspend () -> Bitmap?): Bitmap? {
    var bmp by remember(key) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(key) { bmp = loader() }
    return bmp
}

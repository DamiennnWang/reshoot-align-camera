package com.fieldalign.camera.data

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Looper
import androidx.core.content.ContextCompat
import java.io.File

/** 一次定位结果。acc 为精度半径（米）。 */
data class GeoFix(val lat: Double, val lon: Double, val acc: Double, val time: Long = System.currentTimeMillis())

/** 镜头姿态：朝向（北=0°，顺时针）、俯仰（水平=0°，向上为正）。azimuth 为空表示指南针不可靠。 */
data class Pose(val azimuth: Double?, val pitch: Double)

object Geo {
    /** 两点间距离（米）。 */
    fun distance(aLat: Double, aLon: Double, bLat: Double, bLon: Double): Double {
        val r = FloatArray(1)
        Location.distanceBetween(aLat, aLon, bLat, bLon, r)
        return r[0].toDouble()
    }

    /** “附近”的判定半径：定位越不准半径越大；城市里 GPS 常有几十米误差，最少给 80 米。 */
    fun nearRadius(acc: Double?) = maxOf(80.0, (acc ?: 40.0) * 2 + 50)

    fun fmtDistance(m: Double): String = when {
        m < 1000 -> "${m.toInt()} 米"
        m < 10_000 -> "%.1f 公里".format(m / 1000)
        else -> "${(m / 1000).toInt()} 公里"
    }

    /** 导入的照片：读 EXIF 里的 GPS（没有返回 null）。 */
    fun fromExif(file: File): GeoFix? = runCatching {
        val ex = androidx.exifinterface.media.ExifInterface(file.absolutePath)
        val ll = ex.latLong ?: return@runCatching null
        if (ll[0] == 0.0 && ll[1] == 0.0) return@runCatching null
        GeoFix(ll[0], ll[1], 30.0, file.lastModified())
    }.getOrNull()

    /** 两个朝向的夹角（0~180°）。 */
    fun angleDiff(a: Double, b: Double): Double { val d = Math.abs(((a - b) % 360 + 360) % 360); return if (d > 180) 360 - d else d }

    /** 朝向是否吻合：同一点位不同方向的照片属于不同系列。朝向 ±60°、俯仰 ±30° 以内算同一方向；缺数据时不排除。 */
    fun poseCompatible(az1: Double?, p1: Double?, az2: Double?, p2: Double?): Boolean {
        if (az1 != null && az2 != null && angleDiff(az1, az2) > 60) return false
        if (p1 != null && p2 != null && Math.abs(p1 - p2) > 30) return false
        return true
    }

    fun fmtAzimuth(a: Double): String {
        val names = listOf("北", "东北", "东", "东南", "南", "西南", "西", "西北")
        return names[(((a + 22.5) % 360) / 45).toInt()] + " ${a.toInt()}°"
    }

    fun hasPermission(ctx: Context) =
        ContextCompat.checkSelfPermission(ctx, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
}

/**
 * 相机页打开期间持续定位（GPS + 网络定位，取更准、更新的那个）。不依赖 Google 服务，国产手机也能用。
 * 没有权限或定位关闭时 latest 一直为 null，其余功能照常。
 */
class GeoTracker(private val ctx: Context, private val onFix: (GeoFix) -> Unit = {}) {
    @Volatile var latest: GeoFix? = null; private set
    private val lm = ctx.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    private val listener = LocationListener { accept(it) }

    private fun accept(l: Location) {
        val f = GeoFix(l.latitude, l.longitude, if (l.hasAccuracy()) l.accuracy.toDouble() else 50.0, l.time)
        val cur = latest
        // 新定位明显更旧、又不更准时不替换
        if (cur != null && f.time < cur.time - 10_000 && f.acc >= cur.acc) return
        if (cur != null && f.time - cur.time < 15_000 && f.acc > cur.acc * 2) return
        latest = f; onFix(f)
    }

    /** 当前定位；超过 2 分钟没更新视为过期。 */
    fun fresh(): GeoFix? = latest?.takeIf { System.currentTimeMillis() - it.time < 120_000 }

    @SuppressLint("MissingPermission")
    fun start() {
        if (!Geo.hasPermission(ctx)) return
        for (p in listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)) {
            runCatching {
                if (!lm.isProviderEnabled(p)) return@runCatching
                lm.getLastKnownLocation(p)?.let { if (System.currentTimeMillis() - it.time < 120_000) accept(it) }
                lm.requestLocationUpdates(p, 2000L, 0f, listener, Looper.getMainLooper())
            }
        }
    }

    fun stop() { runCatching { lm.removeUpdates(listener) } }
}

/**
 * 镜头朝向：旋转矢量传感器（陀螺仪 + 加速度计 + 磁力计融合）。
 * 取“后摄像头指向”的方向（设备 -Z 轴），竖着拿手机也正确；指南针精度低（附近有钢轨、钢结构）时 azimuth 为空。
 */
class PoseTracker(ctx: Context) : android.hardware.SensorEventListener {
    private val sm = ctx.getSystemService(Context.SENSOR_SERVICE) as android.hardware.SensorManager
    private val sensor = sm.getDefaultSensor(android.hardware.Sensor.TYPE_ROTATION_VECTOR)
    private val r = FloatArray(9)
    private var e = 0.0; private var n = 0.0; private var u = 0.0   // 低通滤波后的镜头方向（东、北、天）
    @Volatile var magReliable = true; private set
    @Volatile var latest: Pose? = null; private set
    val available get() = sensor != null

    fun start() { sensor?.let { sm.registerListener(this, it, android.hardware.SensorManager.SENSOR_DELAY_UI) } }
    fun stop() { sm.unregisterListener(this) }

    override fun onSensorChanged(ev: android.hardware.SensorEvent) {
        android.hardware.SensorManager.getRotationMatrixFromVector(r, ev.values)
        // world = R · device；设备 Z 轴在世界坐标中是 R 的第 3 列，后摄指向 -Z
        val k = 0.15
        e += k * (-r[2] - e); n += k * (-r[5] - n); u += k * (-r[8] - u)
        val horiz = Math.hypot(e, n)
        val pitch = Math.toDegrees(Math.atan2(u, horiz))
        // 镜头接近竖直朝上/下时朝向无意义
        val az = if (horiz < 0.2 || !magReliable) null else (Math.toDegrees(Math.atan2(e, n)) + 360) % 360
        latest = Pose(az, pitch)
    }

    override fun onAccuracyChanged(s: android.hardware.Sensor, accuracy: Int) {
        magReliable = accuracy >= android.hardware.SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM
    }
}

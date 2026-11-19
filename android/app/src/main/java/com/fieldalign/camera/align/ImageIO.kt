package com.fieldalign.camera.align

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.max
import kotlin.math.roundToInt

object ImageIO {
    /** 读取照片 → 按 EXIF 摆正 → 缩放到工作分辨率（长边 2000）。同一文件结果确定，保证参考坐标可复现。 */
    fun loadWorkBitmap(path: String, maxSide: Int = AlignEngine.WORK_MAX_SIDE): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
        val bmp = BitmapFactory.decodeFile(path, BitmapFactory.Options().apply {
            inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888
        }) ?: error("无法读取照片 $path")
        val rot = when (ExifInterface(path).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        val s = maxSide.toFloat() / max(bmp.width, bmp.height)
        val m = Matrix()
        if (s < 1f) m.postScale(s, s)
        if (rot != 0f) m.postRotate(rot)
        if (m.isIdentity) return bmp
        val out = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        if (out !== bmp) bmp.recycle()
        return out
    }

    fun loadWorkMat(path: String): Mat = bitmapToMat(loadWorkBitmap(path))

    fun bitmapToMat(b: Bitmap): Mat = Mat().also { Utils.bitmapToMat(b, it) }   // RGBA

    fun matToBitmap(m: Mat): Bitmap {
        val b = Bitmap.createBitmap(m.cols(), m.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(m, b)
        return b
    }

    fun saveJpeg(rgba: Mat, file: File, quality: Int = 90) {
        file.parentFile?.mkdirs()
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        Imgcodecs.imwrite(file.absolutePath, bgr, org.opencv.core.MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, quality))
    }

    fun loadThumb(path: String, side: Int = 360): Bitmap? = try {
        loadWorkBitmap(path, side)
    } catch (e: Exception) { null }

    fun roundPct(v: Double) = (v * 1000).roundToInt() / 10.0
}

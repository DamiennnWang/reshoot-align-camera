package com.fieldalign.camera.data

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import com.google.zxing.BarcodeFormat
import com.google.zxing.EncodeHintType
import com.google.zxing.qrcode.QRCodeWriter
import com.google.zxing.qrcode.decoder.ErrorCorrectionLevel
import java.io.File

/**
 * 批量编号靶标（和电脑端 make_target.py --sheet 同一版式）：
 * A4 每页 2 × 3 个标签；每个标签 = 40 mm 二维码（内容 = 编号，纠错 H）+ 编号文字 + 3 个 16 mm 圆点，虚线为裁切线。
 * 圆点 / 二维码 = 0.4 是对齐算法的预设比例，不要改。
 */
object TargetSheet {
    const val QR_MM = 40f
    private const val DOT_MM = 16f
    private const val CELL_W = 95f
    private const val CELL_H = 92f
    private const val COLS = 2
    private const val ROWS = 3
    const val PER_PAGE = COLS * ROWS
    private const val PT = 72f / 25.4f          // 1 mm = 2.835 pt（PDF 单位）

    private fun qrMatrix(code: String) = QRCodeWriter().encode(code, BarcodeFormat.QR_CODE, 0, 0,
        mapOf(EncodeHintType.ERROR_CORRECTION to ErrorCorrectionLevel.H, EncodeHintType.MARGIN to 0, EncodeHintType.CHARACTER_SET to "UTF-8"))

    /** 在 canvas 上画一个标签；坐标单位 mm，s = 每 mm 多少画布单位。 */
    fun drawLabel(c: Canvas, x0: Float, y0: Float, code: String, s: Float) {
        val dash = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE; color = 0xFFAAAAAA.toInt(); strokeWidth = 0.25f * s
            pathEffect = DashPathEffect(floatArrayOf(2f * s, 1.5f * s), 0f)
        }
        c.drawRect(x0 * s, y0 * s, (x0 + CELL_W) * s, (y0 + CELL_H) * s, dash)
        val black = Paint().apply { color = 0xFF000000.toInt(); style = Paint.Style.FILL; isAntiAlias = false }
        val bm = qrMatrix(code)
        val n = bm.width
        val qx = x0 + (CELL_W - QR_MM) / 2; val qy = y0 + 7f
        val m = QR_MM / n
        for (yy in 0 until n) {
            var xx = 0
            while (xx < n) {                      // 一行里连续的黑块合并成一个矩形，PDF 更小、没有缝
                if (!bm[xx, yy]) { xx++; continue }
                val st = xx
                while (xx < n && bm[xx, yy]) xx++
                // 向右下多画 0.06 mm 和下一行重叠：PDF 阅读器抗锯齿时相邻模块之间不会出现白色细缝
                val e = if (xx < n || yy < n - 1) 0.06f else 0f
                c.drawRect((qx + st * m) * s, (qy + yy * m) * s, (qx + xx * m + (if (xx < n) e else 0f)) * s,
                    (qy + (yy + 1) * m + (if (yy < n - 1) e else 0f)) * s, black)
            }
        }
        val txt = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF000000.toInt(); textSize = 6.2f * s; textAlign = Paint.Align.CENTER
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD) }
        c.drawText(code, (x0 + CELL_W / 2) * s, (qy + QR_MM + 8f) * s, txt)
        val tip = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF707070.toInt(); textSize = 2.6f * s; textAlign = Paint.Align.CENTER }
        c.drawText("复拍对齐靶标 · 二维码和圆点分开剪下", (x0 + CELL_W / 2) * s, (qy + QR_MM + 12.5f) * s, tip)
        val dot = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF000000.toInt() }
        val dash2 = Paint(dash).apply { color = 0xFFCCCCCC.toInt() }
        for (i in 0 until 3) {
            val cx = x0 + CELL_W / 2 + (i - 1) * 27f; val cy = y0 + 76f
            c.drawCircle(cx * s, cy * s, DOT_MM / 2 * s, dot)
            val r = DOT_MM / 2 + 4.5f
            c.drawRect((cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s, dash2)
        }
    }

    /** 生成 PDF，返回文件。 */
    fun pdf(codes: List<String>, out: File): File {
        val doc = PdfDocument()
        try {
            codes.chunked(PER_PAGE).forEachIndexed { pi, chunk ->
                val page = doc.startPage(PdfDocument.PageInfo.Builder((210 * PT).toInt(), (297 * PT).toInt(), pi + 1).create())
                val c = page.canvas
                val head = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF5A5A5A.toInt(); textSize = 3f * PT }
                c.drawText("复拍对齐靶标  ${chunk.first()} – ${chunk.last()}    按 100%（实际大小）打印，二维码边长应为 40 mm", 10 * PT, 8 * PT, head)
                chunk.forEachIndexed { i, code ->
                    val x0 = (210 - COLS * CELL_W) / 2 + (i % COLS) * CELL_W
                    val y0 = 11f + (i / COLS) * CELL_H
                    drawLabel(c, x0, y0, code, PT)
                }
                doc.finishPage(page)
            }
            out.parentFile?.mkdirs()
            out.outputStream().use { doc.writeTo(it) }
        } finally { doc.close() }
        return out
    }

    /** 屏幕预览：一个标签的位图。 */
    fun preview(code: String, widthPx: Int = 600): Bitmap {
        val s = widthPx / CELL_W
        val b = Bitmap.createBitmap(widthPx, (CELL_H * s).toInt(), Bitmap.Config.ARGB_8888)
        val c = Canvas(b); c.drawColor(0xFFFFFFFF.toInt())
        drawLabel(c, 0f, 0f, code, s)
        return b
    }
}

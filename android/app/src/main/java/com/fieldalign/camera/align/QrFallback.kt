package com.fieldalign.camera.align

import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import org.opencv.core.Mat

/** OpenCV 读不出二维码时的后备：ML Kit 对模糊、小尺寸、大斜角的码更稳。必须在后台线程调用。 */
object QrFallback {
    private val scanner by lazy {
        BarcodeScanning.getClient(BarcodeScannerOptions.Builder().setBarcodeFormats(Barcode.FORMAT_QR_CODE).build())
    }

    fun detect(work: Mat): Pair<String, List<P>>? = runCatching {
        val codes = Tasks.await(scanner.process(InputImage.fromBitmap(ImageIO.matToBitmap(work), 0)))
        val c = codes.firstOrNull { it.rawValue != null && it.cornerPoints?.size == 4 } ?: return@runCatching null
        c.rawValue!! to c.cornerPoints!!.map { P(it.x.toDouble(), it.y.toDouble()) }
    }.getOrNull()

    /** 参考照片有码而新照片读不出时：按特征摆正放大后再解码（见 AlignEngine.recoverQr）。 */
    fun recover(refWork: Mat, rm: AlignEngine.Markers, newWork: Mat): AlignEngine.Markers? =
        AlignEngine.recoverQr(refWork, rm, newWork) { detect(it) }?.let { AlignEngine.detectMarkers(newWork, it) }

    /** 先 OpenCV，读不出再 ML Kit。 */
    fun detectMarkers(work: Mat): AlignEngine.Markers {
        val m = AlignEngine.detectMarkers(work)
        if (m.qrText != null) return m
        return detect(work)?.let { AlignEngine.detectMarkers(work, it) } ?: m
    }
}

package com.fieldalign.camera.align

import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max

/**
 * 一次对比的完整病害分析：按测点类型选基元 → 汇总成“无变化 / 轻微 / 明显 / 建议重点关注”+ 一段文字 + 标注图 + JSON。
 * App 的处理流程和设备测试都走这里，保证同一套逻辑。
 */
object Analysis {
    val LEVELS = listOf("无变化", "轻微", "明显", "建议重点关注")

    class Result(
        val crack: Measure.Change?, val mmpp: Double?, val level: String, val text: String,
        val areas: List<Defects.Area>, val disp: Defects.Disp?, val marks: List<Defects.Mark>, val parts: List<Defects.Part>,
        val image: Mat, val items: List<String>, val timing: String,
    )

    /**
     * @param cmpMask 对比区：有效 ∩ 一致区（ROI）∩ 非靶标
     * @param fullMask 有效 ∩ 非靶标（位移类测点用：移动的一侧不在一致区里）
     */
    fun run(kinds: Set<Defects.Kind>, refWork: Mat, aligned: Mat, cmpMask: Mat, fullMask: Mat, c: AlignEngine.CompareResult,
            scale: Measure.Scale?, qrCenter: P?): Result {
        val plan = Defects.Plan(kinds)
        val t = StringBuilder(); var t0 = System.currentTimeMillis()
        fun lap(s: String) { val n = System.currentTimeMillis(); t.append("$s ${n - t0}ms "); t0 = n }
        val mmppAt: ((P) -> Double)? = scale?.let { s -> { p: P -> s.mmPerPx(p) } }
        val refG = AlignEngine.gray(refWork); val alG = AlignEngine.gray(aligned)
        // 位移先做：整块移动的区域搬回原位后，再做裂缝/面状/部件，避免把“整体移动”当成表面变化
        val disp = if (plan.disp) Defects.displacement(refG, alG, if (plan.dispFull) fullMask else cmpMask, mmppAt)
            ?.takeIf { (it.mm ?: (it.px / 4)) >= 0.5 && it.px >= 2.0 } else null
        lap("位移")
        val al2 = Defects.compensate(aligned, disp)
        val al2G = if (disp == null) alG else AlignEngine.gray(al2)
        // 裂缝（变化图里去掉反光区）
        val m = Mat(); Core.compare(c.changeMask, Scalar(64.0), m, Core.CMP_NE); Core.bitwise_and(m, cmpMask, m)
        val ch = if (plan.line) Measure.compare(refG.clone(), al2G.clone(), m) else null   // compare 会原地模糊输入
        lap("裂缝")
        val mmpp = scale?.let { s ->
            val ks = ch?.new?.cracks?.ifEmpty { ch.ref.cracks } ?: emptyList()
            if (ks.isEmpty()) s.mmPerPx(qrCenter ?: P(refWork.cols() / 2.0, refWork.rows() / 2.0))
            else s.mmPerPx(P(ks.sumOf { it.cx * it.lengthPx } / ks.sumOf { it.lengthPx }, ks.sumOf { it.cy * it.lengthPx } / ks.sumOf { it.lengthPx }))
        }
        val areas = if (plan.area) Defects.areas(refWork, al2, cmpMask, c.changeMask, mmppAt) else emptyList()
        lap("面状")
        val marks = if (plan.marks) Defects.marks(refWork, aligned, fullMask) else emptyList()
        val parts = if (plan.parts) Defects.parts(refG, al2G, c.changeMask, cmpMask) else emptyList()
        lap("部件")

        // ---------------------------------------------------------- 结论
        var sev = 0
        val sections = ArrayList<String>(); val items = ArrayList<String>()
        var crackLevel: String? = null
        if (ch != null) {
            val (lv, tx) = Measure.conclude(ch, mmpp, c.lightRegions.size, c.lightRatio)
            crackLevel = lv
            sev = max(sev, when (lv) { "无裂缝", "稳定" -> 0; "延长", "变宽" -> 2; else -> 3 })
            sections.add(if (plan.general || kinds.size > 1) "裂缝：$tx" else tx)
        }
        if (disp != null) {
            val k = disp.mm
            sev = max(sev, if (k != null) (if (k >= 3) 3 else if (k >= 1) 2 else 1) else (if (disp.px >= 12) 3 else if (disp.px >= 4) 2 else 1))
            val u = if (k != null) "mm" else "像素"
            val parts_ = listOfNotNull(disp.opening?.let { if (abs(it) >= 0.2 * (k ?: disp.px)) (if (it > 0) "张开 " else "闭合 ") + fmt(abs(it), u) else null },
                disp.shear?.let { if (it >= 0.2 * (k ?: disp.px)) "错动 " + fmt(it, u) else null })
            val s = "一侧相对另一侧整体移动 ${fmt(k ?: disp.px, u)}" + (if (parts_.isNotEmpty()) "（${parts_.joinToString("，")}）" else "") +
                "，方向 ${dirText(disp.dx, disp.dy)}"
            sections.add("位移：$s。若画面里有不同深度的构件，也可能是拍摄角度造成的视差，请结合现场确认")
            items.add("D  $s")
        } else if (plan.dispFull) sections.add("位移：未发现整体移动（检测下限约 4 像素${mmpp?.let { "，约 %.1f mm".format(4 * it) } ?: ""}）")
        if (areas.isNotEmpty()) {
            val by = areas.groupBy { it.cls }
            val s = by.entries.joinToString("；") { (cls, l) ->
                "$cls ${l.size} 处" + (l.mapNotNull { it.mm2 }.takeIf { it.size == l.size }?.let { "，共约 " + fmtArea(it.sum()) } ?: "")
            }
            areas.forEachIndexed { i, a ->
                sev = max(sev, areaSev(a))
                items.add("A${i + 1}  ${a.cls}" + (a.mm2?.let { "  约 " + fmtArea(it) } ?: "  ${a.px.toInt()} 像素"))
            }
            val note = if (areas.any { it.cls == "渗水/湿痕" || it.cls == "泛碱/白化" }) "。渗水、泛碱按颜色明暗判断，也可能是阴影、潮湿或污渍，需人工确认" else ""
            sections.add("面状：$s$note")
        } else if (plan.area && !plan.general) sections.add("面状：未发现剥落、锈蚀、渗水、泛碱等面积变化（≥1 cm²）")
        if (plan.marks) {
            if (marks.isEmpty()) sections.add("螺栓：参考照片里没有找到防松标记线（建议用红色或黄色油漆在螺母和钢板上画一条直线）")
            else {
                val cnt = marks.groupingBy { it.status }.eachCount()
                val loose = marks.filter { it.status == "疑似松动" }
                sev = max(sev, if (loose.isNotEmpty()) 3 else if (marks.any { it.status != "正常" }) 2 else 0)
                sections.add("螺栓：防松标记 ${marks.size} 处，" + listOf("正常", "疑似松动", "标记残缺", "标记缺失").filter { it in cnt }
                    .joinToString("，") { "$it ${cnt[it]}" } + (if (loose.isNotEmpty()) "（转角约 ${loose.joinToString("、") { "%.0f°".format(it.dang) }}）" else ""))
                marks.forEachIndexed { i, mk -> items.add("M${i + 1}  ${mk.status}" + if (mk.status == "疑似松动") "，转角约 %.0f°".format(mk.dang) else "") }
            }
        }
        if (parts.isNotEmpty()) {
            val cnt = parts.groupingBy { it.status }.eachCount()
            sev = max(sev, if ("疑似缺失" in cnt) 3 else if ("新增物体" in cnt) 2 else 1)
            sections.add("部件：变化 ${parts.size} 处，" + cnt.entries.joinToString("，") { "${it.key} ${it.value}" })
            parts.forEachIndexed { i, p -> items.add("P${i + 1}  ${p.status}") }
        } else if (plan.parts) sections.add("部件：未发现缺失或新增")
        // 只测裂缝的测点：沿用裂缝等级（无裂缝/稳定/延长/变宽/新增），和以前的结论一致
        val level = if (kinds == setOf(Defects.Kind.CRACK) && crackLevel != null) crackLevel else LEVELS[sev]
        val text = sections.joinToString("\n").ifEmpty { "未发现变化" }
        lap("结论")
        val img = draw(ch?.let { Measure.draw(al2, it, mmpp) } ?: al2.clone(), areas, disp, marks, parts)
        lap("标注")
        return Result(ch, mmpp, level, text, areas, disp, marks, parts, img, items, t.toString())
    }

    private fun fmt(v: Double, u: String) = if (u == "mm") "%.1f mm".format(v) else "%.0f 像素".format(v)
    private fun fmtArea(mm2: Double) = if (mm2 >= 100) "%.1f cm²".format(mm2 / 100) else "%.0f mm²".format(mm2)
    private fun dirText(dx: Double, dy: Double): String {
        val h = if (abs(dx) >= 0.4 * abs(dy)) (if (dx > 0) "向右" else "向左") else ""
        val v = if (abs(dy) >= 0.4 * abs(dx)) (if (dy > 0) "向下" else "向上") else ""
        return (h + v).ifEmpty { "—" } + "（以参考照片画面为准）"
    }

    private fun areaSev(a: Defects.Area): Int {
        val mm2 = a.mm2 ?: (a.px / 4.0)
        val heavy = a.cls == "剥落/掉块" || a.cls == "锈蚀"
        return when { mm2 >= (if (heavy) 3000 else 10000) -> 3; mm2 >= 1000 -> 2; else -> 1 }
    }

    // ---------------------------------------------------------- 标注图（RGBA）
    private val COL = mapOf("锈蚀" to Scalar(255.0, 120.0, 0.0, 255.0), "剥落/掉块" to Scalar(255.0, 40.0, 40.0, 255.0),
        "表面缺损" to Scalar(255.0, 40.0, 40.0, 255.0), "渗水/湿痕" to Scalar(40.0, 140.0, 255.0, 255.0),
        "泛碱/白化" to Scalar(190.0, 120.0, 255.0, 255.0), "表面变化" to Scalar(255.0, 200.0, 0.0, 255.0))

    private var fs = 1.3; private var lw = 5      // 标注字号、线宽：按图像宽度缩放，手机上看得清

    private fun tag(img: Mat, s: String, at: Point, col: Scalar) {
        val p = Point(at.x, max(at.y, 30 * fs))
        Imgproc.putText(img, s, p, Imgproc.FONT_HERSHEY_SIMPLEX, fs, Scalar(0.0, 0.0, 0.0, 255.0), (fs * 6).toInt(), Imgproc.LINE_AA)
        Imgproc.putText(img, s, p, Imgproc.FONT_HERSHEY_SIMPLEX, fs, col, (fs * 2.2).toInt(), Imgproc.LINE_AA)
    }

    fun draw(base: Mat, areas: List<Defects.Area>, disp: Defects.Disp?, marks: List<Defects.Mark>, parts: List<Defects.Part>): Mat {
        val img = base
        fs = max(1.3, img.cols() / 700.0); lw = max(5, img.cols() / 300)
        areas.forEachIndexed { i, a ->
            val col = COL[a.cls] ?: COL.getValue("表面变化")
            if (a.contour.size >= 3) Imgproc.polylines(img, listOf(MatOfPoint(*a.contour.map { Point(it.x, it.y) }.toTypedArray())), true, col, lw, Imgproc.LINE_AA)
            tag(img, "A${i + 1}", Point(a.x.toDouble(), a.y - 10.0), col)
        }
        disp?.let { d ->
            val cyan = Scalar(0.0, 230.0, 255.0, 255.0)
            Imgproc.rectangle(img, Point(d.box[0].toDouble(), d.box[1].toDouble()), Point((d.box[0] + d.box[2]).toDouble(), (d.box[1] + d.box[3]).toDouble()), cyan, lw)
            val cx = d.box[0] + d.box[2] / 2.0; val cy = d.box[1] + d.box[3] / 2.0
            val k = max(60.0, 8 * d.px) / max(d.px, 1e-6)
            Imgproc.arrowedLine(img, Point(cx, cy), Point(cx + d.dx * k, cy + d.dy * k), Scalar(255.0, 0.0, 255.0, 255.0), lw + 2, Imgproc.LINE_AA, 0, 0.3)
            tag(img, "D", Point(d.box[0] + 8.0, d.box[1] + 40 * fs), cyan)
        }
        marks.forEachIndexed { i, mk ->
            val col = when (mk.status) { "正常" -> Scalar(40.0, 220.0, 90.0, 255.0); "疑似松动" -> Scalar(255.0, 30.0, 30.0, 255.0); else -> Scalar(255.0, 150.0, 0.0, 255.0) }
            Imgproc.circle(img, Point(mk.cx, mk.cy), (mk.length / 2 + 16).toInt(), col, lw, Imgproc.LINE_AA)
            tag(img, "M${i + 1}", Point(mk.cx + mk.length / 2 + 18, mk.cy), col)
        }
        parts.forEachIndexed { i, p ->
            val col = when (p.status) { "疑似缺失" -> Scalar(255.0, 30.0, 30.0, 255.0); "新增物体" -> Scalar(255.0, 150.0, 0.0, 255.0); else -> Scalar(255.0, 220.0, 0.0, 255.0) }
            Imgproc.rectangle(img, Point(p.x - 8.0, p.y - 8.0), Point(p.x + p.w + 8.0, p.y + p.h + 8.0), col, lw)
            tag(img, "P${i + 1}", Point(p.x - 8.0, p.y - 16.0), col)
        }
        return img
    }

    // ---------------------------------------------------------- JSON（存库 + 导出档案）
    fun toJson(r: Result, kinds: Set<Defects.Kind>, imagePath: String?): String = JSONObject().apply {
        put("kinds", Defects.encode(kinds) ?: "general")
        put("level", r.level); put("mm_per_px", r.mmpp ?: JSONObject.NULL)
        put("image", imagePath ?: JSONObject.NULL)
        put("items", JSONArray(r.items))
        put("areas", JSONArray(r.areas.map { a -> JSONObject().apply {
            put("class", a.cls); put("box", JSONArray(listOf(a.x, a.y, a.w, a.h))); put("area_px", a.px)
            put("area_mm2", a.mm2 ?: JSONObject.NULL); put("dL", a.dL)
            put("contour", JSONArray(a.contour.map { JSONArray(listOf(it.x.toInt(), it.y.toInt())) })) } }))
        put("displacement", r.disp?.let { d -> JSONObject().apply {
            put("dx_px", d.dx); put("dy_px", d.dy); put("mm", d.mm ?: JSONObject.NULL); put("opening", d.opening ?: JSONObject.NULL)
            put("shear", d.shear ?: JSONObject.NULL); put("blocks", d.blocks); put("box", JSONArray(d.box.toList())) } } ?: JSONObject.NULL)
        put("marks", JSONArray(r.marks.map { m -> JSONObject().apply {
            put("cx", m.cx); put("cy", m.cy); put("angle", m.ang); put("length_px", m.length); put("status", m.status)
            put("rotation_deg", m.dang); put("coverage", m.coverage) } }))
        put("parts", JSONArray(r.parts.map { p -> JSONObject().apply {
            put("box", JSONArray(listOf(p.x, p.y, p.w, p.h))); put("status", p.status); put("ncc", p.ncc); put("edge_ratio", p.eRatio); put("d_mean", p.dMean) } }))
    }.toString()

    fun itemsOf(json: String?): List<String> = runCatching {
        val a = JSONObject(json ?: return emptyList()).getJSONArray("items"); (0 until a.length()).map { a.getString(it) }
    }.getOrDefault(emptyList())

    fun imageOf(json: String?): String? = runCatching { JSONObject(json ?: return null).optString("image").takeIf { it.isNotEmpty() && it != "null" } }.getOrNull()
}

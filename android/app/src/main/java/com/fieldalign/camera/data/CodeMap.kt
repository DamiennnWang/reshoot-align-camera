package com.fieldalign.camera.data

import org.json.JSONObject
import java.io.File

/**
 * 编号对应表：二维码编号（如 T0001）→ 项目编号 / 名称 / 备注。
 *
 * 靶标先批量打印、随手贴；之后在表里（或导入 CSV）填上每个编号对应的项目编号，
 * 已建档的测点自动改名，以后第一次拍到这个编号时也直接用对应的名称建档。
 * 存成 App 私有目录里的一个 JSON 文件（条目不多，不必进数据库）。
 */
class CodeMap(dir: File) {
    data class Entry(val code: String, val project: String = "", val name: String = "", val note: String = "", val createdAt: Long = 0) {
        /** 测点显示名：项目编号 + 名称；都没填时为 null（用编号本身）。 */
        val display: String? get() = listOf(project, name).filter { it.isNotBlank() }.joinToString(" ").ifBlank { null }
    }

    private val file = File(dir, "code_map.json")
    private val map = LinkedHashMap<String, Entry>()

    init { load() }

    @Synchronized private fun load() {
        map.clear()
        if (!file.exists()) return
        runCatching {
            val o = JSONObject(file.readText())
            o.keys().forEach { k ->
                val e = o.getJSONObject(k)
                map[k] = Entry(k, e.optString("project"), e.optString("name"), e.optString("note"), e.optLong("created"))
            }
        }
    }

    @Synchronized private fun save() {
        val o = JSONObject()
        map.values.forEach { e -> o.put(e.code, JSONObject().put("project", e.project).put("name", e.name).put("note", e.note).put("created", e.createdAt)) }
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(o.toString()); tmp.renameTo(file)
    }

    @Synchronized fun all(): List<Entry> = map.values.sortedBy { it.code }
    @Synchronized fun get(code: String?): Entry? = code?.let { map[it] ?: map[base(it)] }
    fun displayFor(code: String?): String? = get(code)?.display

    @Synchronized fun put(e: Entry) { map[e.code] = e.copy(createdAt = map[e.code]?.createdAt ?: e.createdAt.takeIf { it > 0 } ?: System.currentTimeMillis()); save() }
    @Synchronized fun remove(code: String) { map.remove(code); save() }

    /** 生成靶标时登记新编号（已有的不动）。 */
    @Synchronized fun register(codes: List<String>) {
        val now = System.currentTimeMillis()
        codes.forEach { if (it !in map) map[it] = Entry(it, createdAt = now) }
        save()
    }

    /**
     * 导入 CSV：第一列编号，后面依次 项目编号、名称、备注（表头可有可无，逗号或制表符分隔）。
     * 只覆盖非空的格子。返回导入条数。
     */
    @Synchronized fun importCsv(text: String): Int {
        var n = 0
        text.removePrefix("﻿").lineSequence().forEach { line ->
            val cells = splitCsv(line)
            val code = cells.getOrNull(0)?.trim().orEmpty()
            if (code.isEmpty() || code == "编号" || code.equals("code", true)) return@forEach
            val old = map[code] ?: Entry(code, createdAt = System.currentTimeMillis())
            fun c(i: Int, cur: String) = cells.getOrNull(i)?.trim()?.takeIf { it.isNotEmpty() } ?: cur
            map[code] = old.copy(project = c(1, old.project), name = c(2, old.name), note = c(3, old.note))
            n++
        }
        save()
        return n
    }

    fun toCsv(entries: List<Entry> = all(), zoneOf: (String) -> String? = { null }): String =
        "﻿编号,项目编号,名称,备注,测点\n" + entries.joinToString("") { e ->
            listOf(e.code, e.project, e.name, e.note, zoneOf(e.code).orEmpty()).joinToString(",") { q(it) } + "\n"
        }

    companion object {
        /** 二维码里可能带尺寸、类型（B01-Z023;40mm;裂缝），对应表按分号前的编号查。 */
        fun base(qr: String) = qr.substringBefore(';').trim()

        private fun q(s: String) = if (s.any { it == ',' || it == '"' || it == '\n' }) "\"" + s.replace("\"", "\"\"") + "\"" else s

        fun splitCsv(line: String): List<String> {
            val sep = if ('\t' in line && ',' !in line) '\t' else ','
            val out = ArrayList<String>(); val sb = StringBuilder(); var inQ = false; var i = 0
            while (i < line.length) {
                val ch = line[i]
                when {
                    inQ && ch == '"' && i + 1 < line.length && line[i + 1] == '"' -> { sb.append('"'); i++ }
                    ch == '"' -> inQ = !inQ
                    ch == sep && !inQ -> { out.add(sb.toString()); sb.clear() }
                    ch != '\r' -> sb.append(ch)
                }
                i++
            }
            out.add(sb.toString())
            return out
        }

        fun codes(prefix: String, start: Int, count: Int, digits: Int) =
            (start until start + count).map { prefix + it.toString().padStart(digits, '0') }
    }
}

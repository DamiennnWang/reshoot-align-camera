package com.fieldalign.camera

import com.fieldalign.camera.align.Defects
import com.fieldalign.camera.align.Defects.Kind
import org.junit.Assert.assertEquals
import org.junit.Test

/** 测点类型：二维码里写类型、存库编码/解码。 */
class DefectKindTest {
    @Test fun fromQr() {
        assertEquals(setOf(Kind.BOLT), Defects.fromQr("B01-Z023;40mm;螺栓"))
        assertEquals(setOf(Kind.JOINT, Kind.CRACK), Defects.fromQr("Q12-伸缩缝-03;伸缩缝,裂缝"))
        assertEquals(setOf(Kind.SEEP, Kind.SPALL), Defects.fromQr("T7|渗水|掉块"))
        assertEquals(setOf(Kind.RUST, Kind.BOLT), Defects.fromQr("S3;rust;bolt"))
        assertEquals(emptySet<Kind>(), Defects.fromQr("B01-Z023"))            // 编号本身不算类型
        assertEquals(emptySet<Kind>(), Defects.fromQr("B01-Z023;40mm"))
        assertEquals(emptySet<Kind>(), Defects.fromQr(null))
    }

    @Test fun encodeParse() {
        val k = setOf(Kind.BOLT, Kind.CRACK, Kind.SEEP)
        assertEquals("crack,seep,bolt", Defects.encode(k))
        assertEquals(k, Defects.parse(Defects.encode(k)))
        assertEquals(null, Defects.encode(emptySet()))
        assertEquals(emptySet<Kind>(), Defects.parse(null))
        assertEquals("通用", Defects.labels(emptySet()))
    }

    @Test fun plan() {
        val g = Defects.Plan(emptySet())
        assert(g.line && g.area && g.disp && !g.dispFull && !g.marks && !g.parts)
        val b = Defects.Plan(setOf(Kind.BOLT))
        assert(!b.line && !b.area && !b.disp && b.marks && b.parts)
        val j = Defects.Plan(setOf(Kind.JOINT))
        assert(j.disp && j.dispFull && !j.area && !j.line)
    }
}

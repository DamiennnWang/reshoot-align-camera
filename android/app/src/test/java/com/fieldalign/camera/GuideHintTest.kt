package com.fieldalign.camera

import com.fieldalign.camera.align.FrameGuide
import com.fieldalign.camera.align.Homography
import com.fieldalign.camera.align.P
import com.fieldalign.camera.align.convexHullArea
import com.fieldalign.camera.align.pointInQuad
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** 纯 Kotlin 部分（不依赖 OpenCV 原生库）的单元测试。 */
class GuideHintTest {
    private val frame = listOf(P(0.0, 0.0), P(1.0, 0.0), P(1.0, 1.0), P(0.0, 1.0))
    private fun shift(dx: Double, dy: Double) = frame.map { P(it.x + dx, it.y + dy) }
    private fun scale(s: Double) = frame.map { P(0.5 + (it.x - 0.5) * s, 0.5 + (it.y - 0.5) * s) }

    @Test fun alignedWhenFramesMatch() = assertEquals("已对准", FrameGuide.hintFor(frame).first)
    @Test fun refToTheRightMeansMoveRight() = assertEquals("镜头向右移", FrameGuide.hintFor(shift(0.12, 0.0)).first)
    @Test fun refAboveMeansMoveUp() = assertEquals("镜头向上移", FrameGuide.hintFor(shift(0.0, -0.1)).first)
    @Test fun refBiggerMeansStepBack() = assertEquals("后退一点", FrameGuide.hintFor(scale(1.3)).first)
    @Test fun refSmallerMeansGetCloser() = assertEquals("靠近一点", FrameGuide.hintFor(scale(0.75)).first)

    @Test fun rotatedClockwise() {
        val a = Math.toRadians(8.0)
        val q = frame.map { P(0.5 + (it.x - 0.5) * Math.cos(a) - (it.y - 0.5) * Math.sin(a),
                              0.5 + (it.x - 0.5) * Math.sin(a) + (it.y - 0.5) * Math.cos(a)) }
        assertEquals("手机顺时针转一点", FrameGuide.hintFor(q).first)
    }

    @Test fun homographyInverseRoundTrip() {
        val h = Homography(doubleArrayOf(1.1, 0.05, 12.0, -0.03, 0.95, -7.0, 1e-4, -2e-4, 1.0))
        val p = P(321.0, 123.0)
        val back = h.inverse().apply(h.apply(p))
        assertTrue(back.dist(p) < 1e-9)
        assertTrue((h * h.inverse()).apply(p).dist(p) < 1e-9)
    }

    @Test fun hullAreaOfUnitSquare() = assertEquals(1.0, convexHullArea(frame + P(0.5, 0.5)), 1e-12)

    @Test fun quadContainment() {
        assertTrue(pointInQuad(frame, P(0.5, 0.5)))
        assertFalse(pointInQuad(frame, P(1.2, 0.5)))
        assertTrue(pointInQuad(frame, P(1.2, 0.5), grow = 1.5))
    }
}

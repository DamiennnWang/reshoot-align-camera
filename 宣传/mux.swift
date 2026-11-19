// 把无声画面和音轨合成一个 mp4：swift mux.swift 画面.mp4 音轨.m4a 输出.mp4
import AVFoundation
import Foundation

let a = CommandLine.arguments
guard a.count >= 4 else { print("用法: swift mux.swift video audio out"); exit(1) }
let v = AVURLAsset(url: URL(fileURLWithPath: a[1])), au = AVURLAsset(url: URL(fileURLWithPath: a[2]))
let out = URL(fileURLWithPath: a[3])
try? FileManager.default.removeItem(at: out)
let comp = AVMutableComposition()
let sem = DispatchSemaphore(value: 0)
Task {
    do {
        let vt = try await v.loadTracks(withMediaType: .video)[0]
        let at = try await au.loadTracks(withMediaType: .audio)[0]
        let vd = try await v.load(.duration), ad = try await au.load(.duration)
        let d = CMTimeMinimum(vd, ad)
        let cv = comp.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)!
        let ca = comp.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)!
        try cv.insertTimeRange(CMTimeRange(start: .zero, duration: d), of: vt, at: .zero)
        try ca.insertTimeRange(CMTimeRange(start: .zero, duration: d), of: at, at: .zero)
        let ex = AVAssetExportSession(asset: comp, presetName: a.count > 4 ? a[4] : AVAssetExportPresetPassthrough)!
        ex.outputURL = out; ex.outputFileType = .mp4; ex.shouldOptimizeForNetworkUse = true
        await ex.export()
        if ex.status == .completed { print("OK", a[3], CMTimeGetSeconds(d)) } else { print("FAILED", ex.error ?? "?"); exit(2) }
    } catch { print("ERROR", error); exit(3) }
    sem.signal()
}
sem.wait()

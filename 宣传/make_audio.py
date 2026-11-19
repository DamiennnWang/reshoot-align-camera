"""
宣传片音轨：程序合成的原创配乐（120 BPM，A 小调）+ 豆包大模型语音旁白，混成 素材/音轨.wav。

  ../.venv/bin/python make_audio.py

配乐完全由本脚本用正弦波/噪声合成，没有使用任何第三方音乐素材，不存在版权问题。
旁白用豆包语音合成 2.0（音色「大壹」，凭据在钥匙串，由 robot 项目的 doubao_tts.py 读取）；\n换音色：DB_VOICE=zh_male_m191_uranus_bigtts ../.venv/bin/python make_audio.py
"""
import json
import os
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import film_script as fs  # noqa: E402

SR = 44100
N = int(fs.TOTAL * SR)
rng = np.random.default_rng(2026)
OUT = os.path.join(HERE, "素材"); os.makedirs(OUT, exist_ok=True)
TMP = os.path.join(OUT, "旁白"); os.makedirs(TMP, exist_ok=True)


# ------------------------------------------------------------------ 基础
def tt(dur):
    return np.arange(int(dur * SR)) / SR


def fftfilt(x, lo=0.0, hi=1e9):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    g = 1 / (1 + (lo / np.maximum(f, 1e-6)) ** 6) if lo > 0 else np.ones_like(f)
    if hi < SR / 2:
        g = g / (1 + (f / hi) ** 6)
    return np.fft.irfft(X * g, len(x))


class Bus:
    def __init__(self):
        self.b = np.zeros((N, 2), np.float32)

    def add(self, sig, t0, gain=1.0, pan=0.0):
        i = int(t0 * SR)
        if i >= N or i + len(sig) <= 0:
            return
        s = sig[max(-i, 0):N - i] if i + len(sig) > N or i < 0 else sig
        i = max(i, 0)
        l = gain * np.sqrt(0.5 * (1 - pan)); r = gain * np.sqrt(0.5 * (1 + pan))
        self.b[i:i + len(s), 0] += s * l * 1.414
        self.b[i:i + len(s), 1] += s * r * 1.414


def note(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


# ------------------------------------------------------------------ 音色
def kick(power=1.0):
    t = tt(0.4)
    f = 44 + 120 * np.exp(-t * 30)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 8.5)
    s[:120] += rng.normal(0, 0.25, 120) * np.linspace(1, 0, 120)
    return np.tanh(s * 1.6) * power


def hat(open_=False):
    t = tt(0.22 if open_ else 0.07)
    return fftfilt(rng.normal(0, 1, len(t)), lo=7000) * np.exp(-t * (16 if open_ else 70)) * 0.5


def clap():
    t = tt(0.3)
    env = sum(np.exp(-np.maximum(t - d, 0) * 34) * (t >= d) for d in (0.0, 0.012, 0.026))
    return fftfilt(rng.normal(0, 1, len(t)), lo=900, hi=6500) * env * 0.5


def snare():
    t = tt(0.18)
    return (fftfilt(rng.normal(0, 1, len(t)), lo=1500) * np.exp(-t * 26) * 0.5 + np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30) * 0.4)


def bass(f, dur):
    t = tt(dur)
    s = sum(np.sin(2 * np.pi * k * f * t) / k * np.exp(-t * (1.5 + 2.2 * k)) for k in range(1, 9))
    s += 0.8 * np.sin(2 * np.pi * f * t)
    env = np.minimum(t / 0.005, 1) * np.minimum((dur - t) / 0.03, 1)
    return np.tanh(s * 1.2) * env * 0.5


def pluck(f, dur=0.35, bright=1.0):
    t = tt(dur)
    s = sum(np.sin(2 * np.pi * k * f * t + k) / k ** 1.1 * np.exp(-t * (5 + 3.0 * k / bright)) for k in range(1, 8))
    return s * np.minimum(t / 0.003, 1) * 0.35


def pad(freqs, dur, attack=1.0, release=1.2, bright=5):
    t = tt(dur)
    s = np.zeros_like(t)
    for f in freqs:
        for det in (-0.004, 0.0, 0.004):
            ff = f * (1 + det)
            s += sum(np.sin(2 * np.pi * k * ff * t + rng.uniform(0, 6.28)) / k ** 1.4 for k in range(1, bright + 1))
    env = np.minimum(t / attack, 1) * np.minimum((dur - t) / release, 1)
    trem = 1 + 0.12 * np.sin(2 * np.pi * 0.35 * t)
    return s * env * trem * 0.06


def thump(f=58, decay=14, dur=0.45):
    t = tt(dur)
    ff = f + 40 * np.exp(-t * 40)
    return np.sin(2 * np.pi * np.cumsum(ff) / SR) * np.exp(-t * decay)


def riser(dur, f0=180, f1=2600):
    t = tt(dur); p = t / dur
    n = fftfilt(rng.normal(0, 1, len(t)), lo=1200) * p ** 2 * 0.5
    f = f0 * (f1 / f0) ** p
    sw = np.sin(2 * np.pi * np.cumsum(f) / SR) * p ** 2 * 0.25
    return n + sw


def impact(big=1.0):
    t = tt(3.0)
    boom = np.sin(2 * np.pi * np.cumsum(34 + 50 * np.exp(-t * 12)) / SR) * np.exp(-t * 1.9)
    crash = fftfilt(rng.normal(0, 1, len(t)), lo=2500) * np.exp(-t * 2.6) * 0.35
    body = fftfilt(rng.normal(0, 1, len(t)), hi=500) * np.exp(-t * 7) * 0.8
    return np.tanh((boom * 1.3 + crash + body) * 1.2) * big


def rev_cymbal(dur):
    t = tt(dur)
    return fftfilt(rng.normal(0, 1, len(t)), lo=3000) * (t / dur) ** 3 * 0.5


def drip():
    f = rng.uniform(1400, 2600); t = tt(0.25)
    return np.sin(2 * np.pi * (f + 600 * np.exp(-t * 60)) * t) * np.exp(-t * 28) * 0.25


def glitch(dur):
    t = tt(dur); f = rng.choice([110, 220, 330, 440, 880])
    s = np.sign(np.sin(2 * np.pi * f * t)) * 0.25
    q = max(int(SR / rng.choice([2000, 4000, 8000])), 1)
    return np.repeat(s[::q], q)[:len(t)] * np.minimum((dur - t) / 0.005, 1)


# ------------------------------------------------------------------ 编曲
def compose():
    dry, wet = Bus(), Bus()       # wet 送混响
    beat = 0.5
    CH = [(45, [57, 60, 64]), (41, [53, 57, 60]), (48, [55, 60, 64]), (43, [55, 59, 62])]   # Am F C G：低音根音, 和弦音

    # --- 0–6 冷开场：低频长音 + 心跳 + 水滴
    dry.add(pad([note(33), note(33) * 1.004], 6.8, attack=2.0, release=1.5, bright=3) * 2.4, 0.0)
    wet.add(pad([note(69), note(70)], 6.0, attack=2.5, release=1.5, bright=2) * 0.6, 0.0)
    for k in range(6):
        g = 0.55 + 0.07 * k
        dry.add(thump(), 0.5 + k, g); dry.add(thump(52), 0.78 + k, g * 0.7)
    for t0 in (1.3, 2.9, 4.2, 5.1):
        wet.add(drip(), t0, 1.0, rng.uniform(-0.7, 0.7))

    # --- 6–16 压迫感：滴答 + 低音脉冲 + 不协和长音，13 秒起故障音，随后上升音效
    wet.add(pad([note(45), note(46), note(52)], 10.0, attack=4.0, release=0.3, bright=4) * 1.3, 6.0)
    for i in range(40):
        t0 = 6 + i * 0.25
        dry.add(hat(), t0, 0.16 + 0.006 * i, 0.3 if i % 2 else -0.3)
    for i in range(20):
        t0 = 6 + i * beat
        dry.add(bass(note(33), 0.42), t0, 0.55 + 0.02 * i)
        if t0 >= 9.5:
            dry.add(thump(70, 12, 0.3), t0, 0.5)
    for i in range(24):
        t0 = 13 + i * 0.125
        if rng.random() < 0.7:
            dry.add(glitch(0.09), t0, 0.5, rng.uniform(-0.8, 0.8))
    dry.add(riser(1.9), 13.85, 0.9); wet.add(rev_cymbal(1.0), 14.75, 0.9)
    # 15.75–16 留白

    # --- 16–20 口号：重击 + 三个字各一记
    dry.add(impact(1.0), 16.0, 1.0); wet.add(impact(0.5), 16.0, 0.6)
    for j, t0 in enumerate((16.5, 17.0, 17.5)):
        dry.add(kick(1.1), t0, 1.0)
        for m in (45, 57, 64, 69 + 3 * j):
            wet.add(pluck(note(m), 1.2, 1.6), t0, 0.9, (m % 5 - 2) * 0.2)
    wet.add(pad([note(57), note(64), note(72)], 2.6, attack=1.6, release=0.3) * 1.4, 17.6)
    dry.add(riser(1.5, 300, 3200), 18.5, 0.6)

    # --- 20–57 主体：四拍底鼓、反拍镲、掌声、低音、琶音；36 起加主旋律
    lead = {0: [76, None, 81, None, 84, 83, 81, None], 1: [77, None, 81, None, 84, None, 81, 79],
            2: [79, None, 84, None, 88, 86, 84, None], 3: [74, None, 79, None, 83, 81, 79, None]}
    for bi in range(int((57 - 20) / beat)):
        t0 = 20 + bi * beat
        sec = "build" if 34 <= t0 < 36 else ("soft" if 46 <= t0 < 48 else "main")
        root, tones = CH[int((t0 - 20) // 4) % 4]
        full = t0 >= 36
        if sec != "build":
            dry.add(kick(), t0, 0.95)
            dry.add(hat(), t0 + 0.25, 0.42, 0.25)
            if full:
                dry.add(hat(), t0 + 0.125, 0.2, -0.3); dry.add(hat(True), t0 + 0.375, 0.18, 0.4)
            if bi % 2 == 1 and sec == "main":
                wet.add(clap(), t0, 0.75)
        # 低音：八分音符，反拍高八度
        dry.add(bass(note(root - 12), 0.22), t0, 0.75)
        dry.add(bass(note(root), 0.2), t0 + 0.25, 0.55)
        # 琶音：十六分音符
        seq = tones + [tones[0] + 12, tones[1] + 12, tones[2] + 12, tones[1] + 12, tones[0] + 12]
        for k in range(4):
            m = seq[(bi * 4 + k) % len(seq)] + (12 if full else 0)
            wet.add(pluck(note(m), 0.3, 1.2), t0 + k * 0.125, 0.34 if not full else 0.4, np.sin(bi + k) * 0.6)
        if full and t0 < 46 or 48 <= t0 < 57:
            m = lead[int((t0 - 20) // 4) % 4][bi % 8]
            if m:
                wet.add(pluck(note(m), 0.9, 2.2), t0, 0.62, 0.1)
                wet.add(pluck(note(m), 0.6, 2.0), t0 + 0.375, 0.2, -0.5)      # 回声
    for ci in range(int((57 - 20) / 4) + 1):
        t0 = 20 + ci * 4
        root, tones = CH[ci % 4]
        wet.add(pad([note(m) for m in tones] + [note(root)], min(4.2, 57.2 - t0), attack=0.4, release=0.5) * 1.0, t0)
    # 34–36 推进：军鼓滚奏越来越密 + 上升音效
    t0 = 34.0
    for step, cnt in ((0.25, 4), (0.125, 4), (0.0625, 8)):
        for _ in range(cnt):
            dry.add(snare(), t0, 0.5 + 0.25 * (t0 - 34)); t0 += step
    dry.add(riser(2.0), 34.0, 0.8); wet.add(rev_cymbal(1.0), 35.0, 0.9)
    dry.add(impact(0.9), 36.0, 0.9); wet.add(impact(0.4), 36.0, 0.6)
    # 41–44.5 红色裂缝生长：上升音效 → 44.5 读数定格重击
    dry.add(riser(3.4, 200, 3600), 41.0, 0.55)
    dry.add(impact(1.0), 44.5, 1.0); wet.add(impact(0.5), 44.5, 0.7)
    # 52–57 卖点卡片：每张一记和弦
    for j in range(4):
        t0 = 52 + j * 1.25
        for m in (57, 64, 69, 72 + 2 * j):
            wet.add(pluck(note(m), 1.0, 2.0), t0, 0.8, (j - 1.5) * 0.3)
        dry.add(thump(80, 10, 0.3), t0, 0.6)
    dry.add(riser(1.0, 400, 3000), 56.0, 0.5)

    # --- 57–63 片尾：重击后长和弦渐弱，两声心跳
    dry.add(impact(1.0), 57.0, 1.0); wet.add(impact(0.5), 57.0, 0.7)
    wet.add(pad([note(45), note(57), note(64), note(71), note(72)], 6.0, attack=0.3, release=3.5) * 1.8, 57.0)
    for m, d in ((81, 0.0), (84, 0.5), (88, 1.0), (93, 1.5)):
        wet.add(pluck(note(m), 2.0, 2.5), 57.5 + d, 0.5, d - 0.7)
    for t0 in (60.0, 61.0):
        dry.add(thump(), t0, 0.5); dry.add(thump(52), t0 + 0.28, 0.35)
    return dry.b, wet.b


def reverb(x):
    n = int(2.2 * SR); t = np.arange(n) / SR
    out = np.zeros_like(x)
    L = len(x) + n
    for c in range(2):
        ir = fftfilt(rng.normal(0, 1, n), lo=250, hi=7000) * np.exp(-t * 3.2)
        ir[:int(0.012 * SR)] = 0
        y = np.fft.irfft(np.fft.rfft(x[:, c], L) * np.fft.rfft(ir, L), L)[:len(x)]
        out[:, c] = y
    return out / (np.abs(out).max() + 1e-9) * np.abs(x).max()


# ------------------------------------------------------------------ 旁白
DOUBAO = os.path.expanduser("~/Desktop/robot/教材制作/工具/录屏/lib/doubao_tts.py")   # 凭据由该脚本从钥匙串读取
VOICE = os.environ.get("DB_VOICE", "zh_male_dayi_uranus_bigtts")                       # 豆包 2.0「大壹」
CONTEXT = "高端科技宣传片旁白。低沉、克制、有磁性，语速沉稳，句中自然停顿，有电影感，不要播音腔，不要夸张。"
CACHE = os.path.join(OUT, "旁白缓存"); os.makedirs(CACHE, exist_ok=True)


def doubao(text_, rate):
    import hashlib
    h = hashlib.sha1(f"{VOICE}|{rate}|{CONTEXT}|{text_}".encode()).hexdigest()[:16]
    mp3, wav_ = os.path.join(CACHE, h + ".mp3"), os.path.join(CACHE, h + ".wav")
    if not os.path.exists(wav_):
        if not os.path.exists(mp3):
            subprocess.run([sys.executable, "-I", DOUBAO, "--voice", VOICE, "--text", text_, "--out", mp3,
                            "--rate", str(rate), "--context", CONTEXT], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@44100", "-c", "1", mp3, wav_], check=True)
    with wave.open(wav_) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def speak(i, text_, max_dur):
    for rate in (-18, -10, 0, 10, 20, 30):
        a = doubao(text_, rate)
        nz = np.nonzero(np.abs(a) > 0.01)[0]
        a = a[nz[0]:nz[-1] + 1] if len(nz) else a
        if len(a) / SR <= max_dur:
            break
    return a, rate


def main():
    print("合成配乐…")
    dry, wet = compose()
    music = dry + 0.55 * wet + 0.5 * reverb(wet)
    music = music / (np.percentile(np.abs(music), 99.9) + 1e-9) * 0.85
    # 总音量曲线：开场压低 → 逐渐推高 → 口号前留白 → 全开（由静到爆）
    tx = np.arange(N) / SR
    curve = np.interp(tx, [0, 6, 13, 15.6, 15.8, 15.97, 16.0, 20, 34, 36, 57, 63], [0.42, 0.5, 0.7, 0.95, 0.06, 0.04, 1.0, 0.82, 0.9, 1.0, 1.0, 0.9])
    music = music * curve[:, None].astype(np.float32)
    print("生成旁白…")
    voice = np.zeros(N, np.float32); times = []
    for i, (t0, sub, say_text, mx) in enumerate(fs.NARRATION):
        a, rate = speak(i, say_text or sub, mx)
        a = fftfilt(a, lo=90) ; a = a / (np.abs(a).max() + 1e-9) * 0.9
        j = int(t0 * SR); a = a[:N - j]
        voice[j:j + len(a)] += a
        times.append({"start": t0, "dur": round(len(a) / SR, 2), "text": sub, "rate": rate})
        print(f"  {t0:5.1f}s  {len(a) / SR:4.1f}s  语速{rate}  {sub}")
    # 旁白出现时压低配乐（侧链）
    env = np.abs(voice)
    k = int(0.08 * SR); env = np.convolve(env, np.ones(k) / k, "same")
    env = np.clip(env / 0.06, 0, 1)
    k2 = int(0.25 * SR); env = np.convolve(env, np.ones(k2) / k2, "same")
    duck = 1 - 0.5 * env
    mix = music * duck[:, None] + (voice * 1.15)[:, None]
    fade = np.ones(N); nf = int(1.2 * SR); fade[-nf:] = np.linspace(1, 0, nf)
    mix = np.tanh(mix * 1.1) * fade[:, None]
    mix = mix / np.abs(mix).max() * 0.94
    for name, sig in (("音轨.wav", mix), ("配乐_无旁白.wav", np.tanh(music * 1.1) * fade[:, None] * 0.94)):
        with wave.open(os.path.join(OUT, name), "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((np.clip(sig, -1, 1) * 32767).astype(np.int16).tobytes())
    with open(os.path.join(OUT, "旁白时间.json"), "w", encoding="utf-8") as f:
        json.dump(times, f, ensure_ascii=False, indent=1)
    print("已生成 素材/音轨.wav、素材/配乐_无旁白.wav")


if __name__ == "__main__":
    main()

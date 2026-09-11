/* ==========================================================================
   audio.js — مؤثّرات صوتية مولَّدة بـ WebAudio (بدون ملفات صوت خارجية)
   ========================================================================== */
'use strict';

const Sfx = (function () {
  let ctx = null, master = null, enabled = true, musicOn = false, musicTimer = null;

  function ensure() {
    if (ctx) return ctx;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
    master = ctx.createGain();
    master.gain.value = 0.28;
    master.connect(ctx.destination);
    return ctx;
  }

  function resume() {
    ensure();
    if (ctx && ctx.state === 'suspended') ctx.resume();
  }

  /** نغمة بسيطة */
  function tone(freq, dur, type, vol, slideTo, delay) {
    if (!enabled) return;
    if (!ensure()) return;
    const t0 = ctx.currentTime + (delay || 0);
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = type || 'sine';
    osc.frequency.setValueAtTime(freq, t0);
    if (slideTo) osc.frequency.exponentialRampToValueAtTime(Math.max(40, slideTo), t0 + dur);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(vol == null ? 0.5 : vol, t0 + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.connect(g); g.connect(master);
    osc.start(t0); osc.stop(t0 + dur + 0.02);
  }

  /** ضجيج أبيض قصير — للانفجارات */
  function noise(dur, vol, filterFreq, delay) {
    if (!enabled) return;
    if (!ensure()) return;
    const t0 = ctx.currentTime + (delay || 0);
    const n = Math.floor(ctx.sampleRate * dur);
    const buf = ctx.createBuffer(1, n, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / n);
    const src = ctx.createBufferSource(); src.buffer = buf;
    const f = ctx.createBiquadFilter();
    f.type = 'lowpass'; f.frequency.value = filterFreq || 1200;
    const g = ctx.createGain(); g.gain.value = vol == null ? 0.5 : vol;
    src.connect(f); f.connect(g); g.connect(master);
    src.start(t0);
  }

  const API = {
    setEnabled(v) { enabled = !!v; if (enabled) resume(); },
    isEnabled() { return enabled; },
    resume,

    swap()      { tone(520, 0.09, 'sine', 0.35, 700); },
    invalid()   { tone(180, 0.12, 'square', 0.22, 120); },
    /** صوت المطابقة يرتفع مع كل موجة تتابع */
    match(chain) {
      const base = 440 * Math.pow(1.1225, Math.min(chain || 0, 10));
      tone(base, 0.13, 'triangle', 0.42, base * 1.5);
      tone(base * 2, 0.09, 'sine', 0.18, base * 2.6, 0.02);
    },
    fall()      { tone(240, 0.06, 'sine', 0.14, 180); },
    create()    { tone(660, 0.16, 'triangle', 0.4, 1320); tone(990, 0.2, 'sine', 0.22, 1600, 0.05); },
    rocket()    { noise(0.35, 0.4, 2600); tone(900, 0.3, 'sawtooth', 0.16, 200); },
    tnt()       { noise(0.5, 0.75, 700); tone(90, 0.4, 'square', 0.3, 40); },
    ball()      { for (let i = 0; i < 6; i++) tone(500 + i * 180, 0.18, 'sine', 0.22, 1800, i * 0.045); },
    combo()     { noise(0.7, 0.85, 900); for (let i = 0; i < 4; i++) tone(180 + i * 90, 0.5, 'sawtooth', 0.2, 60, i * 0.06); },
    obstacle()  { noise(0.22, 0.45, 1800); tone(300, 0.12, 'square', 0.16, 140); },
    collect()   { tone(880, 0.12, 'sine', 0.3, 1200); tone(1320, 0.14, 'sine', 0.2, 1760, 0.06); },
    goal()      { tone(740, 0.1, 'triangle', 0.3, 1100); },
    win() {
      const notes = [523, 659, 784, 1047, 1319];
      notes.forEach((f, i) => tone(f, 0.34, 'triangle', 0.38, f, i * 0.11));
    },
    lose() {
      const notes = [440, 392, 330, 262];
      notes.forEach((f, i) => tone(f, 0.32, 'sine', 0.34, f * 0.96, i * 0.14));
    },
    click()     { tone(700, 0.05, 'square', 0.16, 900); },
    coin()      { tone(1000, 0.08, 'square', 0.2, 1500); tone(1500, 0.1, 'square', 0.14, 2000, 0.05); },
    shuffle()   { for (let i = 0; i < 8; i++) tone(300 + Math.random() * 700, 0.07, 'triangle', 0.12, null, i * 0.04); },
    lastMoves() { tone(660, 0.1, 'square', 0.2, 880); tone(660, 0.1, 'square', 0.2, 880, 0.18); },

    /** موسيقى خلفية بسيطة جداً: دورة نغمات هادئة */
    setMusic(on) {
      musicOn = !!on;
      if (musicTimer) { clearInterval(musicTimer); musicTimer = null; }
      if (!musicOn) return;
      ensure();
      const scale = [262, 294, 330, 392, 440, 523, 587, 659];
      let step = 0;
      musicTimer = setInterval(() => {
        if (!enabled || !musicOn) return;
        const f = scale[step % scale.length];
        tone(f / 2, 0.7, 'sine', 0.06);
        if (step % 4 === 0) tone(f, 0.5, 'triangle', 0.04, f * 1.01);
        step++;
      }, 620);
    },
    isMusicOn() { return musicOn; },
  };

  return API;
})();

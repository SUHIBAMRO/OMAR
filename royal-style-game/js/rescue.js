/* ==========================================================================
   rescue.js — شاشة «مهمّات الإنقاذ»: العرض والإدخال وإدارة المرحلة
   المحاكاة نفسها في rescue-sim.js ولا تعتمد على المتصفّح.
   ========================================================================== */
'use strict';

const Rescue = (function () {

  const STEPS_PER_FRAME = 3;      // خطوات فيزياء لكل إطار (كلّما زادت أسرعت السوائل)

  const S = {
    sim: null, def: null, levelNo: 1,
    canvas: null, ctx: null, wrap: null,
    unit: 20, ox: 0, oy: 0, w: 0, h: 0,
    off: null, offCtx: null, img: null,
    running: false, lastTs: 0, done: false,
    shakeT: 0,
  };

  /* ------------------------------------------------------------------ */
  /*                              التشغيل                                */
  /* ------------------------------------------------------------------ */

  function start(levelNo) {
    if (typeof Game !== 'undefined') Game.stop();
    S.levelNo = U.clamp(levelNo, 1, RESCUE_LEVELS.length);
    S.def = RESCUE_LEVELS[S.levelNo - 1];
    S.sim = new RescueSim(S.def);
    S.done = false;
    S.shakeT = 0;
    FX.reset();

    S.canvas = U.$('#rescue-canvas');
    S.ctx = S.canvas.getContext('2d');
    S.wrap = U.$('#rescue-wrap');

    /* لوحة مؤقّتة بدقّة المحاكاة تُكبَّر عند الرسم فتبدو السوائل ناعمة */
    S.off = document.createElement('canvas');
    S.off.width = S.sim.W; S.off.height = S.sim.H;
    S.offCtx = S.off.getContext('2d');
    S.img = S.offCtx.createImageData(S.sim.W, S.sim.H);

    bindInput();
    resize();
    updateHud(true);
    U.$('#rescue-hint').textContent = S.def.hint || '';

    if (!S.running) { S.running = true; S.lastTs = performance.now(); requestAnimationFrame(loop); }
  }

  function stop() { S.running = false; }

  function resize() {
    if (!S.canvas || !S.sim) return;
    const r = S.wrap.getBoundingClientRect();
    const aw = S.sim.artW, ah = S.sim.artH;
    const unit = Math.floor(Math.min((r.width - 12) / aw, (r.height - 12) / ah));
    S.unit = Math.max(8, unit);
    S.w = aw * S.unit; S.h = ah * S.unit;
    S.ox = 0; S.oy = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    S.canvas.width = Math.round(S.w * dpr);
    S.canvas.height = Math.round(S.h * dpr);
    S.canvas.style.width = S.w + 'px';
    S.canvas.style.height = S.h + 'px';
    S.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* ------------------------------------------------------------------ */
  /*                              الإدخال                                */
  /* ------------------------------------------------------------------ */

  let bound = false;
  function bindInput() {
    if (bound) return;
    bound = true;
    S.canvas.addEventListener('pointerdown', onDown);
    S.canvas.addEventListener('contextmenu', e => e.preventDefault());
  }

  function onDown(e) {
    if (!S.sim || S.sim.status !== 'playing' || S.done) return;
    Sfx.resume();
    const r = S.canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const pin = pinAtPoint(x, y);
    if (!pin) return;
    if (S.sim.pull(pin.id)) {
      pin.anim = 1;
      Sfx.swap();
      FX.burst(S.ox + (pin.ax + pin.aw / 2) * S.unit,
               S.oy + (pin.ay + pin.ah / 2) * S.unit, '#ffe08a', 10, 0.8);
    }
  }

  function pinAtPoint(x, y) {
    const pad = 10;
    for (const p of S.sim.pins) {
      if (p.pulled) continue;
      const px = S.ox + p.ax * S.unit, py = S.oy + p.ay * S.unit;
      const pw = p.aw * S.unit, ph = p.ah * S.unit;
      if (x >= px - pad && x <= px + pw + pad && y >= py - pad && y <= py + ph + pad) return p;
    }
    return null;
  }

  /* ------------------------------------------------------------------ */
  /*                             حلقة اللعب                              */
  /* ------------------------------------------------------------------ */

  function loop(ts) {
    if (!S.running) return;
    const dt = Math.min(0.05, (ts - S.lastTs) / 1000);
    S.lastTs = ts;
    update(dt);
    draw();
    requestAnimationFrame(loop);
  }

  function update(dt) {
    if (S.shakeT > 0) S.shakeT -= dt * 1000;
    FX.update(dt);
    for (const p of S.sim.pins) if (p.anim > 0) p.anim = Math.max(0, p.anim - dt * 4);

    if (S.sim.status === 'playing' && !S.done) {
      for (let i = 0; i < STEPS_PER_FRAME; i++) {
        S.sim.step();
        S.sim.advanceTime(dt / STEPS_PER_FRAME);
        if (S.sim.status !== 'playing') break;
      }
      drainEvents();
      updateHud();
    }

    if (!S.done && S.sim.status !== 'playing') {
      S.done = true;
      setTimeout(() => S.sim.status === 'won' ? finishWin() : finishLose(), 700);
    }
  }

  /** يحوّل أحداث المحاكاة إلى مؤثّرات */
  function drainEvents() {
    const ev = S.sim.events;
    if (!ev.length) return;
    let burns = 0, collects = 0, steams = 0;
    for (const e of ev) {
      const x = S.ox + (e.x + 0.5) * (S.unit / S.sim.scale);
      const y = S.oy + (e.y + 0.5) * (S.unit / S.sim.scale);
      switch (e.type) {
        case 'burn':
          if (burns++ < 3) { FX.burst(x, y, '#ff8a3d', 5, 0.9); }
          break;
        case 'steam':
          if (steams++ < 3) { FX.burst(x, y, '#e8f4ff', 4, 0.7); }
          break;
        case 'collect':
          if (collects++ < 2) { FX.burst(x, y, '#ffcc4d', 5, 0.7); }
          break;
        case 'eat':
          FX.burst(x, y, '#8a1f1f', 4, 0.8);
          break;
        default: break;
      }
    }
    if (burns) Sfx.obstacle();
    if (steams) Sfx.rocket();
    if (collects) Sfx.coin();
    ev.length = 0;
  }

  /* ------------------------------------------------------------------ */
  /*                               الرسم                                 */
  /* ------------------------------------------------------------------ */

  const COLORS = {
    [MAT.LAVA]:  [255, 110, 30],
    [MAT.WATER]: [60, 160, 235],
    [MAT.STONE]: [120, 128, 150],
  };

  function draw() {
    const ctx = S.ctx;
    if (!ctx || !S.sim) return;
    const u = S.unit, sim = S.sim;
    ctx.clearRect(0, 0, S.w, S.h);

    ctx.save();
    if (S.shakeT > 0) ctx.translate((Math.random() - 0.5) * 6, (Math.random() - 0.5) * 6);

    /* خلفية الكهف */
    const bg = ctx.createLinearGradient(0, 0, 0, S.h);
    bg.addColorStop(0, '#1b2043');
    bg.addColorStop(1, '#0e1130');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, S.w, S.h);

    /* 1) الجدران من الرسم الأصلي (حوافّ حادّة ومقروءة) */
    for (let ay = 0; ay < sim.artH; ay++) {
      const row = sim.def.art[ay];
      for (let ax = 0; ax < sim.artW; ax++) {
        if (row[ax] !== '#') continue;
        const x = S.ox + ax * u, y = S.oy + ay * u;
        const g = ctx.createLinearGradient(x, y, x, y + u);
        g.addColorStop(0, '#4a5280');
        g.addColorStop(1, '#2b3160');
        ctx.fillStyle = g;
        ctx.fillRect(x, y, u, u);
        ctx.strokeStyle = 'rgba(0,0,0,.35)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, u - 1, u - 1);
      }
    }

    /* 2) السوائل والصخر: صورة بدقّة المحاكاة تُكبَّر بنعومة */
    const d = S.img.data;
    const t = performance.now() / 300;
    for (let i = 0, n = sim.W * sim.H; i < n; i++) {
      const m = sim.grid[i];
      const c = COLORS[m];
      if (!c) { d[i * 4 + 3] = 0; continue; }
      let r = c[0], g = c[1], b = c[2];
      if (m === MAT.LAVA) {
        const f = 0.75 + 0.25 * Math.sin(t + (i % sim.W) * 0.4 + Math.floor(i / sim.W) * 0.3);
        r = Math.min(255, r * f + 60); g = Math.min(255, g * f); b = Math.min(255, b * f);
      }
      d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
    }
    S.offCtx.putImageData(S.img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(S.off, 0, 0, sim.W, sim.H, S.ox, S.oy, S.w, S.h);

    /* توهّج الحمم */
    ctx.save();
    ctx.globalAlpha = 0.25;
    ctx.globalCompositeOperation = 'lighter';
    ctx.drawImage(S.off, 0, 0, sim.W, sim.H, S.ox - 2, S.oy - 2, S.w + 4, S.h + 4);
    ctx.restore();

    /* 3) الذهب والأفعى: دوائر حتى يبدوا حبيبات وجسماً لا بقعاً */
    const cs = u / sim.scale;
    let snakeHead = null;
    ctx.save();
    for (let y = 0; y < sim.H; y++) {
      for (let x = 0; x < sim.W; x++) {
        const m = sim.grid[y * sim.W + x];
        if (m !== MAT.GOLD && m !== MAT.SNAKE) continue;
        const px = S.ox + (x + 0.5) * cs, py = S.oy + (y + 0.5) * cs;
        if (m === MAT.GOLD) {
          ctx.fillStyle = '#ffcc4d';
          ctx.beginPath(); ctx.arc(px, py, cs * 0.62, 0, 7); ctx.fill();
          ctx.fillStyle = 'rgba(255,255,255,.45)';
          ctx.beginPath(); ctx.arc(px - cs * 0.18, py - cs * 0.18, cs * 0.2, 0, 7); ctx.fill();
        } else {
          if (!snakeHead) snakeHead = [px, py];
          ctx.fillStyle = '#2f9c41';
          ctx.beginPath(); ctx.arc(px, py, cs * 0.78, 0, 7); ctx.fill();
          ctx.fillStyle = '#5fd36f';
          ctx.beginPath(); ctx.arc(px, py - cs * 0.15, cs * 0.42, 0, 7); ctx.fill();
        }
      }
    }
    /* عينا الأفعى */
    if (snakeHead) {
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(snakeHead[0] - cs * 0.5, snakeHead[1] - cs * 0.3, cs * 0.35, 0, 7); ctx.fill();
      ctx.beginPath(); ctx.arc(snakeHead[0] + cs * 0.5, snakeHead[1] - cs * 0.3, cs * 0.35, 0, 7); ctx.fill();
      ctx.fillStyle = '#111';
      ctx.beginPath(); ctx.arc(snakeHead[0] - cs * 0.5, snakeHead[1] - cs * 0.3, cs * 0.16, 0, 7); ctx.fill();
      ctx.beginPath(); ctx.arc(snakeHead[0] + cs * 0.5, snakeHead[1] - cs * 0.3, cs * 0.16, 0, 7); ctx.fill();
    }
    ctx.restore();

    /* 4) الملك وحوض الجمع */
    drawKing(ctx, u);

    /* 5) الدبابيس */
    for (const p of sim.pins) drawPin(ctx, p, u);

    /* 6) المؤثّرات */
    FX.draw(ctx);
    ctx.restore();
  }

  function drawKing(ctx, u) {
    const sim = S.sim;
    /* حدود منطقة الملك بإحداثيات الرسم */
    let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1;
    for (let ay = 0; ay < sim.artH; ay++) {
      const row = sim.def.art[ay];
      for (let ax = 0; ax < sim.artW; ax++) {
        if (row[ax] !== 'K') continue;
        x0 = Math.min(x0, ax); y0 = Math.min(y0, ay);
        x1 = Math.max(x1, ax); y1 = Math.max(y1, ay);
      }
    }
    if (x1 < 0) return;
    const x = S.ox + x0 * u, y = S.oy + y0 * u;
    const w = (x1 - x0 + 1) * u, h = (y1 - y0 + 1) * u;

    /* حوض الجمع */
    ctx.save();
    const g = ctx.createLinearGradient(x, y, x, y + h);
    g.addColorStop(0, 'rgba(255,224,138,.22)');
    g.addColorStop(1, 'rgba(201,127,5,.35)');
    ctx.fillStyle = g;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = '#ffcc4d';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
    ctx.setLineDash([]);
    ctx.restore();

    /* الملك في منتصف الحوض */
    const cx = x + w / 2, by = y + h;
    const k = Math.min(u * 1.5, w * 0.8);
    ctx.save();
    ctx.fillStyle = '#5b3fa8';
    ctx.beginPath();
    ctx.moveTo(cx - k * 0.32, by);
    ctx.lineTo(cx - k * 0.22, by - k * 0.55);
    ctx.lineTo(cx + k * 0.22, by - k * 0.55);
    ctx.lineTo(cx + k * 0.32, by);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#ffd9b0';
    ctx.beginPath(); ctx.arc(cx, by - k * 0.68, k * 0.2, 0, 7); ctx.fill();
    ctx.fillStyle = '#ffcc4d';
    ctx.beginPath();
    ctx.moveTo(cx - k * 0.24, by - k * 0.82);
    ctx.lineTo(cx - k * 0.24, by - k * 1.08);
    ctx.lineTo(cx - k * 0.08, by - k * 0.94);
    ctx.lineTo(cx, by - k * 1.14);
    ctx.lineTo(cx + k * 0.08, by - k * 0.94);
    ctx.lineTo(cx + k * 0.24, by - k * 1.08);
    ctx.lineTo(cx + k * 0.24, by - k * 0.82);
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  function drawPin(ctx, p, u) {
    if (p.pulled && p.anim <= 0) return;
    const slide = p.pulled ? (1 - p.anim) * u * 3 : 0;
    const alpha = p.pulled ? p.anim : 1;
    let x = S.ox + p.ax * u, y = S.oy + p.ay * u;
    const w = p.aw * u, h = p.ah * u;
    if (p.dir === 'h') y -= slide; else x -= slide;

    ctx.save();
    ctx.globalAlpha = alpha;
    const g = ctx.createLinearGradient(x, y, x, y + h);
    g.addColorStop(0, '#e9edf7');
    g.addColorStop(0.5, '#b9c3d6');
    g.addColorStop(1, '#78849f');
    ctx.fillStyle = g;
    const r = Math.min(w, h) * 0.35;
    roundRect(ctx, x + 1, y + 1, w - 2, h - 2, r);
    ctx.fill();
    ctx.strokeStyle = '#2b3160';
    ctx.lineWidth = 2;
    ctx.stroke();

    /* حلقة السحب في طرف الدبّوس */
    const hx = p.dir === 'h' ? x + w / 2 : x + w / 2;
    const hy = p.dir === 'h' ? y + h / 2 : y + h / 2;
    ctx.strokeStyle = '#ffcc4d';
    ctx.lineWidth = Math.max(2, u * 0.14);
    ctx.beginPath();
    ctx.arc(hx, hy, Math.min(w, h) * 0.28, 0, 7);
    ctx.stroke();
    ctx.restore();
  }

  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  /* ------------------------------------------------------------------ */
  /*                             شريط الحالة                             */
  /* ------------------------------------------------------------------ */

  function updateHud(force) {
    const t = U.$('#rescue-time');
    if (t) {
      const txt = U.mmss(S.sim.time);
      if (force || t.textContent !== txt) {
        t.textContent = txt;
        t.parentElement.classList.toggle('low', S.sim.time <= 15);
      }
    }
    const g = U.$('#rescue-gold');
    if (g) {
      const txt = Math.min(S.sim.collected, S.sim.goldTarget) + '/' + S.sim.goldTarget;
      if (force || g.textContent !== txt) g.textContent = txt;
    }
    const n = U.$('#rescue-name');
    if (n && force) n.textContent = 'المهمّة ' + S.levelNo + ' — ' + S.def.name;
  }

  /* ------------------------------------------------------------------ */
  /*                            نهاية المهمّة                            */
  /* ------------------------------------------------------------------ */

  function stars() {
    const frac = S.sim.time / (S.def.time || 1);
    if (frac >= 0.6) return 3;
    if (frac >= 0.3) return 2;
    return 1;
  }

  function finishWin() {
    const st = stars();
    const gained = Save.recordRescueWin(S.levelNo, st);
    const coins = 30 + gained * 10;
    Save.addCoins(coins);
    Sfx.win();
    UI.showRescueWin({ level: S.levelNo, stars: st, coins, last: S.levelNo >= RESCUE_LEVELS.length });
  }

  function finishLose() {
    Sfx.lose();
    S.shakeT = 400;
    UI.showRescueLose({ level: S.levelNo, reason: S.sim.reason });
  }

  function retry() { start(S.levelNo); }

  return {
    start, stop, resize, retry,
    get levelNo() { return S.levelNo; },
    get sim() { return S.sim; },
    S,
  };
})();

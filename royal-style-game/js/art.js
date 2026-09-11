/* ==========================================================================
   art.js — كل الرسومات مولَّدة برمجياً على Canvas (لا صور خارجية إطلاقاً)
   كل شكل يُرسم مرّة واحدة في ذاكرة مؤقتة ثم يُنسخ، للسرعة.
   ========================================================================== */
'use strict';

const Art = (function () {

  const cache = new Map();

  function surface(size) {
    const c = document.createElement('canvas');
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    c.width = Math.round(size * dpr);
    c.height = Math.round(size * dpr);
    c._dpr = dpr;
    const ctx = c.getContext('2d');
    ctx.scale(dpr, dpr);
    return { c, ctx };
  }

  /* ------------------------- أدوات رسم صغيرة ------------------------- */
  function rr(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function grad(ctx, x0, y0, x1, y1, a, b) {
    const g = ctx.createLinearGradient(x0, y0, x1, y1);
    g.addColorStop(0, a); g.addColorStop(1, b);
    return g;
  }

  function stroked(ctx, color, width) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.stroke();
  }

  /* لمعة زجاجية أعلى القطعة */
  function gloss(ctx, s, cx, cy, rx, ry) {
    ctx.save();
    ctx.globalAlpha = 0.35;
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, -0.35, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  /* ======================= القطع الملوّنة ======================= */

  function drawCrown(ctx, s, t) {
    const w = s * 0.74, h = s * 0.58, x = (s - w) / 2, y = (s - h) / 2 + s * 0.04;
    ctx.beginPath();
    ctx.moveTo(x, y + h);
    ctx.lineTo(x, y + h * 0.28);
    ctx.lineTo(x + w * 0.24, y + h * 0.62);
    ctx.lineTo(x + w * 0.5, y);
    ctx.lineTo(x + w * 0.76, y + h * 0.62);
    ctx.lineTo(x + w, y + h * 0.28);
    ctx.lineTo(x + w, y + h);
    ctx.closePath();
    ctx.fillStyle = grad(ctx, 0, y, 0, y + h, t.light, t.color);
    ctx.fill();
    ctx.save(); ctx.clip();
    gloss(ctx, s, x + w * 0.3, y + h * 0.45, s * 0.1, s * 0.05);
    ctx.restore();
    stroked(ctx, t.dark, s * 0.055);
    /* قاعدة التاج */
    rr(ctx, x - s * 0.02, y + h * 0.78, w + s * 0.04, h * 0.3, s * 0.05);
    ctx.fillStyle = grad(ctx, 0, y + h * 0.78, 0, y + h * 1.08, t.color, t.dark);
    ctx.fill();
    stroked(ctx, t.dark, s * 0.05);
    /* جواهر */
    const jew = ['#e8443f', '#3b8ee2', '#41c153'];
    for (let i = 0; i < 3; i++) {
      ctx.beginPath();
      ctx.arc(x + w * (0.22 + i * 0.28), y + h * 0.93, s * 0.045, 0, 7);
      ctx.fillStyle = jew[i]; ctx.fill();
    }
  }

  function drawBook(ctx, s, t) {
    const w = s * 0.66, h = s * 0.72, x = (s - w) / 2, y = (s - h) / 2;
    /* الغلاف */
    rr(ctx, x, y, w, h, s * 0.09);
    ctx.fillStyle = grad(ctx, x, y, x + w, y + h, t.light, t.color);
    ctx.fill();
    ctx.save(); ctx.clip();
    gloss(ctx, s, x + w * 0.42, y + h * 0.2, s * 0.11, s * 0.05);
    ctx.restore();
    stroked(ctx, t.dark, s * 0.055);
    /* الكعب */
    rr(ctx, x, y, w * 0.2, h, s * 0.08);
    ctx.fillStyle = t.dark; ctx.fill();
    /* صفحات */
    ctx.fillStyle = '#fdf6e2';
    rr(ctx, x + w * 0.78, y + h * 0.08, w * 0.16, h * 0.84, s * 0.03);
    ctx.fill();
    stroked(ctx, t.dark, s * 0.03);
    /* شعار */
    ctx.fillStyle = 'rgba(255,255,255,.85)';
    ctx.beginPath();
    const cx = x + w * 0.5, cy = y + h * 0.45, r = s * 0.11;
    for (let i = 0; i < 5; i++) {
      const a = -Math.PI / 2 + i * Math.PI * 2 / 5;
      const a2 = a + Math.PI / 5;
      ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
      ctx.lineTo(cx + Math.cos(a2) * r * 0.45, cy + Math.sin(a2) * r * 0.45);
    }
    ctx.closePath(); ctx.fill();
  }

  function drawShield(ctx, s, t) {
    const w = s * 0.66, h = s * 0.74, x = (s - w) / 2, y = (s - h) / 2;
    ctx.beginPath();
    ctx.moveTo(x + w * 0.5, y);
    ctx.lineTo(x + w, y + h * 0.16);
    ctx.lineTo(x + w, y + h * 0.55);
    ctx.quadraticCurveTo(x + w, y + h * 0.9, x + w * 0.5, y + h);
    ctx.quadraticCurveTo(x, y + h * 0.9, x, y + h * 0.55);
    ctx.lineTo(x, y + h * 0.16);
    ctx.closePath();
    ctx.fillStyle = grad(ctx, x, y, x + w, y + h, t.light, t.color);
    ctx.fill();
    stroked(ctx, t.dark, s * 0.055);
    /* الشريط العمودي */
    ctx.save();
    ctx.clip();
    ctx.fillStyle = 'rgba(255,255,255,.55)';
    ctx.fillRect(x + w * 0.42, y, w * 0.16, h);
    ctx.fillRect(x, y + h * 0.3, w, h * 0.12);
    gloss(ctx, s, x + w * 0.26, y + h * 0.18, s * 0.09, s * 0.045);
    ctx.restore();
  }

  function drawLeaf(ctx, s, t) {
    const cx = s / 2, cy = s * 0.52, r = s * 0.34;
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 1.15);
    ctx.bezierCurveTo(cx + r * 1.25, cy - r * 0.15, cx + r * 0.7, cy + r * 0.95, cx, cy + r * 0.55);
    ctx.bezierCurveTo(cx - r * 0.7, cy + r * 0.95, cx - r * 1.25, cy - r * 0.15, cx, cy - r * 1.15);
    ctx.closePath();
    ctx.fillStyle = grad(ctx, cx - r, cy - r, cx + r, cy + r, t.light, t.color);
    ctx.fill();
    ctx.save(); ctx.clip();
    gloss(ctx, s, cx - r * 0.38, cy - r * 0.4, s * 0.08, s * 0.04);
    ctx.restore();
    stroked(ctx, t.dark, s * 0.055);
    /* الساق */
    ctx.beginPath();
    ctx.moveTo(cx, cy + r * 0.45);
    ctx.lineTo(cx, cy + r * 1.0);
    stroked(ctx, t.dark, s * 0.07);
    /* العرق الأوسط */
    ctx.beginPath();
    ctx.moveTo(cx, cy + r * 0.4);
    ctx.lineTo(cx, cy - r * 0.85);
    stroked(ctx, 'rgba(255,255,255,.5)', s * 0.035);
  }

  function drawGem(ctx, s, t) {
    const cx = s / 2, cy = s * 0.52, w = s * 0.62, h = s * 0.66;
    ctx.beginPath();
    ctx.moveTo(cx, cy - h / 2);
    ctx.lineTo(cx + w / 2, cy - h * 0.12);
    ctx.lineTo(cx, cy + h / 2);
    ctx.lineTo(cx - w / 2, cy - h * 0.12);
    ctx.closePath();
    ctx.fillStyle = grad(ctx, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, t.light, t.color);
    ctx.fill();
    ctx.save(); ctx.clip();
    gloss(ctx, s, cx - w * 0.16, cy - h * 0.18, s * 0.07, s * 0.035);
    ctx.restore();
    stroked(ctx, t.dark, s * 0.055);
    /* أوجه القطع */
    ctx.beginPath();
    ctx.moveTo(cx - w / 2, cy - h * 0.12);
    ctx.lineTo(cx, cy - h * 0.02);
    ctx.lineTo(cx + w / 2, cy - h * 0.12);
    ctx.moveTo(cx, cy - h / 2);
    ctx.lineTo(cx, cy - h * 0.02);
    stroked(ctx, 'rgba(255,255,255,.55)', s * 0.035);
  }

  function drawPotion(ctx, s, t) {
    const cx = s / 2, cy = s * 0.58, r = s * 0.25;
    /* الجسم */
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = grad(ctx, cx - r, cy - r, cx + r, cy + r, t.light, t.color);
    ctx.fill();
    ctx.save(); ctx.clip();
    gloss(ctx, s, cx - r * 0.42, cy - r * 0.42, s * 0.06, s * 0.03);
    ctx.restore();
    stroked(ctx, t.dark, s * 0.055);
    /* العنق */
    rr(ctx, cx - s * 0.075, s * 0.2, s * 0.15, s * 0.18, s * 0.03);
    ctx.fillStyle = t.color; ctx.fill();
    stroked(ctx, t.dark, s * 0.05);
    /* السدادة */
    rr(ctx, cx - s * 0.095, s * 0.15, s * 0.19, s * 0.09, s * 0.03);
    ctx.fillStyle = '#c9a06a'; ctx.fill();
    stroked(ctx, '#7d5a2e', s * 0.04);
    /* فقاعات */
    ctx.fillStyle = 'rgba(255,255,255,.65)';
    ctx.beginPath(); ctx.arc(cx - r * 0.3, cy + r * 0.2, s * 0.035, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.arc(cx + r * 0.25, cy - r * 0.1, s * 0.025, 0, 7); ctx.fill();
  }

  const DRAWERS = [drawCrown, drawBook, drawShield, drawLeaf, drawGem, drawPotion];

  /* ======================= القطع الخاصة ======================= */

  function drawRocket(ctx, s, vertical, tint) {
    ctx.save();
    ctx.translate(s / 2, s / 2);
    if (vertical) ctx.rotate(-Math.PI / 2);
    ctx.translate(-s / 2, -s / 2);

    const w = s * 0.82, h = s * 0.42, x = (s - w) / 2, y = (s - h) / 2;
    /* الجسم */
    ctx.beginPath();
    ctx.moveTo(x + w * 0.18, y);
    ctx.lineTo(x + w * 0.78, y);
    ctx.quadraticCurveTo(x + w, y + h * 0.5, x + w * 0.78, y + h);
    ctx.lineTo(x + w * 0.18, y + h);
    ctx.quadraticCurveTo(x + w * 0.02, y + h * 0.5, x + w * 0.18, y);
    ctx.closePath();
    ctx.fillStyle = grad(ctx, 0, y, 0, y + h, '#fdfdff', '#b9c3d6');
    ctx.fill();
    stroked(ctx, '#3a4664', s * 0.05);
    /* الرأس */
    ctx.beginPath();
    ctx.moveTo(x + w * 0.72, y - h * 0.08);
    ctx.lineTo(x + w * 1.02, y + h * 0.5);
    ctx.lineTo(x + w * 0.72, y + h * 1.08);
    ctx.closePath();
    ctx.fillStyle = tint || '#e8443f';
    ctx.fill();
    stroked(ctx, '#8c1b1b', s * 0.045);
    /* الزعانف */
    ctx.beginPath();
    ctx.moveTo(x + w * 0.2, y);
    ctx.lineTo(x + w * 0.06, y - h * 0.34);
    ctx.lineTo(x + w * 0.3, y);
    ctx.moveTo(x + w * 0.2, y + h);
    ctx.lineTo(x + w * 0.06, y + h * 1.34);
    ctx.lineTo(x + w * 0.3, y + h);
    ctx.fillStyle = tint || '#e8443f';
    ctx.fill();
    stroked(ctx, '#8c1b1b', s * 0.04);
    /* النافذة */
    ctx.beginPath();
    ctx.arc(x + w * 0.48, y + h * 0.5, h * 0.22, 0, 7);
    ctx.fillStyle = '#7fd8ff'; ctx.fill();
    stroked(ctx, '#3a4664', s * 0.035);
    ctx.restore();
  }

  function drawTNT(ctx, s) {
    const w = s * 0.66, h = s * 0.6, x = (s - w) / 2, y = (s - h) / 2 + s * 0.06;
    /* البرميل */
    rr(ctx, x, y, w, h, s * 0.08);
    ctx.fillStyle = grad(ctx, x, y, x, y + h, '#f16b6b', '#a41d1d');
    ctx.fill();
    stroked(ctx, '#6c0f0f', s * 0.055);
    /* الأشرطة */
    ctx.fillStyle = 'rgba(255,255,255,.85)';
    ctx.fillRect(x + w * 0.08, y + h * 0.34, w * 0.84, h * 0.3);
    ctx.fillStyle = '#6c0f0f';
    ctx.font = 'bold ' + (s * 0.2) + 'px Arial';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('TNT', s / 2, y + h * 0.5);
    /* الفتيل */
    ctx.beginPath();
    ctx.moveTo(s / 2, y);
    ctx.quadraticCurveTo(s * 0.66, y - s * 0.12, s * 0.6, y - s * 0.2);
    stroked(ctx, '#7d5a2e', s * 0.05);
    /* الشرارة */
    ctx.beginPath();
    ctx.arc(s * 0.6, y - s * 0.22, s * 0.06, 0, 7);
    ctx.fillStyle = '#ffd76a'; ctx.fill();
    ctx.beginPath();
    ctx.arc(s * 0.6, y - s * 0.22, s * 0.032, 0, 7);
    ctx.fillStyle = '#fff'; ctx.fill();
  }

  function drawBall(ctx, s) {
    const cx = s / 2, cy = s / 2, r = s * 0.34;
    const g = ctx.createRadialGradient(cx - r * .3, cy - r * .3, r * .12, cx, cy, r);
    g.addColorStop(0, '#ffffff');
    g.addColorStop(0.45, '#ffe08a');
    g.addColorStop(1, '#ff7ad1');
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7);
    ctx.fillStyle = g; ctx.fill();
    stroked(ctx, '#ffffff', s * 0.04);
    /* أقواس قوس قزح */
    const cols = ['#ff5f6d', '#ffc371', '#5ee7a7', '#57b8ff', '#c86dd7'];
    for (let i = 0; i < cols.length; i++) {
      ctx.beginPath();
      ctx.arc(cx, cy, r * (0.9 - i * 0.14), i * 1.2, i * 1.2 + 1.5);
      stroked(ctx, cols[i], s * 0.045);
    }
    /* بريق */
    ctx.fillStyle = '#fff';
    for (const p of [[0.22, 0.22, 0.075], [0.78, 0.3, 0.05], [0.68, 0.76, 0.06]]) {
      const px = s * p[0], py = s * p[1], pr = s * p[2];
      ctx.beginPath();
      ctx.moveTo(px, py - pr);
      ctx.quadraticCurveTo(px + pr * .2, py - pr * .2, px + pr, py);
      ctx.quadraticCurveTo(px + pr * .2, py + pr * .2, px, py + pr);
      ctx.quadraticCurveTo(px - pr * .2, py + pr * .2, px - pr, py);
      ctx.quadraticCurveTo(px - pr * .2, py - pr * .2, px, py - pr);
      ctx.fill();
    }
  }

  /* ======================= العوائق ======================= */

  function drawBox(ctx, s, hp) {
    const p = s * 0.06, w = s - p * 2;
    rr(ctx, p, p, w, w, s * 0.09);
    ctx.fillStyle = grad(ctx, p, p, p, p + w, hp > 1 ? '#c9955a' : '#e0b277', hp > 1 ? '#7d5a2e' : '#a3743c');
    ctx.fill();
    stroked(ctx, '#5c3f1d', s * 0.06);
    /* ألواح خشبية */
    ctx.save(); ctx.clip();
    ctx.strokeStyle = 'rgba(92,63,29,.55)';
    ctx.lineWidth = s * 0.035;
    for (let i = 1; i < 3; i++) {
      ctx.beginPath();
      ctx.moveTo(p, p + w * i / 3); ctx.lineTo(p + w, p + w * i / 3);
      ctx.stroke();
    }
    /* قطران قطري */
    ctx.beginPath();
    ctx.moveTo(p, p + w); ctx.lineTo(p + w, p);
    ctx.strokeStyle = 'rgba(255,255,255,.25)';
    ctx.stroke();
    ctx.restore();
    if (hp > 1) {
      /* حزام معدني للصندوق ذي الطبقتين */
      ctx.fillStyle = '#8d99b5';
      ctx.fillRect(p, p + w * 0.42, w, w * 0.16);
      ctx.strokeStyle = '#3a4664'; ctx.lineWidth = s * 0.03;
      ctx.strokeRect(p, p + w * 0.42, w, w * 0.16);
      ctx.beginPath(); ctx.arc(s / 2, p + w * 0.5, s * 0.05, 0, 7);
      ctx.fillStyle = '#3a4664'; ctx.fill();
    } else {
      /* شقوق تدلّ على الضرر */
      ctx.strokeStyle = 'rgba(60,40,15,.7)';
      ctx.lineWidth = s * 0.03;
      ctx.beginPath();
      ctx.moveTo(s * 0.35, s * 0.2); ctx.lineTo(s * 0.45, s * 0.42);
      ctx.lineTo(s * 0.32, s * 0.55); ctx.lineTo(s * 0.44, s * 0.78);
      ctx.stroke();
    }
  }

  function drawIce(ctx, s, layers) {
    const p = s * 0.03, w = s - p * 2;
    ctx.save();
    ctx.globalAlpha = layers > 1 ? 0.92 : 0.72;
    rr(ctx, p, p, w, w, s * 0.1);
    const g = ctx.createLinearGradient(p, p, p + w, p + w);
    g.addColorStop(0, 'rgba(226,246,255,.95)');
    g.addColorStop(0.5, 'rgba(150,215,255,.75)');
    g.addColorStop(1, 'rgba(206,240,255,.95)');
    ctx.fillStyle = g; ctx.fill();
    stroked(ctx, 'rgba(255,255,255,.95)', s * 0.05);
    /* أوجه بلّورية */
    ctx.save(); ctx.clip();
    ctx.strokeStyle = 'rgba(255,255,255,.85)';
    ctx.lineWidth = s * 0.03;
    ctx.beginPath();
    ctx.moveTo(p, p + w * .35); ctx.lineTo(p + w * .4, p + w * .1);
    ctx.moveTo(p + w * .55, p + w); ctx.lineTo(p + w, p + w * .55);
    ctx.moveTo(p + w * .15, p + w * .9); ctx.lineTo(p + w * .5, p + w * .5);
    ctx.stroke();
    ctx.restore();
    if (layers > 1) {
      ctx.strokeStyle = 'rgba(120,190,235,.9)';
      ctx.lineWidth = s * 0.055;
      rr(ctx, p + s * 0.07, p + s * 0.07, w - s * 0.14, w - s * 0.14, s * 0.08);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawChain(ctx, s, layers) {
    const n = 4, r = s * 0.085;
    ctx.save();
    ctx.strokeStyle = '#c3cbe0';
    ctx.lineWidth = s * 0.055;
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) / n;
      ctx.beginPath();
      ctx.ellipse(s * t, s * 0.5, r * 0.75, r * 1.25, 0, 0, 7);
      ctx.stroke();
      ctx.strokeStyle = i % 2 ? '#c3cbe0' : '#8d99b5';
    }
    ctx.strokeStyle = '#3a4664';
    ctx.lineWidth = s * 0.02;
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) / n;
      ctx.beginPath();
      ctx.ellipse(s * t, s * 0.5, r * 0.75, r * 1.25, 0, 0, 7);
      ctx.stroke();
    }
    if (layers > 1) {
      ctx.save();
      ctx.translate(s / 2, s / 2); ctx.rotate(Math.PI / 2); ctx.translate(-s / 2, -s / 2);
      ctx.strokeStyle = '#c3cbe0'; ctx.lineWidth = s * 0.055;
      for (let i = 0; i < n; i++) {
        const t = (i + 0.5) / n;
        ctx.beginPath();
        ctx.ellipse(s * t, s * 0.5, r * 0.75, r * 1.25, 0, 0, 7);
        ctx.stroke();
      }
      ctx.restore();
    }
    ctx.restore();
  }

  function drawGrass(ctx, s, layers) {
    const p = s * 0.02, w = s - p * 2;
    rr(ctx, p, p, w, w, s * 0.1);
    ctx.fillStyle = layers > 1
      ? grad(ctx, p, p, p, p + w, '#2f8f3c', '#155a1f')
      : grad(ctx, p, p, p, p + w, '#5fd36f', '#2f9c41');
    ctx.fill();
    stroked(ctx, '#123f19', s * 0.04);
    ctx.save(); ctx.clip();
    ctx.strokeStyle = 'rgba(255,255,255,.35)';
    ctx.lineWidth = s * 0.035;
    for (let i = 0; i < 5; i++) {
      const x = p + w * (0.12 + i * 0.19);
      ctx.beginPath();
      ctx.moveTo(x, p + w * 0.85);
      ctx.quadraticCurveTo(x + s * 0.04, p + w * 0.5, x + s * 0.01, p + w * 0.2);
      ctx.stroke();
    }
    ctx.restore();
  }

  /* الغرض الملكي الذي يجب إنزاله إلى الأسفل */
  function drawItem(ctx, s) {
    const w = s * 0.74, h = s * 0.58, x = (s - w) / 2, y = (s - h) / 2 + s * 0.03;
    /* الصندوق */
    rr(ctx, x, y, w, h, s * 0.07);
    ctx.fillStyle = grad(ctx, x, y, x, y + h, '#c9955a', '#7d5a2e');
    ctx.fill();
    stroked(ctx, '#4a3216', s * 0.055);
    /* الغطاء */
    ctx.beginPath();
    ctx.moveTo(x, y + h * 0.36);
    ctx.quadraticCurveTo(x + w * 0.5, y - h * 0.3, x + w, y + h * 0.36);
    ctx.closePath();
    ctx.fillStyle = grad(ctx, x, y - h * 0.2, x, y + h * 0.36, '#e0b277', '#a3743c');
    ctx.fill();
    stroked(ctx, '#4a3216', s * 0.055);
    /* أحزمة ذهبية */
    ctx.fillStyle = '#ffcc4d';
    ctx.fillRect(x + w * 0.42, y - h * 0.12, w * 0.16, h * 0.95);
    ctx.strokeStyle = '#c97f05'; ctx.lineWidth = s * 0.025;
    ctx.strokeRect(x + w * 0.42, y - h * 0.12, w * 0.16, h * 0.95);
    /* القفل */
    ctx.beginPath(); ctx.arc(x + w * 0.5, y + h * 0.42, s * 0.06, 0, 7);
    ctx.fillStyle = '#ffe08a'; ctx.fill();
    stroked(ctx, '#c97f05', s * 0.03);
  }

  /* بلاطة الخلفية */
  function drawTile(ctx, s, alt) {
    const p = s * 0.02, w = s - p * 2;
    rr(ctx, p, p, w, w, s * 0.14);
    ctx.fillStyle = alt ? 'rgba(255,255,255,.10)' : 'rgba(255,255,255,.055)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,.09)';
    ctx.lineWidth = s * 0.02;
    ctx.stroke();
  }

  /* فتحة التجميع أسفل العمود */
  function drawCollector(ctx, s) {
    const p = s * 0.06, w = s - p * 2;
    rr(ctx, p, p, w, w, s * 0.14);
    ctx.fillStyle = 'rgba(65,193,83,.2)';
    ctx.fill();
    ctx.strokeStyle = '#41c153'; ctx.lineWidth = s * 0.05;
    ctx.setLineDash([s * 0.1, s * 0.07]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(s * 0.32, s * 0.38); ctx.lineTo(s * 0.5, s * 0.62); ctx.lineTo(s * 0.68, s * 0.38);
    ctx.strokeStyle = '#7dff9a'; ctx.lineWidth = s * 0.08;
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.stroke();
  }

  /* ======================= الواجهة العامة ======================= */

  /**
   * يُرجع canvas مرسوماً مسبقاً لأي عنصر.
   * الأسماء: p0..p5 | rh | rv | tnt | ball | box1 | box2 | ice1 | ice2 |
   *          chain1 | chain2 | grass1 | grass2 | item | tile | tileAlt | collector
   */
  function sprite(name, size) {
    size = Math.round(size);
    const key = name + '@' + size;
    if (cache.has(key)) return cache.get(key);

    const { c, ctx } = surface(size);
    const s = size;

    if (/^p[0-5]$/.test(name)) {
      const i = +name[1];
      DRAWERS[i](ctx, s, PIECE_TYPES[i]);
    } else if (/^p[0-5](rh|rv)$/.test(name)) {
      const i = +name[1];
      DRAWERS[i](ctx, s, PIECE_TYPES[i]);
    } else {
      switch (name) {
        case 'rh': drawRocket(ctx, s, false); break;
        case 'rv': drawRocket(ctx, s, true); break;
        case 'tnt': drawTNT(ctx, s); break;
        case 'ball': drawBall(ctx, s); break;
        case 'box1': drawBox(ctx, s, 1); break;
        case 'box2': drawBox(ctx, s, 2); break;
        case 'ice1': drawIce(ctx, s, 1); break;
        case 'ice2': drawIce(ctx, s, 2); break;
        case 'chain1': drawChain(ctx, s, 1); break;
        case 'chain2': drawChain(ctx, s, 2); break;
        case 'grass1': drawGrass(ctx, s, 1); break;
        case 'grass2': drawGrass(ctx, s, 2); break;
        case 'item': drawItem(ctx, s); break;
        case 'tile': drawTile(ctx, s, false); break;
        case 'tileAlt': drawTile(ctx, s, true); break;
        case 'collector': drawCollector(ctx, s); break;
        case 'hammer': drawHammer(ctx, s); break;
        case 'glove': drawGlove(ctx, s); break;
        case 'shuffle': drawShuffle(ctx, s); break;
        case 'moves': drawMovesIcon(ctx, s); break;
        default: break;
      }
    }
    cache.set(key, c);
    return c;
  }

  /* ------------------ أيقونات المعزّزات ------------------ */
  function drawHammer(ctx, s) {
    ctx.save();
    ctx.translate(s / 2, s / 2); ctx.rotate(-0.5); ctx.translate(-s / 2, -s / 2);
    rr(ctx, s * 0.44, s * 0.3, s * 0.12, s * 0.58, s * 0.04);
    ctx.fillStyle = grad(ctx, 0, s * 0.3, 0, s * 0.88, '#c9955a', '#7d5a2e');
    ctx.fill(); stroked(ctx, '#4a3216', s * 0.05);
    rr(ctx, s * 0.2, s * 0.15, s * 0.6, s * 0.22, s * 0.06);
    ctx.fillStyle = grad(ctx, 0, s * 0.15, 0, s * 0.37, '#d7deeb', '#7e89a8');
    ctx.fill(); stroked(ctx, '#3a4664', s * 0.05);
    ctx.restore();
  }
  function drawGlove(ctx, s) {
    rr(ctx, s * 0.26, s * 0.32, s * 0.48, s * 0.46, s * 0.12);
    ctx.fillStyle = grad(ctx, 0, s * 0.32, 0, s * 0.78, '#ffe9a8', '#f0a71b');
    ctx.fill(); stroked(ctx, '#8c5c04', s * 0.05);
    for (let i = 0; i < 3; i++) {
      rr(ctx, s * (0.3 + i * 0.15), s * 0.16, s * 0.12, s * 0.22, s * 0.05);
      ctx.fillStyle = '#ffd76a'; ctx.fill(); stroked(ctx, '#8c5c04', s * 0.04);
    }
    rr(ctx, s * 0.14, s * 0.42, s * 0.16, s * 0.2, s * 0.06);
    ctx.fillStyle = '#ffd76a'; ctx.fill(); stroked(ctx, '#8c5c04', s * 0.04);
  }
  function drawShuffle(ctx, s) {
    ctx.strokeStyle = '#7fd8ff'; ctx.lineWidth = s * 0.09;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.arc(s / 2, s / 2, s * 0.28, 0.6, 5.0);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(s * 0.74, s * 0.24); ctx.lineTo(s * 0.86, s * 0.3); ctx.lineTo(s * 0.76, s * 0.4);
    ctx.fillStyle = '#7fd8ff'; ctx.fill();
    ctx.fillStyle = '#ffe08a';
    ctx.beginPath(); ctx.arc(s / 2, s / 2, s * 0.1, 0, 7); ctx.fill();
  }
  function drawMovesIcon(ctx, s) {
    ctx.fillStyle = '#7fd8ff';
    ctx.font = 'bold ' + s * 0.46 + 'px Arial';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('+5', s / 2, s * 0.54);
    ctx.strokeStyle = '#7fd8ff'; ctx.lineWidth = s * 0.07;
    ctx.beginPath(); ctx.arc(s / 2, s / 2, s * 0.38, 0, 7); ctx.stroke();
  }

  /** رسم قطعة كاملة (لون + خاصّية) في مكان محدّد */
  function drawPiece(ctx, piece, x, y, size) {
    if (!piece) return;
    if (piece.kind === 'item') {
      ctx.drawImage(sprite('item', size), x, y, size, size);
      return;
    }
    switch (piece.special) {
      case SPECIAL.ROCKET_H: ctx.drawImage(sprite('rh', size), x, y, size, size); return;
      case SPECIAL.ROCKET_V: ctx.drawImage(sprite('rv', size), x, y, size, size); return;
      case SPECIAL.TNT:      ctx.drawImage(sprite('tnt', size), x, y, size, size); return;
      case SPECIAL.BALL:     ctx.drawImage(sprite('ball', size), x, y, size, size); return;
      default:
        ctx.drawImage(sprite('p' + piece.color, size), x, y, size, size);
    }
  }

  /** ينشئ عنصر canvas مستقلاً لاستعماله في الواجهة (الأهداف، المعزّزات…) */
  function iconEl(name, size) {
    const src = sprite(name, size);
    const out = document.createElement('canvas');
    out.width = src.width; out.height = src.height;
    out.style.width = size + 'px'; out.style.height = size + 'px';
    out.getContext('2d').drawImage(src, 0, 0);
    return out;
  }

  return { sprite, drawPiece, iconEl, cache };
})();

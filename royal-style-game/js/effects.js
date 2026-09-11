/* ==========================================================================
   effects.js — الجسيمات والمؤثّرات البصرية فوق اللوحة
   ========================================================================== */
'use strict';

const FX = (function () {
  let parts = [];
  let texts = [];
  let rings = [];
  let beams = [];
  let trails = [];

  function reset() { parts = []; texts = []; rings = []; beams = []; trails = []; }

  /** انفجار جسيمات ملوّنة */
  function burst(x, y, color, n, power) {
    n = n || 10;
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2;
      const sp = (0.5 + Math.random()) * (power || 1) * 170;
      parts.push({
        x, y,
        vx: Math.cos(a) * sp,
        vy: Math.sin(a) * sp - 60,
        r: 2 + Math.random() * 4,
        life: 0.45 + Math.random() * 0.35,
        age: 0,
        color,
        shape: Math.random() < 0.35 ? 'star' : 'dot',
        spin: (Math.random() - 0.5) * 12,
        rot: Math.random() * 6,
      });
    }
  }

  /** شظايا خشب/جليد */
  function shards(x, y, color, n) {
    for (let i = 0; i < (n || 8); i++) {
      const a = Math.random() * Math.PI * 2;
      const sp = (0.4 + Math.random()) * 200;
      parts.push({
        x, y,
        vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 90,
        r: 3 + Math.random() * 5,
        life: 0.5 + Math.random() * 0.3, age: 0,
        color, shape: 'shard',
        spin: (Math.random() - 0.5) * 16, rot: Math.random() * 6,
      });
    }
  }

  /** حلقة صدمة */
  function ring(x, y, maxR, color) {
    rings.push({ x, y, r: maxR * 0.18, maxR, life: 0.42, age: 0, color: color || '#ffe08a' });
  }

  /** شعاع من الكرة المضيئة إلى الهدف */
  function beam(x1, y1, x2, y2, color) {
    beams.push({ x1, y1, x2, y2, life: 0.3, age: 0, color: color || '#ffffff' });
  }

  /** أثر الصاروخ */
  function trail(x, y, dirX, dirY, len, color) {
    trails.push({ x, y, dx: dirX, dy: dirY, len, life: 0.3, age: 0, color: color || '#ffffff' });
  }

  /** نصّ طائر (نقاط أو رسالة) */
  function text(x, y, str, color, size) {
    texts.push({ x, y, str, color: color || '#fff', size: size || 18, life: 0.9, age: 0 });
  }

  function update(dt) {
    for (let i = parts.length - 1; i >= 0; i--) {
      const p = parts[i];
      p.age += dt;
      if (p.age >= p.life) { parts.splice(i, 1); continue; }
      p.vy += 900 * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.rot += p.spin * dt;
    }
    for (let i = texts.length - 1; i >= 0; i--) {
      const t = texts[i];
      t.age += dt; t.y -= 46 * dt;
      if (t.age >= t.life) texts.splice(i, 1);
    }
    for (let i = rings.length - 1; i >= 0; i--) {
      const r = rings[i];
      r.age += dt;
      if (r.age >= r.life) rings.splice(i, 1);
    }
    for (let i = beams.length - 1; i >= 0; i--) {
      const b = beams[i];
      b.age += dt;
      if (b.age >= b.life) beams.splice(i, 1);
    }
    for (let i = trails.length - 1; i >= 0; i--) {
      const t = trails[i];
      t.age += dt;
      if (t.age >= t.life) trails.splice(i, 1);
    }
  }

  function draw(ctx) {
    /* الأشعة */
    for (const b of beams) {
      const k = 1 - b.age / b.life;
      ctx.save();
      ctx.globalAlpha = k;
      ctx.strokeStyle = b.color;
      ctx.lineWidth = 5 * k + 1;
      ctx.shadowColor = b.color; ctx.shadowBlur = 14;
      ctx.beginPath(); ctx.moveTo(b.x1, b.y1); ctx.lineTo(b.x2, b.y2); ctx.stroke();
      ctx.restore();
    }
    /* آثار الصواريخ */
    for (const t of trails) {
      const k = 1 - t.age / t.life;
      ctx.save();
      ctx.globalAlpha = k * 0.85;
      const g = ctx.createLinearGradient(t.x, t.y, t.x + t.dx * t.len, t.y + t.dy * t.len);
      g.addColorStop(0, 'rgba(255,255,255,.95)');
      g.addColorStop(0.5, 'rgba(255,200,90,.6)');
      g.addColorStop(1, 'rgba(255,120,40,0)');
      ctx.strokeStyle = g;
      ctx.lineWidth = 16 * k;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(t.x, t.y);
      ctx.lineTo(t.x + t.dx * t.len, t.y + t.dy * t.len);
      ctx.stroke();
      ctx.restore();
    }
    /* حلقات الصدمة */
    for (const r of rings) {
      const k = r.age / r.life;
      ctx.save();
      ctx.globalAlpha = (1 - k) * 0.9;
      ctx.strokeStyle = r.color;
      ctx.lineWidth = 9 * (1 - k) + 2;
      ctx.beginPath();
      ctx.arc(r.x, r.y, U.lerp(r.r, r.maxR, U.easeOutCubic(k)), 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    /* الجسيمات */
    for (const p of parts) {
      const k = 1 - p.age / p.life;
      ctx.save();
      ctx.globalAlpha = Math.min(1, k * 1.6);
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.fillStyle = p.color;
      if (p.shape === 'dot') {
        ctx.beginPath(); ctx.arc(0, 0, p.r * k + 0.6, 0, 7); ctx.fill();
      } else if (p.shape === 'shard') {
        ctx.fillRect(-p.r * 0.6, -p.r * 0.6, p.r * 1.2, p.r * 1.2);
      } else {
        const R = p.r * 1.6 * k + 1;
        ctx.beginPath();
        for (let i = 0; i < 5; i++) {
          const a = -Math.PI / 2 + i * Math.PI * 2 / 5;
          const a2 = a + Math.PI / 5;
          ctx.lineTo(Math.cos(a) * R, Math.sin(a) * R);
          ctx.lineTo(Math.cos(a2) * R * 0.45, Math.sin(a2) * R * 0.45);
        }
        ctx.closePath(); ctx.fill();
      }
      ctx.restore();
    }
    /* النصوص الطائرة */
    for (const t of texts) {
      const k = 1 - t.age / t.life;
      ctx.save();
      ctx.globalAlpha = Math.min(1, k * 1.8);
      ctx.font = '900 ' + t.size + 'px "Tajawal",Arial,sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(0,0,0,.75)';
      ctx.strokeText(t.str, t.x, t.y);
      ctx.fillStyle = t.color;
      ctx.fillText(t.str, t.x, t.y);
      ctx.restore();
    }
  }

  function busy() { return parts.length + rings.length + beams.length > 0; }

  return { reset, burst, shards, ring, beam, trail, text, update, draw, busy };
})();

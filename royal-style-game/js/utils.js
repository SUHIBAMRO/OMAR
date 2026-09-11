/* ==========================================================================
   utils.js — دوال مساعدة عامة
   ========================================================================== */
'use strict';

const U = {
  /** عدد صحيح عشوائي في [a,b) */
  randInt(a, b) { return a + Math.floor(Math.random() * (b - a)); },

  /** عنصر عشوائي من مصفوفة */
  pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; },

  /** حصر قيمة بين حدّين */
  clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); },

  /** استيفاء خطّي */
  lerp(a, b, t) { return a + (b - a) * t; },

  /** منحنى تسارع/تباطؤ ناعم */
  easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); },
  easeInCubic(t) { return t * t * t; },
  easeOutBack(t) {
    const c1 = 1.70158, c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
  easeInOutQuad(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; },

  /** خلط مصفوفة في مكانها (فيشر-ييتس) */
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  },

  /** تنسيق عدد بفواصل */
  fmt(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ','); },

  /** تحويل ثوانٍ إلى mm:ss */
  mmss(sec) {
    sec = Math.max(0, Math.ceil(sec));
    const m = Math.floor(sec / 60), s = sec % 60;
    return m + ':' + String(s).padStart(2, '0');
  },

  /** مولّد أرقام شبه عشوائي ببذرة — لتوليد مراحل ثابتة لكل رقم مرحلة */
  rng(seed) {
    let s = seed >>> 0 || 1;
    return function () {
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5;  s >>>= 0;
      return s / 4294967296;
    };
  },

  /** إنشاء عنصر DOM بسرعة */
  el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  },

  $(sel) { return document.querySelector(sel); },
  $$(sel) { return Array.prototype.slice.call(document.querySelectorAll(sel)); },
};

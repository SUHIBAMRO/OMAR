/* ==========================================================================
   storage.js — الحفظ المحلي: التقدّم، العملات، القلوب، الإعدادات
   كل شيء داخل مفتاح واحد في localStorage (مع تحمّل الفشل بصمت).
   ========================================================================== */
'use strict';

const Save = (function () {
  const KEY = 'royal_castle_save_v1';

  const DEFAULTS = {
    version: 1,
    level: 1,                 // أعلى مرحلة مفتوحة
    stars: {},                // { "3": 2 }  نجوم كل مرحلة
    best:  {},                // { "3": 8400 } أفضل نتيجة
    coins: CFG.economy.startCoins,
    lives: CFG.lives.max,
    livesAt: 0,               // وقت آخر استهلاك قلب (ms)
    boosters: { hammer: 3, glove: 2, rocket: 2, tnt: 1, ball: 1, shuffle: 2 },
    sound: CFG.soundDefault,
    music: CFG.musicDefault,
    hints: CFG.hintEnabled,
    seenRules: false,
    totalWins: 0,
  };

  let data = null;

  function load() {
    if (data) return data;
    let raw = null;
    try { raw = localStorage.getItem(KEY); } catch (e) { raw = null; }
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        data = Object.assign({}, DEFAULTS, parsed);
        data.boosters = Object.assign({}, DEFAULTS.boosters, parsed.boosters || {});
        data.stars = parsed.stars || {};
        data.best = parsed.best || {};
      } catch (e) { data = Object.assign({}, DEFAULTS); }
    } else {
      data = Object.assign({}, DEFAULTS);
      data.livesAt = Date.now();
    }
    regenLives();
    return data;
  }

  function flush() {
    try { localStorage.setItem(KEY, JSON.stringify(data)); } catch (e) { /* وضع التصفح الخاص */ }
  }

  /* ------------------------- القلوب ------------------------- */
  function regenLives() {
    if (data.lives >= CFG.lives.max) { data.livesAt = Date.now(); return; }
    const elapsed = Date.now() - (data.livesAt || Date.now());
    if (elapsed <= 0) return;
    const gained = Math.floor(elapsed / CFG.lives.refillMs);
    if (gained > 0) {
      data.lives = Math.min(CFG.lives.max, data.lives + gained);
      data.livesAt = data.lives >= CFG.lives.max
        ? Date.now()
        : (data.livesAt + gained * CFG.lives.refillMs);
      flush();
    }
  }

  /** الثواني المتبقية حتى القلب التالي (0 إن كانت ممتلئة) */
  function nextLifeIn() {
    regenLives();
    if (data.lives >= CFG.lives.max) return 0;
    const elapsed = Date.now() - data.livesAt;
    return Math.max(0, (CFG.lives.refillMs - elapsed) / 1000);
  }

  const API = {
    get data() { return load(); },

    init() { load(); return API; },
    save() { flush(); },

    /* -------- عملات -------- */
    coins() { return load().coins; },
    addCoins(n) { load().coins = Math.max(0, data.coins + n); flush(); return data.coins; },
    spend(n) {
      load();
      if (data.coins < n) return false;
      data.coins -= n; flush(); return true;
    },

    /* -------- قلوب -------- */
    lives() { load(); regenLives(); return data.lives; },
    nextLifeIn,
    useLife() {
      load(); regenLives();
      if (data.lives <= 0) return false;
      if (data.lives === CFG.lives.max) data.livesAt = Date.now();
      data.lives--; flush(); return true;
    },
    addLife(n) {
      load(); regenLives();
      data.lives = Math.min(CFG.lives.max, data.lives + (n || 1));
      if (data.lives >= CFG.lives.max) data.livesAt = Date.now();
      flush();
    },
    fillLives() { load(); data.lives = CFG.lives.max; data.livesAt = Date.now(); flush(); },

    /* -------- تقدّم -------- */
    maxLevel() { return load().level; },
    starsFor(lv) { return load().stars[lv] || 0; },
    bestFor(lv) { return load().best[lv] || 0; },
    totalStars() {
      load();
      let t = 0;
      for (const k in data.stars) t += data.stars[k];
      return t;
    },
    /** يسجّل فوزاً ويُرجع عدد النجوم الجديدة المكتسبة */
    recordWin(lv, stars, score) {
      load();
      const prev = data.stars[lv] || 0;
      const gained = Math.max(0, stars - prev);
      if (stars > prev) data.stars[lv] = stars;
      if (score > (data.best[lv] || 0)) data.best[lv] = score;
      if (lv >= data.level) data.level = lv + 1;
      data.totalWins++;
      flush();
      return gained;
    },

    /* -------- معزّزات -------- */
    booster(k) { return load().boosters[k] || 0; },
    addBooster(k, n) { load(); data.boosters[k] = (data.boosters[k] || 0) + n; flush(); },
    useBooster(k) {
      load();
      if ((data.boosters[k] || 0) <= 0) return false;
      data.boosters[k]--; flush(); return true;
    },

    /* -------- إعدادات -------- */
    setFlag(k, v) { load(); data[k] = v; flush(); },
    flag(k) { return load()[k]; },

    reset() {
      data = Object.assign({}, DEFAULTS);
      data.stars = {}; data.best = {};
      data.boosters = Object.assign({}, DEFAULTS.boosters);
      data.livesAt = Date.now();
      flush();
    },
  };

  return API;
})();

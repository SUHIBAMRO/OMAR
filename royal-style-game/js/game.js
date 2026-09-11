/* ==========================================================================
   game.js — حلقة اللعب: العرض، الإدخال، وآلة حالات الدور
   ---------------------------------------------------------------------------
   دورة الدور الواحد:
     idle → swap → (clear ⇄ fall)* → idle
   وعند انتهاء الأهداف: bonus (تحويل الحركات المتبقّية إلى صواريخ) → blast → win
   ========================================================================== */
'use strict';

const Game = (function () {

  const S = {
    canvas: null, ctx: null, wrap: null,
    board: null, def: null, levelNo: 1,

    tile: 60, pad: 10, ox: 0, oy: 0, w: 0, h: 0,

    state: 'idle',
    t: 0,                 // مؤقّت الحالة (ms)
    clearDur: 0,
    fallTimer: 0,
    bonusTimer: 0,

    moves: 0,
    timed: false,
    timeLeft: 0,
    bombBlown: null,
    bonusLeft: 0,
    cascade: 0,
    ending: false,
    endStage: '',
    running: false,
    lastTs: 0,

    sel: null,            // [r,c] المحدّدة
    drag: null,
    swapAnim: null,       // {a:{r,c}, b:{r,c}, t, dur, back}
    activeBooster: null,
    gloveFirst: null,

    idleTime: 0,
    hint: null,
    shakeT: 0,

    goalCache: [],
    lastScore: 0,
    sfxBudget: {},
  };

  /* ------------------------------------------------------------------ */
  /*                            بدء المرحلة                              */
  /* ------------------------------------------------------------------ */

  function start(levelNo, pre) {
    S.levelNo = levelNo;
    S.def = Levels.get(levelNo);
    S.board = new Board(S.def);
    S.timed = !!S.def.timeLimit;
    S.timeLeft = S.def.timeLimit || 0;
    S.bombBlown = null;
    S.moves = S.timed ? 0 : S.def.moves;
    if (pre && pre.moves) {
      if (S.timed) S.timeLeft += 20;                 // معزّز البداية يمنح وقتاً في المراحل المؤقّتة
      else S.moves += CFG.economy.extraMovesAmount;
    }
    S.state = 'idle';
    S.t = 0; S.cascade = 0; S.ending = false; S.endStage = '';
    S.sel = null; S.drag = null; S.swapAnim = null;
    S.activeBooster = null; S.gloveFirst = null;
    S.idleTime = 0; S.hint = null; S.shakeT = 0;
    S.lastScore = 0;
    FX.reset();

    /* معزّزات ما قبل البدء: تُزرع على اللوحة مباشرة */
    if (pre) {
      const put = (sp, n) => {
        for (let i = 0; i < n; i++) {
          const spots = [];
          for (let r = 0; r < S.board.rows; r++) for (let c = 0; c < S.board.cols; c++) {
            if (S.board.isFree(r, c)) {
              const p = S.board.grid[r][c].piece;
              if (p.kind === 'normal' && p.special === SPECIAL.NONE) spots.push([r, c]);
            }
          }
          if (!spots.length) return;
          const [r, c] = U.pick(spots);
          S.board.grid[r][c].piece.special = sp;
        }
      };
      if (pre.rocket) put(Math.random() < 0.5 ? SPECIAL.ROCKET_H : SPECIAL.ROCKET_V, 1);
      if (pre.tnt) put(SPECIAL.TNT, 1);
      if (pre.ball) put(SPECIAL.BALL, 1);
    }

    S.canvas = U.$('#board-canvas');
    S.ctx = S.canvas.getContext('2d');
    S.wrap = U.$('#board-wrap');
    bindInput();
    resize();
    UI.buildGoals(S.def, S.board);
    UI.buildBoosterBar();
    const lbl = U.$('#level-label');
    if (lbl) lbl.textContent = 'المرحلة ' + levelNo;
    updateHud(true);

    if (!S.running) { S.running = true; S.lastTs = performance.now(); requestAnimationFrame(loop); }
  }

  function stop() { S.running = false; }

  /* ------------------------------------------------------------------ */
  /*                             القياسات                                */
  /* ------------------------------------------------------------------ */

  function resize() {
    if (!S.canvas || !S.board) return;
    const wrap = S.wrap.getBoundingClientRect();
    const chrome = 96;                     // الأعلام + شريط اسم المرحلة + القاعدة
    const availW = Math.max(200, wrap.width - 6);
    const availH = Math.max(200, wrap.height - chrome);
    const pad = 10;
    const tile = Math.floor(Math.min(
      (availW - pad * 2) / S.board.cols,
      (availH - pad * 2) / S.board.rows
    ));
    S.tile = Math.max(26, tile);
    S.pad = pad;
    S.w = S.board.cols * S.tile + pad * 2;
    S.h = S.board.rows * S.tile + pad * 2;
    S.ox = pad; S.oy = pad;

    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    S.canvas.width = Math.round(S.w * dpr);
    S.canvas.height = Math.round(S.h * dpr);
    S.canvas.style.width = S.w + 'px';
    S.canvas.style.height = S.h + 'px';
    S.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  const cx = c => S.ox + c * S.tile;
  const cy = r => S.oy + r * S.tile;
  const ccx = c => S.ox + (c + 0.5) * S.tile;
  const ccy = r => S.oy + (r + 0.5) * S.tile;

  /* ------------------------------------------------------------------ */
  /*                              الإدخال                                */
  /* ------------------------------------------------------------------ */

  let inputBound = false;
  function bindInput() {
    if (inputBound) return;
    inputBound = true;
    const cv = S.canvas;
    cv.addEventListener('pointerdown', onDown);
    cv.addEventListener('pointermove', onMove);
    cv.addEventListener('pointerup', onUp);
    cv.addEventListener('pointercancel', onUp);
    cv.addEventListener('contextmenu', e => e.preventDefault());
  }

  function cellFromEvent(e) {
    const rect = S.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const c = Math.floor((x - S.ox) / S.tile);
    const r = Math.floor((y - S.oy) / S.tile);
    if (!S.board.inside(r, c)) return null;
    return { r, c, x, y };
  }

  function onDown(e) {
    if (S.state !== 'idle' || S.ending) return;
    const p = cellFromEvent(e);
    if (!p) return;
    if (S.canvas.setPointerCapture) { try { S.canvas.setPointerCapture(e.pointerId); } catch (err) {} }
    S.idleTime = 0; S.hint = null;
    Sfx.resume();

    if (S.activeBooster) { applyBooster(p.r, p.c); return; }
    if (!S.board.isFree(p.r, p.c)) { Sfx.invalid(); return; }

    /* التحديد نفسه لا يتغيّر هنا — يُحسم عند رفع الإصبع،
       حتى تعمل طريقتا اللعب: السحب، والنقر ثم النقر على المجاور. */
    S.drag = { from: p, moved: false };
  }

  function onMove(e) {
    if (!S.drag || S.state !== 'idle') return;
    const rect = S.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const dx = x - ccx(S.drag.from.c);
    const dy = y - ccy(S.drag.from.r);
    const th = S.tile * 0.42;
    if (Math.abs(dx) < th && Math.abs(dy) < th) return;

    let r2 = S.drag.from.r, c2 = S.drag.from.c;
    if (Math.abs(dx) > Math.abs(dy)) c2 += dx > 0 ? 1 : -1;
    else r2 += dy > 0 ? 1 : -1;

    const from = S.drag.from;
    S.drag = null;
    S.sel = null;
    tryMove(from.r, from.c, r2, c2);
  }

  function onUp(e) {
    if (!S.drag) return;
    const from = S.drag.from;
    S.drag = null;
    const p = cellFromEvent(e);
    /* رفع الإصبع خارج الخليّة التي بدأ منها يُعامَل كنقرة على خليّة البداية */
    const cell = p && S.board.isFree(p.r, p.c) ? p : from;

    if (S.sel && (S.sel[0] !== cell.r || S.sel[1] !== cell.c)) {
      if (Math.abs(S.sel[0] - cell.r) + Math.abs(S.sel[1] - cell.c) === 1) {
        const a = S.sel;
        S.sel = null;
        tryMove(a[0], a[1], cell.r, cell.c);
      } else {
        S.sel = [cell.r, cell.c];
        Sfx.click();
      }
      return;
    }

    if (S.sel) {
      /* النقر على قطعة خاصّة محدّدة مسبقاً يفعّلها في مكانها */
      const piece = S.board.grid[cell.r][cell.c].piece;
      S.sel = null;
      if (piece && piece.special) {
        consumeMove();
        S.board.detonateAt(cell.r, cell.c);
        startClear();
      }
      return;
    }

    S.sel = [cell.r, cell.c];
    Sfx.click();
  }

  /** محاولة تبديل قطعتين */
  function tryMove(r1, c1, r2, c2, free) {
    if (!S.board.canSwap(r1, c1, r2, c2)) { Sfx.invalid(); shake(); return false; }
    const useful = free || S.board.swapIsUseful(r1, c1, r2, c2);
    S.sel = null;
    S.swapAnim = {
      a: { r: r1, c: c1 }, b: { r: r2, c: c2 },
      t: 0, dur: CFG.time.swap, back: false, free: !!free, ok: useful,
    };
    S.state = 'swap';
    Sfx.swap();
    return true;
  }

  function shake() { S.shakeT = 260; }

  /* ------------------------------------------------------------------ */
  /*                             المعزّزات                               */
  /* ------------------------------------------------------------------ */

  function armBooster(key) {
    if (S.state !== 'idle' || S.ending) return;
    if (S.activeBooster === key) { S.activeBooster = null; S.gloveFirst = null; UI.buildBoosterBar(); return; }
    if (Save.booster(key) <= 0) { UI.showBuyBooster(key); return; }
    if (key === 'shuffle') {
      Save.useBooster(key);
      S.board.shuffle();
      Sfx.shuffle();
      UI.banner('خَلْط!');
      UI.buildBoosterBar();
      return;
    }
    S.activeBooster = key;
    S.gloveFirst = null;
    S.sel = null;
    UI.buildBoosterBar();
    UI.banner(BOOSTER_INFO[key].name + ': اختر خانة');
  }

  function applyBooster(r, c) {
    const key = S.activeBooster;
    const cell = S.board.at(r, c);
    if (!cell || !cell.exists) return;

    if (key === 'glove') {
      if (!S.board.isFree(r, c)) { Sfx.invalid(); return; }
      if (!S.gloveFirst) { S.gloveFirst = [r, c]; S.sel = [r, c]; Sfx.click(); return; }
      const [r1, c1] = S.gloveFirst;
      if (Math.abs(r1 - r) + Math.abs(c1 - c) !== 1) { S.gloveFirst = [r, c]; S.sel = [r, c]; return; }
      if (!Save.useBooster('glove')) return;
      S.activeBooster = null; S.gloveFirst = null;
      UI.buildBoosterBar();
      tryMove(r1, c1, r, c, true);
      return;
    }

    if (key === 'hammer') {
      if (cell.box === 0 && cell.ice === 0 && cell.chain === 0 && !cell.piece) { Sfx.invalid(); return; }
      if (!Save.useBooster('hammer')) return;
      S.activeBooster = null; UI.buildBoosterBar();
      S.board.beginPhase();
      S.board.markClear(r, c, 0, { fromMatch: true });
      Sfx.obstacle();
      startClear();
      return;
    }

    /* صاروخ / قنبلة / كرة: تُزرع في الخانة ثم تنفجر */
    if (key === 'rocket' || key === 'tnt' || key === 'ball') {
      if (!S.board.isFree(r, c) || cell.piece.kind !== 'normal') { Sfx.invalid(); return; }
      if (!Save.useBooster(key)) return;
      S.activeBooster = null; UI.buildBoosterBar();
      const p = cell.piece;
      p.special = key === 'rocket'
        ? (Math.random() < 0.5 ? SPECIAL.ROCKET_H : SPECIAL.ROCKET_V)
        : (key === 'tnt' ? SPECIAL.TNT : SPECIAL.BALL);
      S.board.beginPhase();
      S.board.markClear(r, c, 0);
      startClear();
    }
  }

  /* ------------------------------------------------------------------ */
  /*                           آلة الحالات                               */
  /* ------------------------------------------------------------------ */

  function consumeMove() {
    if (!S.timed) {
      S.moves = Math.max(0, S.moves - 1);
      if (S.moves === 3) Sfx.lastMoves();
    }
    /* كل حركة تُنقص فتيل كل قنبلة على اللوحة */
    const blown = S.board.tickBombs();
    if (blown && !S.bombBlown) S.bombBlown = blown;
    else if (S.board.minFuse() <= CFG.bomb.warnAt) Sfx.lastMoves();
    updateHud();
  }

  /** إضافة ثوانٍ إلى مؤقّت المراحل المؤقّتة */
  function addTime(sec) {
    if (!S.timed || sec <= 0) return;
    sec = Math.min(sec, CFG.timed.maxAdd);
    S.timeLeft += sec;
    const el = U.$('#moves-count');
    if (el) FX.text(S.w / 2, S.h * 0.12, '+' + sec.toFixed(1) + ' ث', '#7dff9a', S.tile * 0.42);
  }

  function startClear() {
    S.board.cascade = S.cascade;
    S.clearDur = S.board.maxDelay() + CFG.time.pop + CFG.time.settleGap;
    S.t = 0;
    S.state = 'clear';
    S.sfxBudget = {};
    for (const f of S.board.fx) f.fired = false;
  }

  function afterSettle() {
    /* انفجار قنبلة موقوتة = خسارة فورية */
    if (S.bombBlown && !S.ending) {
      const { r, c } = S.bombBlown;
      S.bombBlown = null;
      FX.ring(ccx(c), ccy(r), S.tile * 4, '#ff6b6b');
      FX.burst(ccx(c), ccy(r), '#ff4d4d', 40, 2.6);
      shake();
      Sfx.tnt();
      UI.banner('انفجرت القنبلة!');
      setTimeout(() => finishLose('bomb'), 900);
      S.state = 'done';
      return;
    }

    /* تسلسل النهاية */
    if (S.ending) {
      if (S.endStage === 'blast') { finishWin(); return; }
      S.state = 'bonus';
      return;
    }

    const more = S.board.applyMatches();
    if (more > 0) {
      S.cascade++;
      if (S.cascade >= 3) UI.banner(comboWord(S.cascade));
      Sfx.match(S.cascade);
      startClear();
      return;
    }

    S.cascade = 0;
    S.board.cascade = 0;

    if (S.board.goalsDone()) {
      S.ending = true;
      S.endStage = 'bonus';
      S.state = 'bonus';
      S.bonusTimer = 0;
      if (S.timed) {
        /* الوقت المتبقّي يتحوّل نقاطاً، ثم ثلاثة صواريخ مكافأة */
        S.board.stats.score += Math.round(S.timeLeft * CFG.timed.leftoverSecScore);
        S.timeLeft = 0;
        S.bonusLeft = 3;
      } else {
        S.bonusLeft = S.moves;
      }
      UI.banner('أحسنت!');
      Sfx.win();
      return;
    }
    if (S.timed) { if (S.timeLeft <= 0) { finishLose('time'); return; } }
    else if (S.moves <= 0) { finishLose(); return; }
    if (!S.board.hasMoves()) {
      UI.banner('لا توجد حركات — خَلْط!');
      Sfx.shuffle();
      S.board.shuffle();
      S.state = 'idle';
      return;
    }
    S.state = 'idle';
    S.idleTime = 0;
  }

  function comboWord(n) {
    if (n >= 6) return 'لا يُصدَّق!';
    if (n >= 5) return 'مذهل!';
    if (n >= 4) return 'رائع!';
    return 'جميل!';
  }

  function finishWin() {
    S.state = 'done';
    const stars = S.board.stars();
    const score = S.board.stats.score;
    const gainedStars = Save.recordWin(S.levelNo, stars, score);
    let coins = CFG.economy.winCoins + gainedStars * CFG.economy.starCoins;
    if (Save.starsFor(S.levelNo) === stars && gainedStars === stars) coins += CFG.economy.firstWinBonus;
    Save.addCoins(coins);
    Sfx.win();
    UI.showWin({ level: S.levelNo, stars, score, coins });
  }

  function finishLose(reason) {
    S.state = 'done';
    Sfx.lose();
    UI.showLose({
      level: S.levelNo,
      score: S.board.stats.score,
      board: S.board,
      reason: reason || (S.timed ? 'time' : 'moves'),
      timed: S.timed,
    });
  }

  /** شراء 5 حركات إضافية والاستمرار */
  function continueWithMoves() {
    if (S.timed) S.timeLeft += 30;
    else S.moves += CFG.economy.extraMovesAmount;
    S.bombBlown = null;
    /* قنبلة انفجرت: أعِد ضبط فتائل ما تبقّى حتى لا تنفجر فوراً مرّة أخرى */
    for (let r = 0; r < S.board.rows; r++) {
      for (let c = 0; c < S.board.cols; c++) {
        const pc = S.board.grid[r][c].piece;
        if (pc && pc.kind === 'bomb' && pc.fuse <= 0) pc.fuse = S.board.bombFuse;
      }
    }
    S.ending = false; S.endStage = '';
    S.state = 'idle';
    S.idleTime = 0;
    updateHud();
  }

  function giveUp() {
    S.state = 'done';
    stop();
  }

  /* ------------------------------------------------------------------ */
  /*                            التحديث                                  */
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
    const ms = dt * 1000;
    if (S.shakeT > 0) S.shakeT -= ms;

    /* المؤقّت الزمني يعمل ما دامت المرحلة جارية */
    if (S.timed && !S.ending && S.state !== 'done') {
      S.timeLeft -= dt;
      if (S.timeLeft <= 0) {
        S.timeLeft = 0;
        if (S.state === 'idle') { finishLose('time'); return; }
      }
    }
    FX.update(dt);
    tweenPieces(dt);

    switch (S.state) {
      case 'idle': {
        S.idleTime += ms;
        if (Save.flag('hints') && S.idleTime > CFG.time.hintDelay && !S.hint) {
          S.hint = S.board.findHint();
        }
        break;
      }

      case 'swap': {
        const a = S.swapAnim;
        if (!a) { S.state = 'idle'; break; }
        a.t += ms;
        if (a.t >= a.dur) {
          const A = S.board.grid[a.a.r][a.a.c], B = S.board.grid[a.b.r][a.b.c];
          S.board.rawSwap(A, B);
          if (a.back) {
            S.swapAnim = null;
            S.state = 'idle';
            break;
          }
          if (!a.ok) {
            /* تبديل غير مفيد: رجوع */
            a.back = true; a.t = 0; a.dur = CFG.time.swapBack;
            Sfx.invalid();
            break;
          }
          S.swapAnim = null;
          if (!a.free) consumeMove();
          S.board.lastSwap = [a.a.r + ',' + a.a.c, a.b.r + ',' + a.b.c];
          const combo = S.board.activateSwap(a.a.r, a.a.c, a.b.r, a.b.c);
          if (combo) {
            if (combo !== 'single') { UI.banner(comboName(combo)); Sfx.combo(); }
            startClear();
          } else {
            const n = S.board.applyMatches();
            if (n > 0) { Sfx.match(0); startClear(); }
            else { S.state = 'idle'; }
          }
        }
        break;
      }

      case 'clear': {
        S.t += ms;
        playFx(S.t);
        if (S.t >= S.clearDur) {
          const cleared = countClearing();
          S.board.finishClear();
          if (S.timed) addTime(cleared * CFG.timed.secPerPiece);
          S.board.fx.length = 0;
          S.state = 'fall';
          S.fallTimer = 0;
        }
        break;
      }

      case 'fall': {
        S.fallTimer += ms;
        let guard = 0;
        while (S.fallTimer >= CFG.time.fallStep && guard++ < 12) {
          S.fallTimer -= CFG.time.fallStep;
          const moved = S.board.gravityStep();
          playFx(1e9);
          S.board.fx.length = 0;
          if (!moved && piecesSettled()) { afterSettle(); break; }
          if (moved) Sfx.fall();
        }
        break;
      }

      case 'bonus': {
        S.bonusTimer += ms;
        if (S.bonusTimer < 170) break;
        S.bonusTimer = 0;
        if (S.bonusLeft > 0) {
          S.bonusLeft--;
          if (!S.timed) S.moves--;
          S.board.stats.score += CFG.score.leftoverMove;
          S.board.placeBonusRocket();
          const f = S.board.fx[S.board.fx.length - 1];
          if (f) { FX.ring(ccx(f.c), ccy(f.r), S.tile * 0.9, '#ffe08a'); }
          S.board.fx.length = 0;
          Sfx.create();
          updateHud();
        } else {
          S.endStage = 'blast';
          const n = S.board.detonateAllSpecials();
          if (n > 0) { Sfx.combo(); startClear(); }
          else finishWin();
        }
        break;
      }

      default: break;
    }

    updateHud();
  }

  function comboName(k) {
    switch (k) {
      case 'ball+ball': return 'مسح كامل!';
      case 'ball+rocket': return 'وابل صواريخ!';
      case 'ball+tnt': return 'عاصفة قنابل!';
      case 'rocket+rocket': return 'صليب الصواريخ!';
      case 'rocket+tnt': return 'انفجار عملاق!';
      case 'tnt+tnt': return 'قنبلة مضاعفة!';
      case 'ball+color': return 'كرة مضيئة!';
      default: return '';
    }
  }

  /** عدد القطع المعلَّمة للمسح في الموجة الحالية */
  function countClearing() {
    let n = 0;
    for (let r = 0; r < S.board.rows; r++) {
      for (let c = 0; c < S.board.cols; c++) {
        const p = S.board.grid[r][c].piece;
        if (p && p.clearing) n++;
      }
    }
    return n;
  }

  /** هل استقرّت مواضع العرض؟ */
  function piecesSettled() {
    for (let r = 0; r < S.board.rows; r++) {
      for (let c = 0; c < S.board.cols; c++) {
        const p = S.board.grid[r][c].piece;
        if (p && (Math.abs(p.rr - p.r) > 0.03 || Math.abs(p.cc - p.c) > 0.03)) return false;
      }
    }
    return true;
  }

  /** تقريب مواضع العرض نحو المواضع المنطقية */
  function tweenPieces(dt) {
    const fallSpeed = 15;       // صفوف في الثانية
    const slideSpeed = 13;
    for (let r = 0; r < S.board.rows; r++) {
      for (let c = 0; c < S.board.cols; c++) {
        const p = S.board.grid[r][c].piece;
        if (!p) continue;
        const dr = p.r - p.rr;
        if (Math.abs(dr) > 0.001) {
          const step = fallSpeed * dt * (1 + Math.min(2, Math.abs(dr)));
          p.rr += U.clamp(dr, -step, step);
          if (Math.abs(p.r - p.rr) < 0.02) p.rr = p.r;
        }
        const dc = p.c - p.cc;
        if (Math.abs(dc) > 0.001) {
          const step = slideSpeed * dt * (1 + Math.min(2, Math.abs(dc)));
          p.cc += U.clamp(dc, -step, step);
          if (Math.abs(p.c - p.cc) < 0.02) p.cc = p.c;
        }
      }
    }
  }

  /** تحويل أحداث اللوحة إلى مؤثّرات بصرية/صوتية عند حلول وقتها */
  function playFx(t) {
    const budget = S.sfxBudget;
    const can = k => { budget[k] = (budget[k] || 0) + 1; return budget[k] <= 3; };

    for (const f of S.board.fx) {
      if (f.fired) continue;
      if ((f.delay || 0) > t) continue;
      f.fired = true;
      const x = ccx(f.c), y = ccy(f.r);
      switch (f.type) {
        case 'pop': {
          const t2 = PIECE_TYPES[f.color] || PIECE_TYPES[0];
          FX.burst(x, y, t2.color, 8, 1);
          break;
        }
        case 'box':
          FX.shards(x, y, f.broken ? '#c9955a' : '#a3743c', f.broken ? 12 : 6);
          if (f.broken) FX.ring(x, y, S.tile * 0.8, '#e0b277');
          if (can('ob')) Sfx.obstacle();
          break;
        case 'ice':
          FX.shards(x, y, '#bfe9ff', f.broken ? 12 : 6);
          if (can('ob')) Sfx.obstacle();
          break;
        case 'chain':
          FX.shards(x, y, '#c3cbe0', f.broken ? 12 : 6);
          if (can('ob')) Sfx.obstacle();
          break;
        case 'grass':
          FX.shards(x, y, '#5fd36f', 8);
          if (can('ob')) Sfx.obstacle();
          break;
        case 'create':
          FX.ring(x, y, S.tile * 1.1, '#ffe08a');
          FX.burst(x, y, '#ffe08a', 14, 1.2);
          Sfx.create();
          break;
        case 'rocketH':
          FX.trail(x, y, -1, 0, S.tile * S.board.cols, '#fff');
          FX.trail(x, y, 1, 0, S.tile * S.board.cols, '#fff');
          if (can('rk')) Sfx.rocket();
          break;
        case 'rocketV':
          FX.trail(x, y, 0, -1, S.tile * S.board.rows, '#fff');
          FX.trail(x, y, 0, 1, S.tile * S.board.rows, '#fff');
          if (can('rk')) Sfx.rocket();
          break;
        case 'tnt':
          FX.ring(x, y, S.tile * (f.rad + 0.6), '#ff9a3d');
          FX.burst(x, y, '#ffcc4d', 26, 2);
          if (can('tnt')) Sfx.tnt();
          break;
        case 'ball':
          FX.ring(x, y, S.tile * 2.2, '#ffffff');
          FX.burst(x, y, '#ff7ad1', 22, 1.6);
          if (can('ball')) Sfx.ball();
          break;
        case 'beam':
          FX.beam(ccx(f.fc), ccy(f.fr), x, y,
            f.color >= 0 && PIECE_TYPES[f.color] ? PIECE_TYPES[f.color].light : '#ffffff');
          break;
        case 'defuse':
          FX.ring(x, y, S.tile * 1.3, '#7dff9a');
          FX.burst(x, y, '#7dff9a', 18, 1.3);
          FX.text(x, y, 'تم!', '#7dff9a', S.tile * 0.34);
          Sfx.collect();
          break;
        case 'collect':
          FX.burst(x, y, '#ffcc4d', 18, 1.4);
          FX.text(x, y, '+1', '#ffe08a', S.tile * 0.4);
          Sfx.collect();
          break;
        default: break;
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /*                              الرسم                                  */
  /* ------------------------------------------------------------------ */

  function draw() {
    const ctx = S.ctx;
    if (!ctx) return;
    ctx.clearRect(0, 0, S.w, S.h);

    ctx.save();
    if (S.shakeT > 0) {
      const k = S.shakeT / 260;
      ctx.translate(Math.sin(S.shakeT * 0.09) * 6 * k, 0);
    }

    /* إطار اللوحة */
    ctx.save();
    ctx.fillStyle = 'rgba(8,11,36,.55)';
    roundRect(ctx, 2, 2, S.w - 4, S.h - 4, 18);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,224,138,.35)';
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.restore();

    const T = S.tile;

    /* 1) خلفيات الخلايا + العشب + خانات التجميع */
    for (let r = 0; r < S.board.rows; r++) {
      for (let c = 0; c < S.board.cols; c++) {
        const cell = S.board.grid[r][c];
        if (!cell.exists) continue;
        ctx.drawImage(Art.sprite((r + c) % 2 ? 'tile' : 'tileAlt', T), cx(c), cy(r), T, T);
        if (cell.grass > 0) ctx.drawImage(Art.sprite(cell.grass > 1 ? 'grass2' : 'grass1', T), cx(c), cy(r), T, T);
        if (cell.collector) ctx.drawImage(Art.sprite('collector', T), cx(c), cy(r), T, T);
      }
    }

    /* 2) القطع */
    const now = performance.now();
    for (let r = 0; r < S.board.rows; r++) {
      for (let c = 0; c < S.board.cols; c++) {
        const cell = S.board.grid[r][c];
        const p = cell.piece;
        if (!p || cell.box > 0) continue;

        let px = S.ox + p.cc * T, py = S.oy + p.rr * T;
        let scale = 1, alpha = 1;

        /* حركة التبديل */
        if (S.swapAnim) {
          const a = S.swapAnim;
          const k = U.easeInOutQuad(Math.min(1, a.t / a.dur));
          if (a.a.r === r && a.a.c === c) {
            px = U.lerp(cx(a.a.c), cx(a.b.c), k);
            py = U.lerp(cy(a.a.r), cy(a.b.r), k);
          } else if (a.b.r === r && a.b.c === c) {
            px = U.lerp(cx(a.b.c), cx(a.a.c), k);
            py = U.lerp(cy(a.b.r), cy(a.a.r), k);
          }
        }

        /* تلاشي القطع الممسوحة */
        if (p.clearing) {
          const k = U.clamp((S.t - p.clearDelay) / CFG.time.pop, 0, 1);
          if (S.t < p.clearDelay) { scale = 1; }
          else { scale = 1 + 0.35 * Math.sin(k * Math.PI); alpha = 1 - k; }
        }

        /* نبض القطع الخاصّة المنشأة حديثاً */
        if (p.special && !p.clearing) {
          scale *= 1 + Math.sin(now / 260 + p.id) * 0.035;
        }

        /* تمييز المحدّدة (أو التي بدأ منها السحب) */
        const dragFrom = S.drag && S.drag.from;
        if ((S.sel && S.sel[0] === r && S.sel[1] === c) ||
            (dragFrom && dragFrom.r === r && dragFrom.c === c)) {
          scale *= 1.12;
          ctx.save();
          ctx.strokeStyle = '#ffe08a'; ctx.lineWidth = 3;
          roundRect(ctx, cx(c) + 2, cy(r) + 2, T - 4, T - 4, T * 0.2);
          ctx.stroke();
          ctx.restore();
        }

        /* تلميح */
        if (S.hint && ((S.hint[0] === r && S.hint[1] === c) || (S.hint[2] === r && S.hint[3] === c))) {
          scale *= 1 + Math.sin(now / 180) * 0.09;
        }

        ctx.save();
        ctx.globalAlpha = alpha;
        const size = T * scale;
        const off = (T - size) / 2;
        Art.drawPiece(ctx, p, px + off, py + off, size);
        ctx.restore();
      }
    }

    /* 3) العوائق فوق القطع */
    for (let r = 0; r < S.board.rows; r++) {
      for (let c = 0; c < S.board.cols; c++) {
        const cell = S.board.grid[r][c];
        if (!cell.exists) continue;
        if (cell.chain > 0) ctx.drawImage(Art.sprite(cell.chain > 1 ? 'chain2' : 'chain1', T), cx(c), cy(r), T, T);
        if (cell.ice > 0) ctx.drawImage(Art.sprite(cell.ice > 1 ? 'ice2' : 'ice1', T), cx(c), cy(r), T, T);
        if (cell.box > 0) ctx.drawImage(Art.sprite(cell.box > 1 ? 'box2' : 'box1', T), cx(c), cy(r), T, T);
      }
    }

    /* 4) المؤثّرات */
    FX.draw(ctx);

    /* 5) وضع المعزّز: تظليل خفيف */
    if (S.activeBooster) {
      ctx.save();
      ctx.fillStyle = 'rgba(255,224,138,.08)';
      ctx.fillRect(0, 0, S.w, S.h);
      ctx.restore();
    }

    ctx.restore();
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  /* ------------------------------------------------------------------ */
  /*                            شريط الحالة                              */
  /* ------------------------------------------------------------------ */

  function updateHud(force) {
    const mv = U.$('#moves-count');
    if (mv) {
      if (S.timed) {
        const txt = U.mmss(S.timeLeft);
        if (force || mv.textContent !== txt) {
          mv.textContent = txt;
          mv.parentElement.classList.toggle('low', S.timeLeft <= CFG.timed.warnAt);
        }
      } else if (force || mv.textContent !== String(S.moves)) {
        mv.textContent = S.moves;
        mv.parentElement.classList.toggle('low', S.moves <= 5);
      }
      if (force) {
        const lbl = mv.parentElement.querySelector('.moves-label');
        if (lbl) lbl.textContent = S.timed ? 'الوقت' : 'الحركات';
      }
    }
    const sc = U.$('#score-count');
    const score = S.board ? S.board.stats.score : 0;
    if (sc && (force || score !== S.lastScore)) {
      sc.textContent = U.fmt(score);
      S.lastScore = score;
    }
    if (S.board) UI.updateGoals(S.def, S.board);
  }

  /* ------------------------------------------------------------------ */

  return {
    start, stop, resize, armBooster, continueWithMoves, giveUp,
    get state() { return S.state; },
    get board() { return S.board; },
    get def() { return S.def; },
    get levelNo() { return S.levelNo; },
    get moves() { return S.moves; },
    S,
  };
})();

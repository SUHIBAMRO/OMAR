/* ==========================================================================
   board.js — نموذج اللوحة وكل قواعد اللعب
   ---------------------------------------------------------------------------
   قواعد التطابق:
     • 3 متجاورة بنفس اللون  -> تُمسح (بلا قطعة خاصّة)
     • 4 في خط              -> صاروخ باتجاه الخط
     • 5 في خط              -> كرة مضيئة
     • شكل L أو T (خطّان)    -> قنبلة TNT
   العوائق:
     • صندوق: يشغل الخليّة، يُكسر بمطابقة مجاورة أو بانفجار يصيبه
     • جليد / سلسلة: طبقة تُقفل القطعة، تُكسر بمطابقة القطعة نفسها أو انفجار
     • عشب: يُزال عند مسح القطعة التي فوقه
   ========================================================================== */
'use strict';

let __pieceId = 1;

class Board {

  constructor(def) {
    this.def = def;
    this.rows = def.rows;
    this.cols = def.cols;
    this.colors = def.colors;

    /* شبكة الخلايا */
    this.grid = [];
    for (let r = 0; r < this.rows; r++) {
      const row = [];
      for (let c = 0; c < this.cols; c++) {
        const ch = def.layout[r][c] || '.';
        row.push(this.makeCell(ch));
      }
      this.grid.push(row);
    }

    /* الخلايا المولِّدة: كل خليّة موجودة لا يوجد فوقها خليّة */
    for (let c = 0; c < this.cols; c++) {
      for (let r = 0; r < this.rows; r++) {
        const cell = this.grid[r][c];
        if (!cell.exists) continue;
        const up = r > 0 ? this.grid[r - 1][c] : null;
        cell.spawner = (r === 0) || !up.exists;
      }
    }

    /* أعمدة فيها خانة تجميع — الأغراض تنزل فيها فقط */
    this.itemCols = [];
    for (let c = 0; c < this.cols; c++) {
      for (let r = 0; r < this.rows; r++) {
        if (this.grid[r][c].collector) { this.itemCols.push(c); break; }
      }
    }

    this.itemsLeft = def.items || 0;
    this.itemsOnBoard = 0;

    /* إحصاءات المرحلة */
    this.stats = {
      score: 0,
      colors: new Array(PIECE_TYPES.length).fill(0),
      box: 0, ice: 0, chain: 0, grass: 0, item: 0,
    };

    this.fx = [];          // أحداث بصرية ينتظرها العارض
    this.phase = 0;        // رقم موجة المسح الحالية (لمنع ضرر مزدوج)
    this.cascade = 0;      // عمق التتابع الحالي (يزيد المضاعف)
    this.lastSwap = null;  // خليّتا آخر تبديل — لتحديد مكان القطعة الخاصّة

    this.fill(true);
  }

  /* ------------------------------------------------------------------ */
  /*                              بناء                                   */
  /* ------------------------------------------------------------------ */

  makeCell(ch) {
    const cell = {
      exists: ch !== '#',
      spawner: false,
      collector: ch === '_',
      piece: null,
      box: 0, ice: 0, chain: 0, grass: 0,
      hitPhase: -1,
    };
    switch (ch) {
      case 'b': cell.box = 1; break;
      case 'B': cell.box = 2; break;
      case 'i': cell.ice = 1; break;
      case 'I': cell.ice = 2; break;
      case 'k': cell.chain = 1; break;
      case 'K': cell.chain = 2; break;
      case 'g': cell.grass = 1; break;
      case 'G': cell.grass = 2; break;
      default: break;
    }
    return cell;
  }

  newPiece(color, r, c, special) {
    return {
      id: __pieceId++,
      color: color,
      special: special || SPECIAL.NONE,
      kind: 'normal',
      r: r, c: c,
      rr: r, cc: c,          // إحداثيات العرض (قد تختلف أثناء الحركة)
      clearing: false,
      clearDelay: 0,
      born: 0,
      justMade: false,
    };
  }

  newItem(r, c) {
    const p = this.newPiece(-1, r, c);
    p.kind = 'item';
    return p;
  }

  /** ملء اللوحة بقطع عشوائية بلا تطابقات جاهزة */
  fill(initial) {
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const cell = this.grid[r][c];
        if (!cell.exists || cell.box > 0 || cell.piece) continue;
        cell.piece = this.newPiece(this.safeColor(r, c), r, c);
      }
    }
    if (initial) {
      let guard = 0;
      while (this.findMatches().length && guard++ < 60) {
        for (const g of this.findMatches()) {
          for (const k of g.cells) {
            const [r, c] = k.split(',').map(Number);
            const cell = this.grid[r][c];
            if (cell.piece) cell.piece.color = this.safeColor(r, c);
          }
        }
      }
      guard = 0;
      while (!this.hasMoves() && guard++ < 40) this.shuffle(true);
    }
  }

  /** لون لا يصنع تطابقاً فورياً في هذا الموضع */
  safeColor(r, c) {
    const bad = new Set();
    if (r >= 2) {
      const a = this.colorAt(r - 1, c), b = this.colorAt(r - 2, c);
      if (a >= 0 && a === b) bad.add(a);
    }
    if (c >= 2) {
      const a = this.colorAt(r, c - 1), b = this.colorAt(r, c - 2);
      if (a >= 0 && a === b) bad.add(a);
    }
    const opts = [];
    for (let i = 0; i < this.colors; i++) if (!bad.has(i)) opts.push(i);
    return opts.length ? U.pick(opts) : U.randInt(0, this.colors);
  }

  /* ------------------------------------------------------------------ */
  /*                            استعلامات                                */
  /* ------------------------------------------------------------------ */

  inside(r, c) { return r >= 0 && c >= 0 && r < this.rows && c < this.cols; }
  at(r, c) { return this.inside(r, c) ? this.grid[r][c] : null; }

  /** لون القطعة القابل للمطابقة (-1 إن لم تكن قابلة) */
  colorAt(r, c) {
    const cell = this.at(r, c);
    if (!cell || !cell.exists || cell.box > 0) return -1;
    const p = cell.piece;
    if (!p || p.kind !== 'normal' || p.clearing) return -1;
    if (p.special !== SPECIAL.NONE) return -1;   // القطع الخاصّة بلا لون
    return p.color;
  }

  /** هل يمكن تحريك القطعة في هذه الخليّة؟ */
  isFree(r, c) {
    const cell = this.at(r, c);
    if (!cell || !cell.exists || cell.box > 0) return false;
    if (cell.ice > 0 || cell.chain > 0) return false;
    const p = cell.piece;
    return !!p && !p.clearing;
  }

  /** هل التبديل بين خليّتين مسموح شكلياً؟ */
  canSwap(r1, c1, r2, c2) {
    if (Math.abs(r1 - r2) + Math.abs(c1 - c2) !== 1) return false;
    if (!this.isFree(r1, c1) || !this.isFree(r2, c2)) return false;
    const a = this.grid[r1][c1].piece, b = this.grid[r2][c2].piece;
    if (a.kind === 'item' || b.kind === 'item') return false;
    return true;
  }

  /** هل ينتج عن هذا التبديل حركة صالحة (تطابق أو تفعيل قطعة خاصّة)؟ */
  swapIsUseful(r1, c1, r2, c2) {
    const A = this.grid[r1][c1], B = this.grid[r2][c2];
    if (A.piece.special || B.piece.special) return true;
    this.rawSwap(A, B);
    const ok = this.findMatches().length > 0;
    this.rawSwap(A, B);
    return ok;
  }

  rawSwap(A, B) {
    const t = A.piece; A.piece = B.piece; B.piece = t;
    this.syncCoords();
  }

  /** يعيد ضبط r,c لكل قطعة حسب موقعها في الشبكة */
  syncCoords() {
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const p = this.grid[r][c].piece;
        if (p) { p.r = r; p.c = c; }
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /*                             المطابقة                                */
  /* ------------------------------------------------------------------ */

  /**
   * يجد كل مجموعات التطابق على اللوحة.
   * كل مجموعة: { cells:Set<"r,c">, color, maxH, maxV, best:{dir,len,cells} }
   */
  findMatches() {
    const runs = [];

    for (let r = 0; r < this.rows; r++) {
      let c = 0;
      while (c < this.cols) {
        const col = this.colorAt(r, c);
        if (col < 0) { c++; continue; }
        let e = c;
        while (e + 1 < this.cols && this.colorAt(r, e + 1) === col) e++;
        const len = e - c + 1;
        if (len >= 3) {
          const cells = [];
          for (let x = c; x <= e; x++) cells.push(r + ',' + x);
          runs.push({ dir: 'h', len, color: col, cells, r, c });
        }
        c = e + 1;
      }
    }

    for (let c = 0; c < this.cols; c++) {
      let r = 0;
      while (r < this.rows) {
        const col = this.colorAt(r, c);
        if (col < 0) { r++; continue; }
        let e = r;
        while (e + 1 < this.rows && this.colorAt(e + 1, c) === col) e++;
        const len = e - r + 1;
        if (len >= 3) {
          const cells = [];
          for (let y = r; y <= e; y++) cells.push(y + ',' + c);
          runs.push({ dir: 'v', len, color: col, cells, r, c });
        }
        r = e + 1;
      }
    }

    if (!runs.length) return [];

    /* دمج الخطوط المتقاطعة في مجموعات */
    const owner = new Map();     // "r,c" -> index المجموعة
    const groups = [];

    for (const run of runs) {
      let gi = -1;
      for (const k of run.cells) {
        if (owner.has(k)) { gi = owner.get(k); break; }
      }
      if (gi === -1) {
        gi = groups.length;
        groups.push({ cells: new Set(), color: run.color, runs: [], maxH: 0, maxV: 0 });
      }
      const g = groups[gi];
      g.runs.push(run);
      if (run.dir === 'h') g.maxH = Math.max(g.maxH, run.len);
      else g.maxV = Math.max(g.maxV, run.len);
      for (const k of run.cells) { g.cells.add(k); owner.set(k, gi); }
    }

    /* دمج ثانٍ: قد ينضمّ خطّ إلى مجموعتين */
    const merged = [];
    const seen = new Set();
    for (let i = 0; i < groups.length; i++) {
      if (seen.has(i)) continue;
      const g = groups[i];
      for (let j = i + 1; j < groups.length; j++) {
        if (seen.has(j)) continue;
        const h = groups[j];
        if (h.color !== g.color) continue;
        let touch = false;
        for (const k of h.cells) if (g.cells.has(k)) { touch = true; break; }
        if (touch) {
          for (const k of h.cells) g.cells.add(k);
          g.runs = g.runs.concat(h.runs);
          g.maxH = Math.max(g.maxH, h.maxH);
          g.maxV = Math.max(g.maxV, h.maxV);
          seen.add(j);
        }
      }
      merged.push(g);
    }
    return merged;
  }

  /** يحدّد نوع القطعة الخاصّة الناتجة عن مجموعة */
  specialFor(group) {
    const maxLen = Math.max(group.maxH, group.maxV);
    if (maxLen >= 5) return SPECIAL.BALL;
    if (group.maxH >= 3 && group.maxV >= 3) return SPECIAL.TNT;
    if (maxLen === 4) {
      const run = group.runs.find(x => x.len === 4);
      return run && run.dir === 'h' ? SPECIAL.ROCKET_H : SPECIAL.ROCKET_V;
    }
    return SPECIAL.NONE;
  }

  /** هل الخليّة صالحة لاستضافة قطعة خاصّة جديدة؟ (غير مقفلة) */
  spotOk(k) {
    const [r, c] = k.split(',').map(Number);
    const cell = this.at(r, c);
    return !!cell && cell.exists && cell.box === 0 && cell.ice === 0 && cell.chain === 0;
  }

  /** أين تُوضع القطعة الخاصّة */
  specialSpot(group) {
    /* 1) إحدى خليّتي التبديل إن كانت ضمن المجموعة */
    if (this.lastSwap) {
      for (const k of this.lastSwap) if (group.cells.has(k) && this.spotOk(k)) return k;
    }
    /* 2) نقطة تقاطع خطّ أفقي وخطّ عمودي */
    if (group.maxH >= 3 && group.maxV >= 3) {
      const hs = group.runs.filter(x => x.dir === 'h');
      const vs = group.runs.filter(x => x.dir === 'v');
      for (const h of hs) for (const v of vs) {
        for (const k of h.cells) if (v.cells.indexOf(k) >= 0 && this.spotOk(k)) return k;
      }
    }
    /* 3) منتصف أطول خطّ (مع تفادي الخلايا المقفلة) */
    let best = group.runs[0];
    for (const run of group.runs) if (run.len > best.len) best = run;
    const mid = Math.floor(best.cells.length / 2);
    for (let d = 0; d < best.cells.length; d++) {
      for (const k of [best.cells[mid + d], best.cells[mid - d]]) {
        if (k && this.spotOk(k)) return k;
      }
    }
    for (const k of group.cells) if (this.spotOk(k)) return k;
    return best.cells[mid];
  }

  /* ------------------------------------------------------------------ */
  /*                       المسح والانفجارات                            */
  /* ------------------------------------------------------------------ */

  beginPhase() { this.phase++; }

  /**
   * يضع علامة المسح على خليّة واحدة.
   * @param opts.fromMatch  صحيح إذا جاء من مطابقة (يضرّ الصناديق المجاورة)
   * @param opts.skipSpecial لا يفعّل القطعة الخاصّة الموجودة (تُستعمل عند إنشائها)
   */
  markClear(r, c, delay, opts) {
    opts = opts || {};
    const cell = this.at(r, c);
    if (!cell || !cell.exists) return false;

    /* الصندوق يمتصّ الضربة ولا شيء تحته */
    if (cell.box > 0) {
      if (cell.hitPhase !== this.phase) {
        cell.hitPhase = this.phase;
        cell.box--;
        this.stats.box++;
        this.stats.score += CFG.score.obstacleHit;
        this.fx.push({ type: 'box', r, c, delay, broken: cell.box === 0 });
      }
      return true;
    }

    /* السلسلة ثم الجليد يحميان القطعة */
    if (cell.chain > 0) {
      if (cell.hitPhase !== this.phase) {
        cell.hitPhase = this.phase;
        cell.chain--;
        this.stats.chain++;
        this.stats.score += CFG.score.obstacleHit;
        this.fx.push({ type: 'chain', r, c, delay, broken: cell.chain === 0 });
      }
      return true;
    }
    if (cell.ice > 0) {
      if (cell.hitPhase !== this.phase) {
        cell.hitPhase = this.phase;
        cell.ice--;
        this.stats.ice++;
        this.stats.score += CFG.score.obstacleHit;
        this.fx.push({ type: 'ice', r, c, delay, broken: cell.ice === 0 });
      }
      return true;
    }

    const p = cell.piece;
    if (!p) return false;
    if (p.kind === 'item') return false;      // الأغراض لا تُدمَّر، تُنزَّل فقط
    if (p.clearing) return false;

    p.clearing = true;
    p.clearDelay = delay;

    if (p.special !== SPECIAL.NONE && !opts.skipSpecial) {
      this.queueDetonate(r, c, p.special, delay + CFG.time.blastDelay, p.color);
    } else {
      this.stats.score += Math.round(CFG.score.piece * this.cascadeMul());
      if (p.color >= 0) this.stats.colors[p.color]++;
      this.fx.push({ type: 'pop', r, c, delay, color: p.color });
    }

    /* العشب أسفل القطعة يُزال معها */
    if (cell.grass > 0) {
      cell.grass--;
      this.stats.grass++;
      this.stats.score += CFG.score.obstacleHit;
      this.fx.push({ type: 'grass', r, c, delay, broken: cell.grass === 0 });
    }

    /* المطابقة تضرّ الصناديق المجاورة */
    if (opts.fromMatch) {
      const N = [[1, 0], [-1, 0], [0, 1], [0, -1]];
      for (const [dr, dc] of N) {
        const n = this.at(r + dr, c + dc);
        if (n && n.exists && n.box > 0 && n.hitPhase !== this.phase) {
          n.hitPhase = this.phase;
          n.box--;
          this.stats.box++;
          this.stats.score += CFG.score.obstacleHit;
          this.fx.push({ type: 'box', r: r + dr, c: c + dc, delay, broken: n.box === 0 });
        }
      }
    }
    return true;
  }

  cascadeMul() {
    const t = CFG.score.cascadeMul;
    return t[Math.min(this.cascade, t.length - 1)];
  }

  /** يجدول انفجار قطعة خاصّة */
  queueDetonate(r, c, special, delay, color) {
    this.stats.score += Math.round(CFG.score.specialBlast * this.cascadeMul());
    switch (special) {
      case SPECIAL.ROCKET_H: this.blastRow(r, c, delay, 0); break;
      case SPECIAL.ROCKET_V: this.blastCol(r, c, delay, 0); break;
      case SPECIAL.TNT:      this.blastArea(r, c, CFG.board.tntRadius, delay); break;
      case SPECIAL.BALL:     this.blastBall(r, c, this.mostCommonColor(), delay, 'clear'); break;
      default: break;
    }
  }

  blastRow(r, c, delay, spread) {
    this.fx.push({ type: 'rocketH', r, c, delay });
    for (let d = 0; d <= Math.max(c, this.cols - c); d++) {
      for (const x of [c - d, c + d]) {
        if (x < 0 || x >= this.cols) continue;
        if (d === 0 && x !== c) continue;
        for (let dr = -spread; dr <= spread; dr++) {
          this.markClear(r + dr, x, delay + d * 22);
        }
      }
    }
  }

  blastCol(r, c, delay, spread) {
    this.fx.push({ type: 'rocketV', r, c, delay });
    for (let d = 0; d <= Math.max(r, this.rows - r); d++) {
      for (const y of [r - d, r + d]) {
        if (y < 0 || y >= this.rows) continue;
        if (d === 0 && y !== r) continue;
        for (let dc = -spread; dc <= spread; dc++) {
          this.markClear(y, c + dc, delay + d * 22);
        }
      }
    }
  }

  blastArea(r, c, rad, delay) {
    this.fx.push({ type: 'tnt', r, c, delay, rad });
    for (let dr = -rad; dr <= rad; dr++) {
      for (let dc = -rad; dc <= rad; dc++) {
        if (Math.abs(dr) === rad && Math.abs(dc) === rad) continue;  // بلا أركان
        const dist = Math.max(Math.abs(dr), Math.abs(dc));
        this.markClear(r + dr, c + dc, delay + dist * 40);
      }
    }
  }

  /**
   * الكرة المضيئة.
   * mode: 'clear' تمسح اللون | 'rocket' تحوّله صواريخ | 'tnt' تحوّله قنابل
   */
  blastBall(r, c, color, delay, mode) {
    this.fx.push({ type: 'ball', r, c, delay, color });
    const targets = [];
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        const cell = this.grid[y][x];
        if (!cell.exists || cell.box > 0) continue;
        const p = cell.piece;
        if (!p || p.kind !== 'normal' || p.clearing) continue;
        if (color >= 0 && p.color !== color) continue;
        if (color >= 0 && p.special !== SPECIAL.NONE) continue;
        targets.push([y, x]);
      }
    }
    targets.sort((a, b) => (Math.abs(a[0] - r) + Math.abs(a[1] - c)) - (Math.abs(b[0] - r) + Math.abs(b[1] - c)));
    targets.forEach((t, i) => {
      const d = delay + 60 + i * 26;
      this.fx.push({ type: 'beam', r: t[0], c: t[1], fr: r, fc: c, delay: delay + i * 26 });
      if (mode === 'rocket') {
        const cell = this.grid[t[0]][t[1]];
        if (cell.piece) cell.piece.special = Math.random() < 0.5 ? SPECIAL.ROCKET_H : SPECIAL.ROCKET_V;
        this.markClear(t[0], t[1], d);
      } else if (mode === 'tnt') {
        const cell = this.grid[t[0]][t[1]];
        if (cell.piece) cell.piece.special = SPECIAL.TNT;
        this.markClear(t[0], t[1], d);
      } else {
        this.markClear(t[0], t[1], d);
      }
    });
  }

  mostCommonColor() {
    const n = new Array(this.colors).fill(0);
    for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
      const col = this.colorAt(r, c);
      if (col >= 0) n[col]++;
    }
    let best = 0;
    for (let i = 1; i < this.colors; i++) if (n[i] > n[best]) best = i;
    return best;
  }

  /* ------------------------------------------------------------------ */
  /*                      تطبيق التطابقات والدمج                        */
  /* ------------------------------------------------------------------ */

  /** يطبّق كل التطابقات الحالية. يُرجع عدد المجموعات (0 = لا شيء) */
  applyMatches() {
    const groups = this.findMatches();
    if (!groups.length) return 0;

    this.beginPhase();
    const creations = [];

    for (const g of groups) {
      const sp = this.specialFor(g);
      const spot = sp !== SPECIAL.NONE ? this.specialSpot(g) : null;

      for (const k of g.cells) {
        const [r, c] = k.split(',').map(Number);
        if (k === spot) continue;
        this.markClear(r, c, 0, { fromMatch: true });
      }

      if (sp !== SPECIAL.NONE) {
        const [r, c] = spot.split(',').map(Number);
        creations.push({ r, c, sp, color: g.color });
        this.stats.score += CFG.score.specialCreate;
        /* العشب تحت القطعة الخاصّة يُزال أيضاً */
        const cell = this.grid[r][c];
        if (cell.grass > 0) {
          cell.grass--;
          this.stats.grass++;
          this.fx.push({ type: 'grass', r, c, delay: 0, broken: cell.grass === 0 });
        }
      }
    }

    /* إنشاء القطع الخاصّة بعد انتهاء التعليم حتى لا تتأثّر بالمسح */
    for (const cr of creations) {
      const cell = this.grid[cr.r][cr.c];
      if (!cell.piece) cell.piece = this.newPiece(cr.color, cr.r, cr.c);
      cell.piece.clearing = false;
      cell.piece.special = cr.sp;
      cell.piece.color = cr.color;
      cell.piece.justMade = true;
      this.fx.push({ type: 'create', r: cr.r, c: cr.c, sp: cr.sp, delay: CFG.time.pop * 0.4 });
    }

    this.lastSwap = null;
    return groups.length;
  }

  /** يحذف فعلياً القطع المعلَّمة. يُرجع عددها */
  finishClear() {
    let n = 0;
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const cell = this.grid[r][c];
        const p = cell.piece;
        if (p && p.clearing) { cell.piece = null; n++; }
        if (p) p.justMade = false;
      }
    }
    return n;
  }

  /** أطول تأخير بصري في الموجة الحالية (لضبط التوقيت) */
  maxDelay() {
    let m = 0;
    for (const f of this.fx) m = Math.max(m, f.delay || 0);
    for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
      const p = this.grid[r][c].piece;
      if (p && p.clearing) m = Math.max(m, p.clearDelay);
    }
    return m;
  }

  /* ------------------------------------------------------------------ */
  /*                            الجاذبية                                 */
  /* ------------------------------------------------------------------ */

  /** هل يمكن لهذه الخليّة أن تُغذّى من أعلى عمودها؟ */
  columnFeeds(r, c) {
    for (let y = r - 1; y >= 0; y--) {
      const cell = this.grid[y][c];
      if (!cell.exists || cell.box > 0) return false;
      if (cell.piece) return !(cell.ice > 0 || cell.chain > 0) ? true : false;
      if (cell.spawner) return true;
    }
    return false;
  }

  /**
   * خطوة سقوط واحدة: كل قطعة تنزل خليّة واحدة على الأكثر.
   * يُرجع true إذا تحرّك أو ظهر أي شيء.
   */
  gravityStep() {
    let moved = false;

    for (let r = this.rows - 1; r >= 0; r--) {
      for (let c = 0; c < this.cols; c++) {
        const cell = this.grid[r][c];
        if (!cell.exists || cell.box > 0 || cell.piece) continue;

        /* 1) سقوط مباشر من الأعلى */
        const up = r > 0 ? this.grid[r - 1][c] : null;
        if (up && up.exists && up.box === 0 && up.piece &&
            up.ice === 0 && up.chain === 0) {
          this.movePiece(r - 1, c, r, c);
          moved = true;
          continue;
        }

        /* 2) لا يوجد تغذية عمودية -> انزلاق قطري */
        if (!this.columnFeeds(r, c)) {
          let done = false;
          for (const dc of U.shuffle([-1, 1])) {
            const sr = r - 1, sc = c + dc;
            const src = this.at(sr, sc);
            if (!src || !src.exists || src.box > 0 || !src.piece) continue;
            if (src.ice > 0 || src.chain > 0) continue;
            /* الأغراض الملكية تنزل عمودياً فقط، وإلّا خرجت من عمود التجميع
               وتعذّر جمعها نهائياً */
            if (src.piece.kind === 'item') continue;
            /* القطعة المصدر يجب ألّا تستطيع النزول عمودياً بنفسها */
            const below = this.at(sr + 1, sc);
            if (below && below.exists && below.box === 0 && !below.piece) continue;
            this.movePiece(sr, sc, r, c);
            moved = true; done = true;
            break;
          }
          if (done) continue;
        }

        /* 3) توليد قطعة جديدة من الأعلى */
        if (cell.spawner) {
          let p;
          if (this.itemsLeft > 0 && this.itemCols.indexOf(c) >= 0 &&
              this.itemsOnBoard < 2 && Math.random() < 0.22) {
            p = this.newItem(r, c);
            this.itemsLeft--;
            this.itemsOnBoard++;
          } else {
            p = this.newPiece(U.randInt(0, this.colors), r, c);
          }
          p.rr = r - 1;      // تبدأ فوق اللوحة ثم تنزل
          p.cc = c;
          cell.piece = p;
          moved = true;
        }
      }
    }

    if (moved) this.collectItems();
    return moved;
  }

  movePiece(r1, c1, r2, c2) {
    const A = this.grid[r1][c1], B = this.grid[r2][c2];
    B.piece = A.piece;
    A.piece = null;
    B.piece.r = r2; B.piece.c = c2;
  }

  /** جمع الأغراض التي وصلت إلى خانات التجميع */
  collectItems() {
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const cell = this.grid[r][c];
        if (!cell.collector || !cell.piece) continue;
        if (cell.piece.kind !== 'item') continue;
        cell.piece = null;
        this.itemsOnBoard--;
        this.stats.item++;
        this.stats.score += CFG.score.piece * 3;
        this.fx.push({ type: 'collect', r, c, delay: 0 });
      }
    }
  }

  /** هل استقرّت اللوحة (لا فراغات قابلة للملء)؟ */
  isSettled() {
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const cell = this.grid[r][c];
        if (cell.exists && cell.box === 0 && !cell.piece) return false;
      }
    }
    return true;
  }

  /* ------------------------------------------------------------------ */
  /*                      الحركات المتاحة والخلط                        */
  /* ------------------------------------------------------------------ */

  /** يبحث عن حركة صالحة؛ يُرجع [r1,c1,r2,c2] أو null */
  findHint() {
    const pairs = [];
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        if (c + 1 < this.cols) pairs.push([r, c, r, c + 1]);
        if (r + 1 < this.rows) pairs.push([r, c, r + 1, c]);
      }
    }
    U.shuffle(pairs);
    for (const p of pairs) {
      if (!this.canSwap(p[0], p[1], p[2], p[3])) continue;
      if (this.swapIsUseful(p[0], p[1], p[2], p[3])) return p;
    }
    return null;
  }

  hasMoves() { return !!this.findHint(); }

  /** يخلط كل القطع الحرّة */
  shuffle(silent) {
    const spots = [], colorsList = [];
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        if (!this.isFree(r, c)) continue;
        const p = this.grid[r][c].piece;
        if (p.kind !== 'normal' || p.special !== SPECIAL.NONE) continue;
        spots.push([r, c]);
        colorsList.push(p.color);
      }
    }
    if (spots.length < 3) return false;

    for (let attempt = 0; attempt < 80; attempt++) {
      U.shuffle(colorsList);
      spots.forEach((s, i) => { this.grid[s[0]][s[1]].piece.color = colorsList[i]; });
      if (!this.findMatches().length && this.hasMoves()) {
        if (!silent) this.fx.push({ type: 'shuffle', delay: 0 });
        return true;
      }
    }
    return true;
  }

  /* ------------------------------------------------------------------ */
  /*                       دمج القطع الخاصّة                            */
  /* ------------------------------------------------------------------ */

  /**
   * يفعّل نتيجة تبديل قطعتين إحداهما خاصّة على الأقل.
   * يُرجع اسم الدمج للعرض، أو '' إن لم يكن هناك تفعيل.
   */
  activateSwap(r1, c1, r2, c2) {
    const A = this.grid[r1][c1].piece;
    const B = this.grid[r2][c2].piece;
    if (!A || !B) return '';
    const sa = A.special, sb = B.special;
    if (!sa && !sb) return '';

    this.beginPhase();
    const isRocket = s => s === SPECIAL.ROCKET_H || s === SPECIAL.ROCKET_V;

    /* --- كرة + كرة: مسح اللوحة كاملة --- */
    if (sa === SPECIAL.BALL && sb === SPECIAL.BALL) {
      A.special = SPECIAL.NONE; B.special = SPECIAL.NONE;
      this.fx.push({ type: 'ball', r: r2, c: c2, delay: 0, color: -1 });
      for (let r = 0; r < this.rows; r++) {
        for (let c = 0; c < this.cols; c++) {
          const d = (Math.abs(r - r2) + Math.abs(c - c2)) * 34;
          this.markClear(r, c, d);
        }
      }
      this.stats.score += CFG.score.comboCreate * 2;
      return 'ball+ball';
    }

    /* --- كرة + خاصّة أخرى --- */
    if (sa === SPECIAL.BALL || sb === SPECIAL.BALL) {
      const ballAt = sa === SPECIAL.BALL ? [r1, c1] : [r2, c2];
      const other = sa === SPECIAL.BALL ? B : A;
      const otherAt = sa === SPECIAL.BALL ? [r2, c2] : [r1, c1];
      const ballPiece = sa === SPECIAL.BALL ? A : B;
      ballPiece.special = SPECIAL.NONE;

      if (isRocket(other.special)) {
        other.special = SPECIAL.NONE;
        this.markClear(otherAt[0], otherAt[1], 0);
        this.blastBall(ballAt[0], ballAt[1], this.mostCommonColor(), 0, 'rocket');
        this.markClear(ballAt[0], ballAt[1], 0);
        this.stats.score += CFG.score.comboCreate;
        return 'ball+rocket';
      }
      if (other.special === SPECIAL.TNT) {
        other.special = SPECIAL.NONE;
        this.markClear(otherAt[0], otherAt[1], 0);
        this.blastBall(ballAt[0], ballAt[1], this.mostCommonColor(), 0, 'tnt');
        this.markClear(ballAt[0], ballAt[1], 0);
        this.stats.score += CFG.score.comboCreate;
        return 'ball+tnt';
      }
      /* كرة + قطعة عادية: مسح كل اللون */
      const color = other.color;
      this.blastBall(ballAt[0], ballAt[1], color, 0, 'clear');
      this.markClear(ballAt[0], ballAt[1], 0);
      this.markClear(otherAt[0], otherAt[1], 0);
      return 'ball+color';
    }

    /* --- صاروخ + صاروخ: صليب كامل --- */
    if (isRocket(sa) && isRocket(sb)) {
      A.special = SPECIAL.NONE; B.special = SPECIAL.NONE;
      this.markClear(r1, c1, 0); this.markClear(r2, c2, 0);
      this.blastRow(r2, c2, 0, 0);
      this.blastCol(r2, c2, 0, 0);
      this.stats.score += CFG.score.comboCreate;
      return 'rocket+rocket';
    }

    /* --- صاروخ + TNT: صليب عريض --- */
    if ((isRocket(sa) && sb === SPECIAL.TNT) || (sa === SPECIAL.TNT && isRocket(sb))) {
      A.special = SPECIAL.NONE; B.special = SPECIAL.NONE;
      this.markClear(r1, c1, 0); this.markClear(r2, c2, 0);
      this.blastRow(r2, c2, 0, CFG.board.rocketWide);
      this.blastCol(r2, c2, 0, CFG.board.rocketWide);
      this.stats.score += CFG.score.comboCreate;
      return 'rocket+tnt';
    }

    /* --- TNT + TNT: انفجار ضخم --- */
    if (sa === SPECIAL.TNT && sb === SPECIAL.TNT) {
      A.special = SPECIAL.NONE; B.special = SPECIAL.NONE;
      this.markClear(r1, c1, 0); this.markClear(r2, c2, 0);
      this.blastArea(r2, c2, CFG.board.tntBigRadius, 0);
      this.stats.score += CFG.score.comboCreate;
      return 'tnt+tnt';
    }

    /* --- خاصّة واحدة + قطعة عادية: تفعيل مباشر --- */
    const spR = sa ? r1 : r2, spC = sa ? c1 : c2;
    this.markClear(spR, spC, 0);
    return 'single';
  }

  /** تفعيل قطعة بالنقر عليها (بعد وضعها بمعزّز مثلاً) */
  detonateAt(r, c) {
    const cell = this.at(r, c);
    if (!cell || !cell.piece || !cell.piece.special) return false;
    this.beginPhase();
    this.markClear(r, c, 0);
    return true;
  }

  /* ------------------------------------------------------------------ */
  /*                             الأهداف                                 */
  /* ------------------------------------------------------------------ */

  /** التقدّم الحالي لهدف معيّن */
  goalProgress(goal) {
    switch (goal.type) {
      case GOAL.COLOR: return this.stats.colors[goal.color];
      case GOAL.BOX:   return this.stats.box;
      case GOAL.ICE:   return this.stats.ice;
      case GOAL.CHAIN: return this.stats.chain;
      case GOAL.GRASS: return this.stats.grass;
      case GOAL.ITEM:  return this.stats.item;
      default: return 0;
    }
  }

  goalsDone() {
    return this.def.goals.every(g => this.goalProgress(g) >= g.count);
  }

  /** عدد النجوم حسب النقاط */
  stars() {
    const s = this.stats.score, t = this.def.stars;
    if (s >= t[2]) return 3;
    if (s >= t[1]) return 2;
    if (s >= t[0]) return 1;
    return 1;   // الفوز يمنح نجمة على الأقل
  }

  /** يضع صواريخ عشوائية مكافأةً على الحركات المتبقّية */
  placeBonusRocket() {
    const spots = [];
    for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
      if (this.isFree(r, c)) {
        const p = this.grid[r][c].piece;
        if (p.kind === 'normal' && p.special === SPECIAL.NONE) spots.push([r, c]);
      }
    }
    if (!spots.length) return false;
    const [r, c] = U.pick(spots);
    this.grid[r][c].piece.special = Math.random() < 0.5 ? SPECIAL.ROCKET_H : SPECIAL.ROCKET_V;
    this.fx.push({ type: 'create', r, c, sp: this.grid[r][c].piece.special, delay: 0 });
    return true;
  }

  /** يفجّر كل القطع الخاصّة الموجودة (نهاية المرحلة) */
  detonateAllSpecials() {
    this.beginPhase();
    let n = 0, d = 0;
    for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
      const p = this.grid[r][c].piece;
      if (p && p.special !== SPECIAL.NONE && !p.clearing) {
        this.markClear(r, c, d);
        d += 90; n++;
      }
    }
    return n;
  }
}

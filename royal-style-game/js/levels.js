/* ==========================================================================
   levels.js — تعريف المراحل
   ---------------------------------------------------------------------------
   رموز المخطّط (حرف واحد لكل خليّة):
     .  خليّة عادية
     #  فراغ (لا توجد خليّة — القطع لا تمرّ منه)
     b  صندوق بطبقة واحدة        B  صندوق بطبقتين
     i  جليد بطبقة واحدة         I  جليد بطبقتين
     k  سلسلة بطبقة واحدة        K  سلسلة بطبقتين
     g  عشب بطبقة واحدة          G  عشب بطبقتين
     _  خليّة تجميع (يُجمع فيها الغرض الملكي الهابط)
   كل صفوف المخطّط يجب أن تكون بنفس الطول.
   ========================================================================== */
'use strict';

const Levels = (function () {

  /* اختصار لبناء مرحلة */
  function L(o) { return o; }

  const HANDMADE = [

    /* ---------- 1-3: تعريف بالمطابقة الأساسية ---------- */
    L({
      moves: 20, colors: 4,
      goals: [{ type: GOAL.COLOR, color: 0, count: 18 }],
      stars: [1500, 3000, 4500],
      layout: [
        '.......',
        '.......',
        '.......',
        '.......',
        '.......',
        '.......',
        '.......',
      ],
    }),

    L({
      moves: 22, colors: 4,
      goals: [
        { type: GOAL.COLOR, color: 1, count: 20 },
        { type: GOAL.COLOR, color: 3, count: 20 },
      ],
      stars: [2000, 4000, 6000],
      layout: [
        '........',
        '........',
        '........',
        '........',
        '........',
        '........',
        '........',
        '........',
      ],
    }),

    L({
      moves: 24, colors: 5,
      goals: [
        { type: GOAL.COLOR, color: 0, count: 22 },
        { type: GOAL.COLOR, color: 2, count: 22 },
      ],
      stars: [2500, 5000, 7500],
      layout: [
        '#.......#',
        '.........',
        '.........',
        '.........',
        '.........',
        '.........',
        '.........',
        '#.......#',
      ],
    }),

    /* ---------- 4-6: الصناديق ---------- */
    L({
      moves: 22, colors: 5,
      goals: [{ type: GOAL.BOX, count: 12 }],
      stars: [2200, 4400, 6600],
      layout: [
        '.........',
        '.........',
        '..b...b..',
        '.........',
        '..b.b.b..',
        '.........',
        '..b...b..',
        '.........',
        '.........',
      ],
    }),

    L({
      moves: 24, colors: 5,
      goals: [
        { type: GOAL.BOX, count: 16 },
        { type: GOAL.COLOR, color: 4, count: 20 },
      ],
      stars: [2600, 5200, 7800],
      layout: [
        '.........',
        '.bbb.bbb.',
        '.b.....b.',
        '.........',
        '....B....',
        '.........',
        '.b.....b.',
        '.bbb.bbb.',
        '.........',
      ],
    }),

    L({
      moves: 26, colors: 5,
      goals: [{ type: GOAL.BOX, count: 24 }],
      stars: [3000, 6000, 9000],
      layout: [
        '..bbbbb..',
        '..b...b..',
        '.........',
        'b.......b',
        'b...B...b',
        'b.......b',
        '.........',
        '..b...b..',
        '..bbbbb..',
      ],
    }),

    /* ---------- 7-9: الجليد ---------- */
    L({
      moves: 22, colors: 5,
      goals: [{ type: GOAL.ICE, count: 16 }],
      stars: [2400, 4800, 7200],
      layout: [
        '.........',
        '.........',
        '..iiiii..',
        '..i...i..',
        '..i.i.i..',
        '..i...i..',
        '..iiiii..',
        '.........',
        '.........',
      ],
    }),

    L({
      moves: 24, colors: 5,
      goals: [
        { type: GOAL.ICE, count: 20 },
        { type: GOAL.BOX, count: 8 },
      ],
      stars: [2800, 5600, 8400],
      layout: [
        'II.....II',
        'I.......I',
        '....b....',
        '...b.b...',
        '..b...b..',
        '...b.b...',
        '....b....',
        'I.......I',
        'II.....II',
      ],
    }),

    L({
      moves: 26, colors: 6,
      goals: [{ type: GOAL.ICE, count: 30 }],
      stars: [3200, 6400, 9600],
      layout: [
        'iiiiiiiii',
        'i.......i',
        'i.IIIII.i',
        'i.I...I.i',
        'i.I...I.i',
        'i.I...I.i',
        'i.IIIII.i',
        'i.......i',
        'iiiiiiiii',
      ],
    }),

    /* ---------- 10-12: السلاسل ---------- */
    L({
      moves: 22, colors: 5,
      goals: [{ type: GOAL.CHAIN, count: 15 }],
      stars: [2400, 4800, 7200],
      layout: [
        '.........',
        '.........',
        '.kkk.kkk.',
        '.........',
        '....k....',
        '.........',
        '.kkk.kkk.',
        '.........',
        '.........',
      ],
    }),

    L({
      moves: 25, colors: 5,
      goals: [
        { type: GOAL.CHAIN, count: 18 },
        { type: GOAL.COLOR, color: 2, count: 25 },
      ],
      stars: [3000, 6000, 9000],
      layout: [
        '#.......#',
        '..KKKKK..',
        '..K...K..',
        '..K...K..',
        '..K...K..',
        '..K...K..',
        '..KKKKK..',
        '#.......#',
      ],
    }),

    L({
      moves: 26, colors: 6,
      goals: [
        { type: GOAL.CHAIN, count: 20 },
        { type: GOAL.ICE, count: 12 },
      ],
      stars: [3400, 6800, 10200],
      layout: [
        'kk.....kk',
        'k.iiiii.k',
        '..i...i..',
        '..i...i..',
        'k.i...i.k',
        '..i...i..',
        '..i...i..',
        'k.iiiii.k',
        'kk.....kk',
      ],
    }),

    /* ---------- 13-15: العشب ---------- */
    L({
      moves: 22, colors: 5,
      goals: [{ type: GOAL.GRASS, count: 21 }],
      stars: [2400, 4800, 7200],
      layout: [
        '.........',
        '.........',
        '.........',
        '..ggggg..',
        '..ggggg..',
        '..ggggg..',
        '.........',
        '.........',
        '.........',
      ],
    }),

    L({
      moves: 25, colors: 5,
      goals: [
        { type: GOAL.GRASS, count: 24 },
        { type: GOAL.BOX, count: 10 },
      ],
      stars: [3000, 6000, 9000],
      layout: [
        'ggg...ggg',
        'gg.....gg',
        'g...b...g',
        '..b...b..',
        '....B....',
        '..b...b..',
        'g...b...g',
        'gg.....gg',
        'ggg...ggg',
      ],
    }),

    L({
      moves: 28, colors: 6,
      goals: [{ type: GOAL.GRASS, count: 33 }],
      stars: [3600, 7200, 10800],
      layout: [
        'GGG...GGG',
        'G.......G',
        'G..ggg..G',
        '...g.g...',
        '...ggg...',
        '...g.g...',
        'G..ggg..G',
        'G.......G',
        'GGG...GGG',
      ],
    }),

    /* ---------- 16-18: إنزال الأغراض الملكية ---------- */
    L({
      moves: 24, colors: 5, items: 2,
      goals: [{ type: GOAL.ITEM, count: 2 }],
      stars: [2600, 5200, 7800],
      layout: [
        '.........',
        '.........',
        '.........',
        '.........',
        '.........',
        '.........',
        '.........',
        '.........',
        '..._.._..',
      ],
    }),

    L({
      moves: 26, colors: 5, items: 3,
      goals: [
        { type: GOAL.ITEM, count: 3 },
        { type: GOAL.BOX, count: 12 },
      ],
      stars: [3200, 6400, 9600],
      layout: [
        '.........',
        '..b...b..',
        '.........',
        '.b.....b.',
        '.........',
        '..b...b..',
        '.........',
        '...bbb...',
        '.._..._..',
      ],
    }),

    L({
      moves: 28, colors: 6, items: 3,
      goals: [
        { type: GOAL.ITEM, count: 3 },
        { type: GOAL.ICE, count: 18 },
      ],
      stars: [3600, 7200, 10800],
      layout: [
        '#.......#',
        '.iiiiiii.',
        '.i.....i.',
        '.i.....i.',
        '.i.....i.',
        '.i.....i.',
        '.iiiiiii.',
        '#.......#',
        '#._..._.#',
      ],
    }),

    /* ---------- 19-24: خلطات وأشكال ---------- */
    L({
      moves: 26, colors: 5,
      goals: [
        { type: GOAL.BOX, count: 14 },
        { type: GOAL.GRASS, count: 16 },
      ],
      stars: [3400, 6800, 10200],
      layout: [
        '##.....##',
        '#.......#',
        '..bbbbb..',
        '.gg...gg.',
        '.gg.B.gg.',
        '.gg...gg.',
        '..bbbbb..',
        '#.......#',
        '##.....##',
      ],
    }),

    L({
      moves: 28, colors: 6,
      goals: [
        { type: GOAL.CHAIN, count: 16 },
        { type: GOAL.GRASS, count: 20 },
        { type: GOAL.COLOR, color: 5, count: 25 },
      ],
      stars: [4000, 8000, 12000],
      layout: [
        'g.......g',
        '.k.....k.',
        '..g...g..',
        '...k.k...',
        'gg..G..gg',
        '...k.k...',
        '..g...g..',
        '.k.....k.',
        'g.......g',
      ],
    }),

    L({
      moves: 26, colors: 5, items: 4,
      goals: [{ type: GOAL.ITEM, count: 4 }],
      stars: [3000, 6000, 9000],
      layout: [
        '.........',
        '.bbbbbbb.',
        '.........',
        '.b.....b.',
        '.........',
        '.b.....b.',
        '.........',
        '.bb...bb.',
        '._.._.._.',
      ],
    }),

    L({
      moves: 30, colors: 6,
      goals: [
        { type: GOAL.ICE, count: 26 },
        { type: GOAL.BOX, count: 16 },
      ],
      stars: [4200, 8400, 12600],
      layout: [
        '.........',
        'II.....II',
        'b.iiiii.b',
        'b.i...i.b',
        'b.i.B.i.b',
        'b.i...i.b',
        'b.iiiii.b',
        'II.....II',
        'bbb...bbb',
      ],
    }),

    L({
      moves: 30, colors: 6,
      goals: [
        { type: GOAL.GRASS, count: 28 },
        { type: GOAL.CHAIN, count: 20 },
      ],
      stars: [4400, 8800, 13200],
      layout: [
        'GGGG.GGGG',
        'G..K.K..G',
        'G.k...k.G',
        'Gk.....kG',
        '....g....',
        'Gk.....kG',
        'G.k...k.G',
        'G..K.K..G',
        'GGGG.GGGG',
      ],
    }),

    L({
      moves: 32, colors: 6, items: 4,
      goals: [
        { type: GOAL.ITEM, count: 4 },
        { type: GOAL.BOX, count: 20 },
        { type: GOAL.ICE, count: 16 },
      ],
      stars: [5000, 10000, 15000],
      layout: [
        'iiiiiiiii',
        '.bb...bb.',
        '.........',
        'b..III..b',
        'b..I.I..b',
        'b..III..b',
        '.........',
        '.bb...bb.',
        '_.._.._.._'.slice(0, 9),
      ],
    }),
  ];

  /* ======================= توليد المراحل بعد المصنوعة يدوياً ======================= */
  /* توليد ثابت ببذرة مشتقّة من رقم المرحلة: نفس المرحلة تعطي نفس اللوحة دائماً */
  function generate(levelNo) {
    const rnd = U.rng(levelNo * 7919 + 13);
    const rows = 9, cols = 9;
    const diff = Math.min(1, (levelNo - 25) / 60);     // 0 .. 1
    const colors = levelNo < 40 ? 5 : (rnd() < 0.5 ? 5 : 6);
    const moves = Math.max(18, Math.round(30 - diff * 8 + (rnd() * 4 - 2)));

    const grid = [];
    for (let r = 0; r < rows; r++) grid.push(new Array(cols).fill('.'));

    /* شكل اللوحة: إمّا كاملة، أو بأركان محذوفة، أو معيّن */
    const shape = Math.floor(rnd() * 3);
    if (shape === 1) {
      for (const [r, c] of [[0, 0], [0, 8], [8, 0], [8, 8]]) grid[r][c] = '#';
    } else if (shape === 2) {
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        if (Math.abs(r - 4) + Math.abs(c - 4) > 6) grid[r][c] = '#';
      }
    }

    /* اختيار العوائق */
    const pool = ['b', 'i', 'k', 'g'];
    U.shuffle(pool);
    const kinds = pool.slice(0, 1 + Math.floor(rnd() * 2 + diff));
    const density = 0.14 + diff * 0.18;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (grid[r][c] !== '.') continue;
        if (rnd() < density) {
          let ch = kinds[Math.floor(rnd() * kinds.length)];
          if (rnd() < diff * 0.45) ch = ch.toUpperCase();   // طبقتان
          /* لا تُغلق الصفوف العليا بالكامل حتى لا تُخنق مولّدات القطع */
          if (r === 0 && (ch === 'b' || ch === 'B')) ch = '.';
          grid[r][c] = ch;
        }
      }
    }

    /* أهداف مشتقّة من العوائق الموجودة فعلاً */
    const counts = { b: 0, i: 0, k: 0, g: 0 };
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const ch = grid[r][c].toLowerCase();
      if (counts[ch] != null) counts[ch] += (grid[r][c] === grid[r][c].toUpperCase() ? 2 : 1);
    }

    const goals = [];
    if (counts.b) goals.push({ type: GOAL.BOX, count: counts.b });
    if (counts.i) goals.push({ type: GOAL.ICE, count: counts.i });
    if (counts.k) goals.push({ type: GOAL.CHAIN, count: counts.k });
    if (counts.g) goals.push({ type: GOAL.GRASS, count: counts.g });

    let items = 0;
    if (rnd() < 0.28 && shape !== 2) {
      items = 2 + Math.floor(rnd() * 2 + diff);
      goals.push({ type: GOAL.ITEM, count: items });
      /* خانات التجميع في آخر صف صالح من كل عمود مختار */
      const slots = U.shuffle([1, 2, 3, 4, 5, 6, 7]).slice(0, Math.min(3, items));
      for (const c of slots) {
        for (let r = rows - 1; r >= 0; r--) {
          if (grid[r][c] === '.') { grid[r][c] = '_'; break; }
        }
      }
    }

    if (!goals.length || rnd() < 0.4) {
      const col = Math.floor(rnd() * colors);
      goals.push({ type: GOAL.COLOR, color: col, count: 20 + Math.round(diff * 25) });
    }

    const base = 2500 + Math.round(diff * 4000) + goals.length * 800;
    return {
      moves, colors, items,
      goals,
      stars: [base, base * 2, base * 3],
      layout: grid.map(r => r.join('')),
      generated: true,
    };
  }

  /* العدد الكلي المعروض على الخريطة */
  const TOTAL = 200;

  /* ======================= تنقيح المرحلة والتحقّق منها ======================= */
  /* يضمن أمرين لا تصحّ المرحلة بدونهما:
       (أ) أعلى خليّة موجودة في كل عمود حرّة، وإلّا انسدّ مصدر القطع الجديدة.
       (ب) عدد كل هدف لا يتجاوز ما هو موجود فعلاً على اللوحة، وإلّا استحال الفوز. */
  const HP = { b: 1, B: 2, i: 1, I: 2, k: 1, K: 2, g: 1, G: 2 };
  const FAMILY = { b: 'box', B: 'box', i: 'ice', I: 'ice', k: 'chain', K: 'chain', g: 'grass', G: 'grass' };
  const GOAL_FAMILY = { [GOAL.BOX]: 'box', [GOAL.ICE]: 'ice', [GOAL.CHAIN]: 'chain', [GOAL.GRASS]: 'grass' };

  function sanitize(def) {
    const grid = def.layout.map(r => r.split(''));
    const rows = grid.length, cols = grid[0].length;

    /* (أ) فتح أعلى خليّة في كل عمود — العشب مسموح لأنه تحت القطعة ولا يمنع النزول */
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        const ch = grid[r][c];
        if (ch === '#') continue;
        if ('bBiIkK'.indexOf(ch) >= 0) grid[r][c] = '.';
        break;
      }
    }

    /* (ب) حساب السعة الفعلية لكل نوع عائق */
    const cap = { box: 0, ice: 0, chain: 0, grass: 0 };
    let collectors = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const ch = grid[r][c];
        if (FAMILY[ch]) cap[FAMILY[ch]] += HP[ch];
        if (ch === '_') collectors++;
      }
    }

    def.goals = def.goals.filter(g => {
      const fam = GOAL_FAMILY[g.type];
      if (fam) {
        g.count = Math.min(g.count, cap[fam]);
        return g.count > 0;
      }
      if (g.type === GOAL.ITEM) {
        g.count = Math.min(g.count, def.items || 0);
        return g.count > 0 && collectors > 0;
      }
      return true;
    });

    /* لا يجوز أن تبقى المرحلة بلا هدف */
    if (!def.goals.length) {
      def.goals.push({ type: GOAL.COLOR, color: 0, count: 20 });
    }

    def.layout = grid.map(r => r.join(''));

    /* (ج) عتبات النجوم تُحسب من "نقاط المعيار" حتى تبقى عادلة في كل المراحل:
           هدف اللون أرخص من كسر عائق، وإنزال غرض هو الأغلى، مع وزن للحركات. */
    const W = { [GOAL.COLOR]: 110, [GOAL.BOX]: 230, [GOAL.ICE]: 230,
                [GOAL.CHAIN]: 230, [GOAL.GRASS]: 230, [GOAL.ITEM]: 900 };
    const par = def.goals.reduce((a, g) => a + g.count * (W[g.type] || 150), 0) + def.moves * 260;
    def.par = Math.round(par);
    def.stars = [Math.round(par * 0.55), Math.round(par * 0.90), Math.round(par * 1.30)];
    return def;
  }

  /**
   * يُرجع نسخة قابلة للتعديل من تعريف المرحلة رقم levelNo (يبدأ من 1).
   */
  function get(levelNo) {
    let def;
    if (levelNo <= HANDMADE.length) {
      def = JSON.parse(JSON.stringify(HANDMADE[levelNo - 1]));
    } else {
      def = generate(levelNo);
    }
    def.id = levelNo;
    def.items = def.items || 0;
    def.colors = U.clamp(def.colors || 5, 3, PIECE_TYPES.length);
    def.rows = def.layout.length;
    def.cols = def.layout[0].length;
    /* فحص سلامة المخطّط */
    for (const row of def.layout) {
      if (row.length !== def.cols) {
        console.warn('مخطّط المرحلة ' + levelNo + ' غير متساوي الصفوف');
      }
    }
    return sanitize(def);
  }

  return { get, TOTAL, HANDMADE_COUNT: HANDMADE.length, generate, sanitize };
})();

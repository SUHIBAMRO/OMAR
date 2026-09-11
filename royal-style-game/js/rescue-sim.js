/* ==========================================================================
   rescue-sim.js — محاكاة «مهمّات الإنقاذ» (سحب الدبابيس)
   ---------------------------------------------------------------------------
   محرّك خلايا (falling-sand): كل مادّة تنزل وتنتشر حسب قواعدها البسيطة،
   والتفاعلات بين المواد هي ما يصنع اللغز:
       الحمم + الماء  -> صخر         الحمم + الأفعى -> تحترق الأفعى
       الحمم + الملك  -> خسارة        الأفعى + الملك -> خسارة
       الذهب + الملك  -> يُجمع        الأفعى + الذهب -> تبتلعه
   هذا الملف لا يعتمد على المتصفّح إطلاقاً، فيمكن التحقّق من قابلية حلّ كل
   مرحلة بتشغيله في Node.
   ========================================================================== */
'use strict';

const MAT = {
  EMPTY: 0,
  WALL:  1,
  PIN:   2,
  LAVA:  3,
  WATER: 4,
  GOLD:  5,
  SNAKE: 6,
  KING:  7,
  STONE: 8,
};

/* المواد المتحرّكة: سائلة تنتشر جانبياً، حبيبية تنزل فقط */
const LIQUID = { [MAT.LAVA]: 1, [MAT.WATER]: 1 };
const GRAIN  = { [MAT.GOLD]: 1, [MAT.SNAKE]: 1 };

const ART_CHAR = {
  '.': MAT.EMPTY,
  ' ': MAT.EMPTY,
  '#': MAT.WALL,
  'L': MAT.LAVA,
  'W': MAT.WATER,
  'G': MAT.GOLD,
  'S': MAT.SNAKE,
  'K': MAT.KING,
  'O': MAT.STONE,
};

class RescueSim {

  /**
   * @param {object} def  { art:[string], scale:int, time:seconds, name:string,
   *                        solution:[pinIds] }
   */
  constructor(def) {
    this.def = def;
    this.scale = def.scale || 3;
    this.artW = def.art[0].length;
    this.artH = def.art.length;
    this.W = this.artW * this.scale;
    this.H = this.artH * this.scale;

    this.grid = new Uint8Array(this.W * this.H);
    this.pinAt = new Uint8Array(this.W * this.H);   // رقم الدبّوس + 1
    this.moved = new Uint8Array(this.W * this.H);
    /* اتجاه الجريان المحفوظ لكل خليّة سائلة: بدونه تتمايل السوائل في مكانها
       ولا تُصرَّف أبداً من الأسطح المستوية */
    this.flow = new Int8Array(this.W * this.H);

    this.pins = [];        // { id, cells:[[x,y]], ax, ay, aw, ah, dir, pulled }
    this.events = [];      // أحداث للعرض: burn / steam / collect / eat
    this.tick = 0;
    this.time = def.time || 90;
    this.status = 'playing';
    this.collected = 0;
    this.goalMet = false;      // جُمع ما يكفي من الذهب
    this.stillSteps = 0;       // كم خطوة مرّت بلا سقوط

    this.build();
    /* لا يُشترط جمع كل حبّة: بعض الحبيبات تستقرّ خارج متناول الملك دائماً،
       فالهدف نسبة من المجموع ما لم تحدّد المرحلة غير ذلك. */
    this.goldTotal = this.countMat(MAT.GOLD);
    this.goldTarget = def.goldTarget || Math.max(1, Math.ceil(this.goldTotal * 0.75));
  }

  idx(x, y) { return y * this.W + x; }
  inside(x, y) { return x >= 0 && y >= 0 && x < this.W && y < this.H; }
  get(x, y) { return this.inside(x, y) ? this.grid[y * this.W + x] : MAT.WALL; }
  set(x, y, m) { if (this.inside(x, y)) this.grid[y * this.W + x] = m; }

  countMat(m) {
    let n = 0;
    for (let i = 0; i < this.grid.length; i++) if (this.grid[i] === m) n++;
    return n;
  }

  /* -------------------------------------------------------------- */
  /*                            البناء                               */
  /* -------------------------------------------------------------- */

  build() {
    const s = this.scale;
    const pinCells = {};

    for (let ay = 0; ay < this.artH; ay++) {
      const row = this.def.art[ay];
      for (let ax = 0; ax < this.artW; ax++) {
        const ch = row[ax] || '.';
        let mat = ART_CHAR[ch];
        let pinId = 0;
        if (mat === undefined) {
          if (ch >= '1' && ch <= '9') { mat = MAT.PIN; pinId = +ch; }
          else mat = MAT.EMPTY;
        }
        for (let dy = 0; dy < s; dy++) {
          for (let dx = 0; dx < s; dx++) {
            const x = ax * s + dx, y = ay * s + dy;
            this.grid[this.idx(x, y)] = mat;
            if (pinId) this.pinAt[this.idx(x, y)] = pinId;
          }
        }
        if (pinId) {
          (pinCells[pinId] = pinCells[pinId] || []).push([ax, ay]);
        }
      }
    }

    /* تجميع الدبابيس وحساب امتداد كل واحد بإحداثيات الرسم */
    for (const id in pinCells) {
      const cells = pinCells[id];
      let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1;
      for (const [x, y] of cells) {
        x0 = Math.min(x0, x); y0 = Math.min(y0, y);
        x1 = Math.max(x1, x); y1 = Math.max(y1, y);
      }
      this.pins.push({
        id: +id, cells,
        ax: x0, ay: y0, aw: x1 - x0 + 1, ah: y1 - y0 + 1,
        dir: (x1 - x0) >= (y1 - y0) ? 'h' : 'v',
        pulled: false, anim: 0,
      });
    }
    this.pins.sort((a, b) => a.id - b.id);
  }

  /* -------------------------------------------------------------- */
  /*                          سحب الدبّوس                            */
  /* -------------------------------------------------------------- */

  pull(id) {
    const pin = this.pins.find(p => p.id === id);
    if (!pin || pin.pulled || this.status !== 'playing') return false;
    pin.pulled = true;
    const s = this.scale;
    for (const [ax, ay] of pin.cells) {
      for (let dy = 0; dy < s; dy++) {
        for (let dx = 0; dx < s; dx++) {
          const i = this.idx(ax * s + dx, ay * s + dy);
          if (this.grid[i] === MAT.PIN) this.grid[i] = MAT.EMPTY;
          this.pinAt[i] = 0;
        }
      }
    }
    this.events.push({ type: 'pull', pin });
    return true;
  }

  /* -------------------------------------------------------------- */
  /*                         خطوة فيزياء                             */
  /* -------------------------------------------------------------- */

  /**
   * خطوة واحدة. تُرجع true إذا سقط شيء (الانتشار الجانبي وحده لا يُعدّ حركة،
   * وإلّا بقيت السوائل تتمايل إلى الأبد ولم تستقرّ اللوحة أبداً).
   */
  step() {
    if (this.status !== 'playing') return false;
    this.tick++;
    this.moved.fill(0);
    let anyMove = false;

    for (let y = this.H - 2; y >= 0; y--) {
      const ltr = ((this.tick + y) & 1) === 0;
      for (let k = 0; k < this.W; k++) {
        const x = ltr ? k : this.W - 1 - k;
        const i = this.idx(x, y);
        const m = this.grid[i];
        if (m < MAT.LAVA || m > MAT.SNAKE) continue;   // LAVA..SNAKE فقط
        if (this.moved[i]) continue;
        if (this.tryMove(x, y, m) === 1) anyMove = true;
      }
    }

    this.react();
    if (this.status !== 'playing') return anyMove;

    /* الفوز لا يُعلَن إلا بعد أن تهدأ اللوحة: قد يكون الذهب وصل للملك
       بينما الحمم ما زالت في طريقها إليه. */
    this.stillSteps = anyMove ? 0 : this.stillSteps + 1;
    if (this.goalMet && this.stillSteps > 40) this.status = 'won';
    return anyMove;
  }

  /** 1 = سقوط، 2 = انتشار جانبي، 0 = لم يتحرّك */
  tryMove(x, y, m) {
    /* 1) نزول مباشر */
    if (this.get(x, y + 1) === MAT.EMPTY) { this.swap(x, y, x, y + 1); return 1; }

    /* 2) انزلاق قطري */
    const first = ((this.tick + x) & 1) ? 1 : -1;
    for (const d of [first, -first]) {
      if (this.get(x + d, y + 1) === MAT.EMPTY && this.get(x + d, y) !== MAT.WALL) {
        this.swap(x, y, x + d, y + 1);
        return 1;
      }
    }

    /* 3) انتشار جانبي — للسوائل فقط، مع الاستمرار في نفس الاتجاه ما أمكن */
    if (LIQUID[m]) {
      const keep = this.flow[this.idx(x, y)] || first;
      for (const d of [keep, -keep]) {
        if (this.get(x + d, y) === MAT.EMPTY) {
          this.swap(x, y, x + d, y);
          this.flow[this.idx(x + d, y)] = d;
          return 2;
        }
      }
      /* انسدّ الطريقان: انسَ الاتجاه حتى يُعاد اختياره */
      this.flow[this.idx(x, y)] = 0;
    }
    return 0;
  }

  swap(x1, y1, x2, y2) {
    const a = this.idx(x1, y1), b = this.idx(x2, y2);
    this.grid[b] = this.grid[a];
    this.grid[a] = MAT.EMPTY;
    this.flow[b] = this.flow[a];
    this.flow[a] = 0;
    this.moved[b] = 1;
    return true;
  }

  /* -------------------------------------------------------------- */
  /*                          التفاعلات                              */
  /* -------------------------------------------------------------- */

  react() {
    const N = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    /* النار تمتدّ قطرياً أيضاً، وإلّا نجت خليّة أفعى واحدة في زاوية
       وأتلفت كل الذهب المارّ بجانبها */
    const N8 = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]];
    let goldLeft = 0;

    for (let y = 0; y < this.H; y++) {
      for (let x = 0; x < this.W; x++) {
        const i = this.idx(x, y);
        const m = this.grid[i];
        if (m === MAT.EMPTY || m === MAT.WALL || m === MAT.PIN || m === MAT.STONE) continue;
        if (m === MAT.GOLD) goldLeft++;

        const ring = m === MAT.LAVA ? N8 : N;
        for (const [dx, dy] of ring) {
          const nx = x + dx, ny = y + dy;
          if (!this.inside(nx, ny)) continue;
          const j = this.idx(nx, ny);
          const n = this.grid[j];

          if (m === MAT.LAVA) {
            /* الملك والماء بالجوار المباشر فقط حتى لا تكون الخسارة ظالمة */
            if ((n === MAT.KING || n === MAT.WATER) && dx !== 0 && dy !== 0) continue;
            if (n === MAT.WATER) {
              this.grid[i] = MAT.STONE;
              this.grid[j] = MAT.EMPTY;
              this.events.push({ type: 'steam', x, y });
              break;
            }
            if (n === MAT.SNAKE) {
              this.grid[j] = MAT.EMPTY;
              this.events.push({ type: 'burn', x: nx, y: ny });
            } else if (n === MAT.KING) {
              this.lose('lava');
              return;
            }
          } else if (m === MAT.SNAKE) {
            if (n === MAT.KING) { this.lose('snake'); return; }
            if (n === MAT.GOLD) {
              this.grid[j] = MAT.EMPTY;
              goldLeft--;
              this.events.push({ type: 'eat', x: nx, y: ny });
            }
          } else if (m === MAT.GOLD) {
            if (n === MAT.KING) {
              this.grid[i] = MAT.EMPTY;
              this.collected++;
              goldLeft--;
              this.events.push({ type: 'collect', x, y });
              break;
            }
          }
        }
      }
    }

    this.goldLeft = goldLeft;
    this.goalMet = this.collected >= this.goldTarget;
    if (!this.goalMet && this.collected + goldLeft < this.goldTarget) this.lose('gold');
  }

  lose(reason) {
    if (this.status !== 'playing') return;
    this.status = 'lost';
    this.reason = reason;
  }

  /** تقدّم الزمن؛ يُرجع false إذا نفد الوقت */
  advanceTime(dt) {
    if (this.status !== 'playing') return true;
    this.time -= dt;
    if (this.time <= 0) { this.time = 0; this.lose('time'); return false; }
    return true;
  }

  /** يشغّل الخطوات حتى تستقرّ اللوحة أو ينتهي الحدّ الأقصى */
  settle(maxSteps) {
    maxSteps = maxSteps || 400;
    let still = 0;
    for (let i = 0; i < maxSteps; i++) {
      if (this.status !== 'playing') return i;
      const moved = this.step();
      still = moved ? 0 : still + 1;
      if (still > 12) return i;
    }
    return maxSteps;
  }

  /** نسبة الذهب المجموع */
  progress() { return this.goldTarget ? this.collected / this.goldTarget : 1; }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { RescueSim, MAT };
}

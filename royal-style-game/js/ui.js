/* ==========================================================================
   ui.js — كل الشاشات والنوافذ: الخريطة، الأهداف، المعزّزات، الفوز/الخسارة،
            المتجر، الإعدادات، ولوح القواعد.
   ========================================================================== */
'use strict';

const UI = (function () {

  let bannerTimer = null;
  let modalStack = [];

  /* ------------------------------------------------------------------ */
  /*                             الشاشات                                 */
  /* ------------------------------------------------------------------ */

  function show(id) {
    U.$$('.screen').forEach(s => s.classList.toggle('active', s.id === id));
    if (id === 'screen-map') { buildMap(); startLivesTicker(); }
    if (id === 'screen-game') setTimeout(() => Game.resize(), 30);
  }

  /* ------------------------------------------------------------------ */
  /*                          الشريط العلوي                              */
  /* ------------------------------------------------------------------ */

  let livesTimer = null;
  function startLivesTicker() {
    if (livesTimer) clearInterval(livesTimer);
    refreshTop();
    livesTimer = setInterval(refreshTop, 1000);
  }

  function refreshTop() {
    const lc = U.$('#lives-count'), lt = U.$('#lives-timer');
    if (lc) lc.textContent = Save.lives();
    if (lt) {
      const s = Save.nextLifeIn();
      lt.textContent = s > 0 ? U.mmss(s) : 'ممتلئة';
    }
    const cc = U.$('#coins-count');
    if (cc) cc.textContent = U.fmt(Save.coins());
    const st = U.$('#stars-count');
    if (st) st.textContent = Save.totalStars();
  }

  /* ------------------------------------------------------------------ */
  /*                          خريطة المراحل                              */
  /* ------------------------------------------------------------------ */

  function buildMap() {
    const inner = U.$('#map-inner');
    if (!inner) return;
    const maxLv = Save.maxLevel();
    const count = Math.min(Levels.TOTAL, maxLv + 12);
    const spacing = 104;
    const height = count * spacing + 140;
    inner.style.height = height + 'px';
    inner.innerHTML = '';

    const pos = i => ({
      x: 50 + Math.sin(i * 0.78) * 26,           // نسبة مئوية
      y: height - 90 - i * spacing,
    });

    /* المسار بين العقد */
    for (let i = 0; i < count - 1; i++) {
      const a = pos(i), b = pos(i + 1);
      const ax = a.x / 100 * inner.clientWidth || a.x / 100 * window.innerWidth;
      const bx = b.x / 100 * inner.clientWidth || b.x / 100 * window.innerWidth;
      const dx = bx - ax, dy = b.y - a.y;
      const len = Math.sqrt(dx * dx + dy * dy);
      const ang = Math.atan2(dy, dx) * 180 / Math.PI - 90;
      const seg = U.el('div', 'map-path');
      seg.style.height = len + 'px';
      seg.style.left = a.x + '%';
      seg.style.top = a.y + 'px';
      seg.style.transform = 'translateX(-50%) rotate(' + ang + 'deg)';
      inner.appendChild(seg);
    }

    /* العقد */
    for (let i = 0; i < count; i++) {
      const lv = i + 1;
      const p = pos(i);
      const node = U.el('button', 'map-node');
      const stars = Save.starsFor(lv);
      if (lv < maxLv) node.classList.add('done');
      else if (lv === maxLv) node.classList.add('current');
      else node.classList.add('locked');
      node.textContent = lv;
      node.style.left = p.x + '%';
      node.style.top = p.y + 'px';
      if (lv <= maxLv) {
        const sw = U.el('span', 'node-stars');
        for (let k = 0; k < 3; k++) sw.appendChild(U.el('i', k < stars ? 'on' : ''));
        node.appendChild(sw);
        node.onclick = () => { Sfx.click(); openLevel(lv); };
      }
      inner.appendChild(node);
    }

    /* التمرير إلى المرحلة الحالية */
    const scroll = U.$('#map-scroll');
    requestAnimationFrame(() => {
      const cur = pos(maxLv - 1);
      scroll.scrollTop = Math.max(0, cur.y - scroll.clientHeight * 0.6);
    });
  }

  /* ------------------------------------------------------------------ */
  /*                            النوافذ                                  */
  /* ------------------------------------------------------------------ */

  function openModal(buildFn, opts) {
    opts = opts || {};
    const root = U.$('#modal-root');
    root.classList.add('open');
    root.innerHTML = '';
    const back = U.el('div', 'backdrop');
    const dlg = U.el('div', 'dialog');
    if (opts.closable !== false) {
      const x = U.el('button', 'close-x', '✕');
      x.onclick = () => { Sfx.click(); closeModal(); };
      dlg.appendChild(x);
      back.onclick = () => { Sfx.click(); closeModal(); };
    }
    buildFn(dlg);
    root.appendChild(back);
    root.appendChild(dlg);
    modalStack.push(buildFn);
    return dlg;
  }

  function closeModal() {
    const root = U.$('#modal-root');
    root.classList.remove('open');
    root.innerHTML = '';
    modalStack = [];
    refreshTop();
  }

  function btn(label, cls, onClick, costCoins) {
    const b = U.el('button', 'btn ' + (cls || ''));
    b.innerHTML = label;
    if (costCoins != null) {
      const s = U.el('span', 'cost');
      s.innerHTML = '<span class="ico ico-coin"></span>' + U.fmt(costCoins);
      b.appendChild(s);
      if (Save.coins() < costCoins) b.disabled = true;
    }
    b.onclick = () => { Sfx.click(); onClick(); };
    return b;
  }

  /* ------------------------------------------------------------------ */
  /*                        أيقونات الأهداف                              */
  /* ------------------------------------------------------------------ */

  function goalSprite(goal) {
    switch (goal.type) {
      case GOAL.COLOR: return 'p' + goal.color;
      case GOAL.BOX:   return 'box1';
      case GOAL.ICE:   return 'ice1';
      case GOAL.CHAIN: return 'chain1';
      case GOAL.GRASS: return 'grass1';
      case GOAL.ITEM:  return 'item';
      default: return 'p0';
    }
  }

  /** اسم مختصر للهدف (يُستعمل في قائمة ما تبقّى) */
  function goalShort(goal) {
    switch (goal.type) {
      case GOAL.COLOR: return PIECE_TYPES[goal.color].name;
      case GOAL.BOX:   return 'صندوق';
      case GOAL.ICE:   return 'طبقة جليد';
      case GOAL.CHAIN: return 'سلسلة';
      case GOAL.GRASS: return 'بلاطة عشب';
      case GOAL.ITEM:  return 'صندوق ملكي';
      default: return '';
    }
  }

  function goalLabel(goal) {
    switch (goal.type) {
      case GOAL.COLOR: return 'اجمع ' + goal.count + ' من ' + PIECE_TYPES[goal.color].name;
      case GOAL.BOX:   return 'حطّم ' + goal.count + ' صندوقاً';
      case GOAL.ICE:   return 'اكسر ' + goal.count + ' طبقة جليد';
      case GOAL.CHAIN: return 'اكسر ' + goal.count + ' سلسلة';
      case GOAL.GRASS: return 'نظّف ' + goal.count + ' بلاطة عشب';
      case GOAL.ITEM:  return 'أنزِل ' + goal.count + ' من الصناديق الملكية';
      default: return '';
    }
  }

  /* ------------------------------------------------------------------ */
  /*                        شريط الأهداف داخل اللعب                      */
  /* ------------------------------------------------------------------ */

  let goalEls = [];
  function buildGoals(def, board) {
    const box = U.$('#goals-box');
    box.innerHTML = '';
    goalEls = [];
    for (const g of def.goals) {
      const el = U.el('div', 'goal');
      el.appendChild(Art.iconEl(goalSprite(g), 34));
      const b = U.el('b', '', String(g.count));
      el.appendChild(b);
      el.title = goalLabel(g);
      box.appendChild(el);
      goalEls.push({ el, b, goal: g, last: -1 });
    }
  }

  function updateGoals(def, board) {
    for (const ge of goalEls) {
      const left = Math.max(0, ge.goal.count - board.goalProgress(ge.goal));
      if (left === ge.last) continue;
      if (ge.last !== -1) {
        ge.el.classList.remove('bump');
        void ge.el.offsetWidth;
        ge.el.classList.add('bump');
        if (left === 0) Sfx.goal();
      }
      ge.last = left;
      ge.b.textContent = left === 0 ? '' : left;
      ge.el.classList.toggle('done', left === 0);
    }
  }

  /* ------------------------------------------------------------------ */
  /*                          شريط المعزّزات                             */
  /* ------------------------------------------------------------------ */

  const BOOSTER_ORDER = ['hammer', 'glove', 'rocket', 'tnt', 'ball', 'shuffle'];

  function boosterSprite(k) {
    switch (k) {
      case 'rocket': return 'rh';
      case 'tnt': return 'tnt';
      case 'ball': return 'ball';
      default: return k;
    }
  }

  function buildBoosterBar() {
    const bar = U.$('#boosterbar');
    if (!bar) return;
    bar.innerHTML = '';
    for (const k of BOOSTER_ORDER) {
      const b = U.el('button', 'booster');
      b.appendChild(Art.iconEl(boosterSprite(k), 40));
      const n = Save.booster(k);
      const cnt = U.el('span', 'count' + (n > 0 ? '' : ' buy'), n > 0 ? String(n) : '+');
      b.appendChild(cnt);
      b.title = BOOSTER_INFO[k].name + ' — ' + BOOSTER_INFO[k].desc;
      if (Game.S.activeBooster === k) b.classList.add('active');
      b.onclick = () => { Sfx.click(); Game.armBooster(k); };
      bar.appendChild(b);
    }
  }

  /* ------------------------------------------------------------------ */
  /*                              لافتة                                  */
  /* ------------------------------------------------------------------ */

  function banner(text) {
    const el = U.$('#banner');
    if (!el || !text) return;
    el.textContent = text;
    el.classList.remove('show');
    void el.offsetWidth;
    el.classList.add('show');
    if (bannerTimer) clearTimeout(bannerTimer);
    bannerTimer = setTimeout(() => el.classList.remove('show'), 1200);
  }

  /* ------------------------------------------------------------------ */
  /*                       نافذة بدء المرحلة                             */
  /* ------------------------------------------------------------------ */

  const preChosen = { rocket: false, tnt: false, ball: false, moves: false };

  function openLevel(lv) {
    const def = Levels.get(lv);
    preChosen.rocket = preChosen.tnt = preChosen.ball = preChosen.moves = false;

    openModal(dlg => {
      dlg.appendChild(U.el('h2', '', 'المرحلة ' + lv));

      const stars = Save.starsFor(lv);
      const sr = U.el('div', 'stars-row');
      for (let i = 0; i < 3; i++) {
        const s = U.el('i', (i === 1 ? 'mid ' : '') + (i < stars ? 'on' : ''));
        sr.appendChild(s);
      }
      dlg.appendChild(sr);

      dlg.appendChild(U.el('p', '', 'الأهداف:'));
      const gp = U.el('div', 'goal-preview');
      for (const g of def.goals) {
        const d = U.el('div', 'gp');
        d.appendChild(Art.iconEl(goalSprite(g), 34));
        d.appendChild(U.el('b', '', String(g.count)));
        d.title = goalLabel(g);
        gp.appendChild(d);
      }
      dlg.appendChild(gp);

      dlg.appendChild(U.el('p', '', 'الحركات: <b>' + def.moves + '</b>' +
        (Save.bestFor(lv) ? ' — أفضل نتيجة: <b>' + U.fmt(Save.bestFor(lv)) + '</b>' : '')));

      /* معزّزات ما قبل البدء */
      dlg.appendChild(U.el('p', '', 'معزّزات البداية (اختياري):'));
      const list = U.el('div', 'prelist');
      const items = [
        { k: 'rocket', sp: 'rh', lbl: 'صاروخ' },
        { k: 'tnt', sp: 'tnt', lbl: 'قنبلة' },
        { k: 'ball', sp: 'ball', lbl: 'كرة' },
        { k: 'moves', sp: 'moves', lbl: '+5 حركات' },
      ];
      for (const it of items) {
        const d = U.el('div', 'prebooster');
        d.appendChild(Art.iconEl(it.sp, 44));
        d.appendChild(U.el('span', 'lbl', it.lbl));
        const cost = CFG.economy.preBoosterCost[it.k];
        const pr = U.el('div', 'price');
        pr.innerHTML = '<span class="ico ico-coin"></span>' + cost;
        d.appendChild(pr);
        d.onclick = () => {
          if (!preChosen[it.k] && Save.coins() < totalPreCost() + cost) {
            Sfx.invalid(); banner('العملات لا تكفي'); return;
          }
          preChosen[it.k] = !preChosen[it.k];
          d.classList.toggle('on', preChosen[it.k]);
          Sfx.click();
        };
        list.appendChild(d);
      }
      dlg.appendChild(list);

      dlg.appendChild(btn('ابدأ', 'green', () => {
        const cost = totalPreCost();
        if (cost > 0 && !Save.spend(cost)) { banner('العملات لا تكفي'); return; }
        if (Save.lives() <= 0) { showNoLives(); return; }
        Save.useLife();
        closeModal();
        show('screen-game');
        Game.start(lv, Object.assign({}, preChosen));
      }));
      const c = totalPreCost();
      if (c > 0) dlg.appendChild(U.el('p', 'hint-note', 'تكلفة المعزّزات المختارة: ' + c + ' عملة'));
      dlg.appendChild(U.el('p', 'hint-note', 'بدء المرحلة يستهلك قلباً واحداً.'));
    });
  }

  function totalPreCost() {
    let t = 0;
    for (const k in preChosen) if (preChosen[k]) t += CFG.economy.preBoosterCost[k];
    return t;
  }

  /* ------------------------------------------------------------------ */
  /*                         الفوز والخسارة                              */
  /* ------------------------------------------------------------------ */

  function showWin(res) {
    openModal(dlg => {
      dlg.appendChild(U.el('h2', '', 'أحسنت!'));
      const sr = U.el('div', 'stars-row');
      for (let i = 0; i < 3; i++) sr.appendChild(U.el('i', (i === 1 ? 'mid ' : '') + (i < res.stars ? 'on' : '')));
      dlg.appendChild(sr);
      dlg.appendChild(U.el('p', '', 'النقاط: <b>' + U.fmt(res.score) + '</b>'));
      dlg.appendChild(U.el('p', '', 'مكافأة: <b>' + res.coins + '</b> عملة'));

      dlg.appendChild(btn('المرحلة التالية', 'green', () => {
        const next = res.level + 1;
        closeModal();
        if (next > Levels.TOTAL) { show('screen-map'); return; }
        if (Save.lives() <= 0) { show('screen-map'); showNoLives(); return; }
        openLevel(next);
      }));
      dlg.appendChild(btn('الخريطة', '', () => { closeModal(); Game.stop(); show('screen-map'); }));
    }, { closable: false });
  }

  function showLose(res) {
    openModal(dlg => {
      dlg.appendChild(U.el('h2', '', 'نفدت الحركات'));
      const left = [];
      for (const g of res.board.def.goals) {
        const n = Math.max(0, g.count - res.board.goalProgress(g));
        if (n > 0) left.push(n + ' × ' + goalShort(g));
      }
      dlg.appendChild(U.el('p', '', 'بقي لك: ' + (left.join('، ') || 'لا شيء')));

      const gp = U.el('div', 'goal-preview');
      for (const g of res.board.def.goals) {
        const n = Math.max(0, g.count - res.board.goalProgress(g));
        const d = U.el('div', 'gp');
        d.appendChild(Art.iconEl(goalSprite(g), 34));
        d.appendChild(U.el('b', '', n === 0 ? '✔' : String(n)));
        gp.appendChild(d);
      }
      dlg.appendChild(gp);

      dlg.appendChild(btn('+' + CFG.economy.extraMovesAmount + ' حركات ومتابعة', 'gold', () => {
        if (!Save.spend(CFG.economy.extraMovesCost)) { banner('العملات لا تكفي'); return; }
        closeModal();
        Game.continueWithMoves();
      }, CFG.economy.extraMovesCost));

      dlg.appendChild(btn('إعادة المحاولة', '', () => {
        closeModal();
        if (Save.lives() <= 0) { Game.stop(); show('screen-map'); showNoLives(); return; }
        Save.useLife();
        Game.start(res.level, null);
        refreshTop();
      }));
      dlg.appendChild(btn('الخروج إلى الخريطة', 'red', () => {
        closeModal(); Game.stop(); show('screen-map');
      }));
    }, { closable: false });
  }

  /* ------------------------------------------------------------------ */
  /*                            القلوب                                   */
  /* ------------------------------------------------------------------ */

  function showNoLives() {
    openModal(dlg => {
      dlg.appendChild(U.el('h2', '', 'لا توجد قلوب'));
      dlg.appendChild(U.el('p', '', 'يتجدّد قلب واحد كل ' +
        Math.round(CFG.lives.refillMs / 60000) + ' دقيقة.'));
      const t = U.el('p', '', '');
      dlg.appendChild(t);
      const tick = () => {
        const s = Save.nextLifeIn();
        t.innerHTML = 'القلب التالي بعد: <b>' + (s > 0 ? U.mmss(s) : 'الآن') + '</b>';
        if (Save.lives() > 0) { clearInterval(iv); closeModal(); }
      };
      const iv = setInterval(tick, 1000); tick();

      dlg.appendChild(btn('ملء القلوب', 'gold', () => {
        if (!Save.spend(CFG.lives.refillCost)) { banner('العملات لا تكفي'); return; }
        Save.fillLives();
        clearInterval(iv);
        closeModal();
        refreshTop();
      }, CFG.lives.refillCost));
      dlg.appendChild(btn('حسناً', '', () => { clearInterval(iv); closeModal(); }));
    });
  }

  /* ------------------------------------------------------------------ */
  /*                        شراء معزّز أثناء اللعب                       */
  /* ------------------------------------------------------------------ */

  function showBuyBooster(key) {
    const cost = CFG.economy.boosterCost[key];
    openModal(dlg => {
      dlg.appendChild(U.el('h2', '', BOOSTER_INFO[key].name));
      const ic = Art.iconEl(boosterSprite(key), 64);
      ic.style.margin = '6px auto'; ic.style.display = 'block';
      dlg.appendChild(ic);
      dlg.appendChild(U.el('p', '', BOOSTER_INFO[key].desc));
      dlg.appendChild(btn('شراء واحد', 'gold', () => {
        if (!Save.spend(cost)) { banner('العملات لا تكفي'); return; }
        Save.addBooster(key, 1);
        closeModal();
        buildBoosterBar();
      }, cost));
      dlg.appendChild(btn('إلغاء', '', closeModal));
    });
  }

  /* ------------------------------------------------------------------ */
  /*                             المتجر                                  */
  /* ------------------------------------------------------------------ */

  const BONUS_MS = 8 * 60 * 60 * 1000;

  function showShop() {
    openModal(dlg => {
      dlg.appendChild(U.el('h2', '', 'المتجر'));
      dlg.appendChild(U.el('p', 'hint-note',
        'لا توجد إعلانات ولا مشتريات حقيقية — كل شيء يُشترى بالعملات التي تكسبها من اللعب.'));

      /* مكافأة دورية */
      const last = Save.flag('lastBonus') || 0;
      const ready = Date.now() - last >= BONUS_MS;
      const bonusRow = U.el('div', 'shop-item');
      bonusRow.innerHTML = '<div class="big">🎁</div><div class="info"><b>الهدية الدورية</b>' +
        '<small>' + (ready ? 'جاهزة الآن' : 'بعد ' + U.mmss((BONUS_MS - (Date.now() - last)) / 1000)) +
        '</small></div>';
      const bb = btn('+150', 'green', () => {
        if (Date.now() - (Save.flag('lastBonus') || 0) < BONUS_MS) { banner('لم يحن وقتها بعد'); return; }
        Save.addCoins(150);
        Save.setFlag('lastBonus', Date.now());
        Sfx.coin();
        closeModal(); showShop();
      });
      bb.disabled = !ready;
      bonusRow.appendChild(bb);
      dlg.appendChild(bonusRow);

      /* قلوب */
      const lifeRow = U.el('div', 'shop-item');
      lifeRow.innerHTML = '<div class="big">❤️</div><div class="info"><b>ملء القلوب</b>' +
        '<small>يملأ القلوب الخمسة دفعة واحدة</small></div>';
      const lb = btn(String(CFG.lives.refillCost), 'gold', () => {
        if (Save.lives() >= CFG.lives.max) { banner('القلوب ممتلئة'); return; }
        if (!Save.spend(CFG.lives.refillCost)) { banner('العملات لا تكفي'); return; }
        Save.fillLives(); Sfx.coin(); closeModal(); showShop(); refreshTop();
      });
      lifeRow.appendChild(lb);
      dlg.appendChild(lifeRow);

      /* المعزّزات */
      for (const k of BOOSTER_ORDER) {
        const row = U.el('div', 'shop-item');
        const ic = Art.iconEl(boosterSprite(k), 40);
        row.appendChild(ic);
        const info = U.el('div', 'info');
        info.innerHTML = '<b>' + BOOSTER_INFO[k].name + ' (تملك ' + Save.booster(k) + ')</b>' +
          '<small>' + BOOSTER_INFO[k].desc + '</small>';
        row.appendChild(info);
        const cost = CFG.economy.boosterCost[k];
        row.appendChild(btn(String(cost), 'gold', () => {
          if (!Save.spend(cost)) { banner('العملات لا تكفي'); return; }
          Save.addBooster(k, 1); Sfx.coin();
          closeModal(); showShop(); buildBoosterBar();
        }));
        dlg.appendChild(row);
      }
    });
  }

  /* ------------------------------------------------------------------ */
  /*                            الإعدادات                                */
  /* ------------------------------------------------------------------ */

  function toggleRow(label, flagKey, onChange) {
    const row = U.el('div', 'toggle-row');
    row.appendChild(U.el('span', '', label));
    const sw = U.el('div', 'switch' + (Save.flag(flagKey) ? ' on' : ''));
    sw.onclick = () => {
      const v = !Save.flag(flagKey);
      Save.setFlag(flagKey, v);
      sw.classList.toggle('on', v);
      Sfx.click();
      if (onChange) onChange(v);
    };
    row.appendChild(sw);
    return row;
  }

  function showSettings() {
    openModal(dlg => {
      dlg.appendChild(U.el('h2', '', 'الإعدادات'));
      dlg.appendChild(toggleRow('المؤثّرات الصوتية', 'sound', v => Sfx.setEnabled(v)));
      dlg.appendChild(toggleRow('الموسيقى', 'music', v => Sfx.setMusic(v)));
      dlg.appendChild(toggleRow('التلميحات التلقائية', 'hints'));
      dlg.appendChild(U.el('p', 'hint-note',
        'التقدّم محفوظ في هذا المتصفّح فقط. لا إعلانات ولا تتبّع ولا اتصال بالإنترنت.'));
      dlg.appendChild(btn('القواعد الكاملة', '', () => { closeModal(); showRules(); }));
      dlg.appendChild(btn('تصفير كل التقدّم', 'red', () => {
        openModal(d2 => {
          d2.appendChild(U.el('h2', '', 'تأكيد'));
          d2.appendChild(U.el('p', '', 'سيُحذف كل التقدّم والعملات والنجوم. لا يمكن التراجع.'));
          d2.appendChild(btn('نعم، صفّر', 'red', () => {
            Save.reset(); closeModal(); show('screen-map'); refreshTop();
          }));
          d2.appendChild(btn('إلغاء', '', () => { closeModal(); showSettings(); }));
        });
      }));
    });
  }

  /* ------------------------------------------------------------------ */
  /*                          لوح القواعد                                */
  /* ------------------------------------------------------------------ */

  function tileRow(spriteName, html) {
    const row = U.el('div', 'tile-row');
    row.appendChild(Art.iconEl(spriteName, 38));
    row.appendChild(U.el('span', '', html));
    return row;
  }

  function showRules() {
    openModal(dlg => {
      dlg.appendChild(U.el('h2', '', 'قواعد اللعبة'));
      const box = U.el('div', 'rules');

      box.appendChild(U.el('h3', '', '١) الأساس'));
      box.appendChild(U.el('p', '',
        'بدّل قطعتين متجاورتين لتصنع خطاً من ثلاث قطع متشابهة أو أكثر. ' +
        'كل تبديل ناجح يستهلك حركة واحدة. التبديل الذي لا يصنع تطابقاً يُرجَع ولا يُحتسب.'));

      box.appendChild(U.el('h3', '', '٢) القطع الخاصّة'));
      box.appendChild(tileRow('rh', '<b>الصاروخ</b> — من مطابقة <b>٤</b> في خط. يمسح الصفّ كاملاً (أو العمود بحسب اتجاه المطابقة).'));
      box.appendChild(tileRow('tnt', '<b>القنبلة</b> — من مطابقة على شكل <b>L</b> أو <b>T</b>. تفجّر ما حولها في دائرة نصف قطرها خانتان.'));
      box.appendChild(tileRow('ball', '<b>الكرة المضيئة</b> — من مطابقة <b>٥</b> في خط. بدّلها مع أي قطعة لتمسح كل قطع لونها.'));

      box.appendChild(U.el('h3', '', '٣) الدمج بين القطع الخاصّة'));
      const combos = U.el('ul');
      [['صاروخ + صاروخ', 'يمسح الصفّ والعمود معاً (صليب كامل).'],
       ['صاروخ + قنبلة', 'صليب عريض بعرض ثلاث خانات في كل اتجاه.'],
       ['قنبلة + قنبلة', 'انفجار ضخم نصف قطره ثلاث خانات.'],
       ['كرة + صاروخ', 'يتحوّل كل قطع لون واحد إلى صواريخ تنطلق كلّها.'],
       ['كرة + قنبلة', 'يتحوّل كل قطع لون واحد إلى قنابل تنفجر كلّها.'],
       ['كرة + كرة', 'تمسح اللوحة بالكامل.'],
      ].forEach(([a, b]) => {
        const li = U.el('li', '', '<b>' + a + ':</b> ' + b);
        combos.appendChild(li);
      });
      box.appendChild(combos);

      box.appendChild(U.el('h3', '', '٤) العوائق'));
      box.appendChild(tileRow('box1', '<b>الصندوق</b> — لا يتحرّك ولا يُبدَّل. يُكسر بمطابقة <b>ملاصقة</b> له أو بانفجار يصيبه. بعضها بطبقتين.'));
      box.appendChild(tileRow('ice1', '<b>الجليد</b> — يُجمّد القطعة فلا يمكن تحريكها. اكسره بمطابقة القطعة نفسها ضمن خط، أو بانفجار.'));
      box.appendChild(tileRow('chain1', '<b>السلسلة</b> — مثل الجليد: تقيّد القطعة حتى تُكسر. الطبقتان تحتاجان ضربتين.'));
      box.appendChild(tileRow('grass1', '<b>العشب</b> — يقع تحت القطع. يُنظَّف حين تُمسح القطعة التي فوقه.'));
      box.appendChild(tileRow('item', '<b>الصندوق الملكي</b> — ينزل مع الجاذبية. أفرِغ ما تحته حتى يصل إلى خانة التجميع في الأسفل.'));
      box.appendChild(tileRow('collector', '<b>خانة التجميع</b> — النقطة التي يجب أن يصل إليها الصندوق الملكي.'));

      box.appendChild(U.el('h3', '', '٥) المعزّزات'));
      for (const k of BOOSTER_ORDER) {
        box.appendChild(tileRow(boosterSprite(k), '<b>' + BOOSTER_INFO[k].name + '</b> — ' + BOOSTER_INFO[k].desc));
      }
      box.appendChild(U.el('p', '', 'استعمال المعزّز <b>لا</b> يستهلك حركة. يمكن أيضاً اختيار معزّز يُزرع على اللوحة قبل بدء المرحلة.'));

      box.appendChild(U.el('h3', '', '٦) النقاط والنجوم'));
      const sc = U.el('ul');
      [['قطعة عادية', CFG.score.piece + ' نقطة'],
       ['كسر طبقة عائق', CFG.score.obstacleHit + ' نقطة'],
       ['صنع قطعة خاصّة', CFG.score.specialCreate + ' نقطة'],
       ['دمج قطعتين خاصّتين', CFG.score.comboCreate + ' نقطة'],
       ['كل حركة متبقّية عند الفوز', CFG.score.leftoverMove + ' نقطة + صاروخ مجاني'],
      ].forEach(([a, b]) => sc.appendChild(U.el('li', '', '<b>' + a + ':</b> ' + b)));
      box.appendChild(sc);
      box.appendChild(U.el('p', '',
        'الموجات المتتابعة (cascades) تضاعف النقاط تدريجياً حتى <b>×' +
        CFG.score.cascadeMul[CFG.score.cascadeMul.length - 1] + '</b>. ' +
        'النجوم الثلاث تُمنح حسب مجموع النقاط مقارنةً بمعيار المرحلة.'));

      box.appendChild(U.el('h3', '', '٧) القلوب والعملات'));
      box.appendChild(U.el('p', '',
        'كل محاولة تستهلك قلباً، والحدّ الأقصى ' + CFG.lives.max + ' قلوب، ويتجدّد قلب كل ' +
        Math.round(CFG.lives.refillMs / 60000) + ' دقيقة. ' +
        'تُكسب العملات من الفوز والنجوم والهدية الدورية، وتُصرف على المعزّزات والحركات الإضافية والقلوب.'));

      box.appendChild(U.el('h3', '', '٨) حالات خاصّة'));
      const sp = U.el('ul');
      ['إذا لم تبقَ أي حركة ممكنة تُخلَط اللوحة تلقائياً دون استهلاك حركة.',
       'عند تحقيق كل الأهداف تتحوّل الحركات المتبقّية إلى صواريخ تنفجر دفعةً واحدة.',
       'إذا نفدت الحركات قبل إتمام الأهداف يمكن شراء ٥ حركات إضافية بالعملات ومتابعة المرحلة نفسها.',
       'النقر مرّتين على قطعة خاصّة يفعّلها في مكانها ويستهلك حركة.',
      ].forEach(t => sp.appendChild(U.el('li', '', t)));
      box.appendChild(sp);

      dlg.appendChild(box);
      dlg.appendChild(btn('فهمت', 'green', closeModal));
    });
  }

  /* ------------------------------------------------------------------ */
  /*                          تأكيد الخروج                               */
  /* ------------------------------------------------------------------ */

  function confirmQuit() {
    openModal(dlg => {
      dlg.appendChild(U.el('h2', '', 'الخروج من المرحلة؟'));
      dlg.appendChild(U.el('p', '', 'سيُحتسب القلب المستهلَك ولن يُسترجع.'));
      dlg.appendChild(btn('نعم، اخرج', 'red', () => {
        closeModal(); Game.giveUp(); show('screen-map');
      }));
      dlg.appendChild(btn('متابعة اللعب', 'green', closeModal));
    });
  }

  return {
    show, refreshTop, buildMap, openLevel,
    buildGoals, updateGoals, buildBoosterBar, banner,
    showWin, showLose, showNoLives, showBuyBooster, showShop, showSettings, showRules,
    confirmQuit, openModal, closeModal, startLivesTicker, goalLabel, goalShort, goalSprite,
  };
})();

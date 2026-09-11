/* ==========================================================================
   main.js — التشغيل وربط الأزرار
   ========================================================================== */
'use strict';

(function () {

  function boot() {
    Save.init();
    Sfx.setEnabled(Save.flag('sound'));
    if (Save.flag('music')) Sfx.setMusic(true);

    /* أزرار الخريطة */
    U.$('#btn-play').onclick = () => {
      Sfx.click();
      if (Save.lives() <= 0) { UI.showNoLives(); return; }
      UI.openLevel(Save.maxLevel());
    };
    U.$('#btn-shop').onclick = () => { Sfx.click(); UI.showShop(); };
    U.$('#btn-how').onclick = () => { Sfx.click(); UI.showRules(); };
    U.$('#btn-settings').onclick = () => { Sfx.click(); UI.showSettings(); };
    U.$('#res-coins').onclick = () => { Sfx.click(); UI.showShop(); };
    U.$('#res-lives').onclick = () => { Sfx.click(); if (Save.lives() < CFG.lives.max) UI.showNoLives(); };
    U.$('#btn-quit').onclick = () => { Sfx.click(); UI.confirmQuit(); };

    /* أول تشغيل: اعرض القواعد مرّة واحدة */
    setTimeout(() => {
      UI.show('screen-map');
      if (!Save.flag('seenRules')) {
        Save.setFlag('seenRules', true);
        UI.showRules();
      }
    }, 1200);

    /* تفعيل الصوت عند أول لمسة (سياسة المتصفّحات) */
    const unlock = () => { Sfx.resume(); window.removeEventListener('pointerdown', unlock); };
    window.addEventListener('pointerdown', unlock);

    /* إعادة القياس */
    let rt = null;
    window.addEventListener('resize', () => {
      clearTimeout(rt);
      rt = setTimeout(() => { Game.resize(); UI.buildMap(); }, 120);
    });
    window.addEventListener('orientationchange', () => setTimeout(() => Game.resize(), 250));

    /* إيقاف الصوت عند إخفاء الصفحة */
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && Sfx.isMusicOn()) Sfx.setMusic(false);
      else if (!document.hidden && Save.flag('music')) Sfx.setMusic(true);
      if (!document.hidden) UI.refreshTop();
    });

    /* منع التكبير بالإصبعين داخل اللوحة */
    document.addEventListener('gesturestart', e => e.preventDefault());
    document.addEventListener('dblclick', e => e.preventDefault(), { passive: false });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();

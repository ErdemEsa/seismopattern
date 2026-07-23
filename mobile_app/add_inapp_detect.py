# -*- coding: utf-8 -*-
"""Adds in-app browser detection banner to index.html"""
from pathlib import Path

INDEX = Path("web/index.html")
t = INDEX.read_text(encoding="utf-8")

DETECTION_SCRIPT = """
  <script>
    // In-app browser detection (LinkedIn, Instagram, Facebook, TikTok)
    (function() {
      const ua = navigator.userAgent || '';
      const isInApp =
        /LinkedInApp/i.test(ua) ||
        /Instagram/i.test(ua) ||
        /FBAN|FBAV/i.test(ua) ||
        /TikTok/i.test(ua) ||
        /Twitter/i.test(ua);

      if (!isInApp) return;

      // Uyari overlay olustur
      const overlay = document.createElement('div');
      overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:linear-gradient(135deg,#FF7043 0%,#F4511E 100%);z-index:99999;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;color:white;text-align:center;';

      overlay.innerHTML = [
        '<div style="width:80px;height:80px;border-radius:20px;background:rgba(255,255,255,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:20px;box-shadow:0 8px 30px rgba(0,0,0,0.2);">',
        '  <svg viewBox="0 0 24 24" style="width:48px;height:48px;fill:white;"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.94-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>',
        '</div>',
        '<h1 style="font-size:24px;font-weight:700;margin:0 0 8px;">SeismoPattern</h1>',
        '<p style="font-size:14px;opacity:0.85;margin:0 0 32px;">Kalibre olasiliksal risk izleme</p>',
        '<div style="background:rgba(255,255,255,0.15);border-radius:12px;padding:20px;max-width:340px;margin-bottom:20px;">',
        '  <p style="margin:0 0 16px;font-size:15px;line-height:1.5;font-weight:600;">',
        '    En iyi deneyim icin uygulamayi Chrome veya Safari ile acin',
        '  </p>',
        '  <p style="margin:0;font-size:13px;opacity:0.9;line-height:1.5;">',
        '    Sag ust menuden (...) <b>"Tarayicida ac"</b> secenegini kullanin',
        '  </p>',
        '</div>',
        '<button id="copy-link" style="background:white;color:#F4511E;border:none;padding:14px 28px;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;box-shadow:0 4px 12px rgba(0,0,0,0.15);">',
        '  🔗 Linki Kopyala',
        '</button>',
        '<p style="margin-top:24px;font-size:12px;opacity:0.75;max-width:280px;line-height:1.4;">',
        '  Bu uygulama Flutter Web tabanli oldugundan sosyal medya uygulamalari icinde tam calismaz.',
        '</p>',
      ].join('');

      document.body.appendChild(overlay);

      document.getElementById('copy-link').onclick = function() {
        const url = 'https://seismopattern.onrender.com';
        if (navigator.clipboard) {
          navigator.clipboard.writeText(url).then(function() {
            this.textContent = '✅ Kopyalandi!';
          }.bind(this));
        } else {
          const ta = document.createElement('textarea');
          ta.value = url;
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
          this.textContent = '✅ Kopyalandi!';
        }
      };

      // Flutter yuklenmesin
      window.stop && window.stop();
    })();
  </script>
"""

# </head>'den once ekle
if 'LinkedInApp' not in t:
    t = t.replace('</head>', DETECTION_SCRIPT + '</head>')
    INDEX.write_text(t, encoding='utf-8', newline='\n')
    print('[OK] In-app browser detection added')
else:
    print('Already patched')

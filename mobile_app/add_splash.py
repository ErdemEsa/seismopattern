# -*- coding: utf-8 -*-
"""Adds loading splash screen to Flutter web index.html"""
from pathlib import Path

WEB_DIR = Path("web")
INDEX = WEB_DIR / "index.html"

t = INDEX.read_text(encoding="utf-8")

# CSS + HTML splash screen
SPLASH_CSS = """
  <style>
    body {
      margin: 0;
      background: linear-gradient(135deg, #FF7043 0%, #F4511E 100%);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      overflow: hidden;
    }
    #splash {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 9999;
      transition: opacity 0.5s;
    }
    #splash.hidden {
      opacity: 0;
      pointer-events: none;
    }
    .splash-icon {
      width: 90px;
      height: 90px;
      border-radius: 20px;
      background: rgba(255,255,255,0.15);
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 24px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.2);
    }
    .splash-icon svg {
      width: 56px;
      height: 56px;
      fill: white;
    }
    .splash-title {
      color: white;
      font-size: 32px;
      font-weight: 700;
      margin: 0 0 8px 0;
      letter-spacing: -0.5px;
    }
    .splash-subtitle {
      color: rgba(255,255,255,0.85);
      font-size: 14px;
      margin: 0 0 40px 0;
    }
    .splash-spinner {
      width: 40px;
      height: 40px;
      border: 3px solid rgba(255,255,255,0.25);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
      margin-bottom: 16px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .splash-status {
      color: rgba(255,255,255,0.9);
      font-size: 13px;
      text-align: center;
      max-width: 320px;
      line-height: 1.5;
      padding: 0 20px;
    }
    .splash-hint {
      color: rgba(255,255,255,0.7);
      font-size: 11px;
      margin-top: 24px;
      text-align: center;
      padding: 0 20px;
    }
  </style>
"""

SPLASH_HTML = """
  <div id="splash">
    <div class="splash-icon">
      <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.94-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
    </div>
    <h1 class="splash-title">SeismoPattern</h1>
    <p class="splash-subtitle">v4.0 - Kalibre risk izleme</p>
    <div class="splash-spinner"></div>
    <p class="splash-status" id="splash-status">Uygulama yukleniyor...</p>
    <p class="splash-hint">Ilk acilis 30 saniyeye kadar surebilir</p>
  </div>
  <script>
    // Cold start uyari mesajlari
    const statusEl = document.getElementById('splash-status');
    const messages = [
      'Uygulama yukleniyor...',
      'Modeller hazirlaniyor...',
      'Deprem verileri cekiliyor...',
      'Neredeyse hazir...',
    ];
    let msgIdx = 0;
    const msgInterval = setInterval(() => {
      msgIdx = (msgIdx + 1) % messages.length;
      if (statusEl) statusEl.textContent = messages[msgIdx];
    }, 4000);

    // Flutter yuklendiginde splash'i gizle
    window.addEventListener('flutter-first-frame', () => {
      clearInterval(msgInterval);
      const splash = document.getElementById('splash');
      if (splash) {
        splash.classList.add('hidden');
        setTimeout(() => splash.remove(), 600);
      }
    });
  </script>
"""

# CSS'i </head>'den once ekle
if '#splash' not in t:
    t = t.replace('</head>', SPLASH_CSS + '</head>')
    print('[OK] CSS eklendi')
else:
    print('CSS zaten mevcut')

# HTML'i <body> icine hemen ekle
if '<div id="splash">' not in t:
    t = t.replace('<body>', '<body>' + SPLASH_HTML)
    print('[OK] HTML eklendi')
else:
    print('HTML zaten mevcut')

INDEX.write_text(t, encoding='utf-8', newline='\n')
print(f'[DONE] {INDEX} guncellendi')

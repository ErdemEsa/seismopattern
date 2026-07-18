# SeismoPattern

Kalibrated, cok katmanli, segment olcekli, olasiliksal deprem risk izleme sistemi.

**Deterministik deprem zaman tahmini degildir.**

---

## Sistem Nedir

SeismoPattern, dunya genelinde buyuk depremlerin (Mw 7.0+) oncesinde
gozlemlenen sismik oruntuleri tespit eden bir risk izleme sistemidir.

Sistem sunlari yapar:

- Segment olcekli uzun vadeli tehlike degerlendirmesi
- Kisa vadeli sismik anomali tespiti
- Cok katmanli risk skoru uretimi
- Olasiliksal horizon bazli tehlike hesabi

Sistem sunlari yapmaz:

- Belirli bir tarihte deprem olacagini soylememez
- Kesin uyari vermez
- Resmi erken uyari sisteminin yerini tutmaz

---

## Model Performans Metrikleri

OOS AUC-ROC (5-fold CV)   : 0.7054  CI: 0.6788 - 0.7330
PR-AUC                    : 0.9101
Brier Score               : 0.1791
ECE (isotonic sonrasi)    : 0.0181
Debiased benchmark AUC    : 0.9219
DeLong testi              : 6/6 baseline p<0.05

---

## Veri Kaynaklari

GCMT      : 1976-2025 deprem katalogu     (69944 olay)
ISC FDSN  : Canli bolgesel sismisitesi    (API)
USGS FDSN : Canli yedek kaynak            (API)
GEM       : Aktif fay segmentleri         (16195 segment)
MIDAS     : GPS hiz veritabani            (20168 istasyon)

---

## Hizli Baslangic

Docker ile:

    docker compose -f docker-compose.runtime.yml up

Yerel Python ile:

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    pip install --no-deps xgboost==3.3.0
    python app.py

Web arayuzu : http://127.0.0.1:5000
API docs    : http://127.0.0.1:5000/docs

---

## API Endpoint Ornekleri

    GET  /api/status          sistem durumu
    POST /api/realtime        canli bolge analizi
    GET  /api/dual_risk       dual risk framework
    GET  /api/hazard_table    25 bolge hazard tablosu
    GET  /api/hazard          horizon bazli tehlike
    GET  /api/geodynamic      tum katmanlar birlesik
    GET  /api/pdf             PDF rapor
    GET  /docs                Swagger UI

---

## Prospective Dogrulama

15 Temmuz 2026 tarihinden itibaren haftalik immutable tahmin kaydi yapilmaktadir.

Her tahmin SHA-256 ile hashlenip JSON arsive yazilir.
SQLite trigger ile degistirilemez hale getirilir.
Hash zinciri ile onceki kayitlara baglanir.

Ilk degerlendirme: Ekim 2026 (90 gun penceresi)

    python scripts/prospective_tracker.py --record
    python scripts/prospective_tracker.py --verify
    python scripts/prospective_tracker.py --status

---

## Bilimsel Belgeler

    docs/Limitations.md              sistem sinirlamalari
    docs/Scientific_Methodology.md   bilimsel metodoloji
    docs/Validation_Methodology.md   dogrulama yaklasimi
    docs/Feature_Definitions.md      feature tanimlari
    docs/Risk_Model.md               risk modeli
    docs/Calibration.md              kalibrasyon
    docs/DEPLOYMENT.md               kurulum ve calistirma

---

## Onemli Uyari

Bu sistem arastirma ve karar destek amaclidir.
Tahliye karari, resmi afet yonetimi veya muhendislik standardi
belirleme icin tek basina kullanilmamalidir.

Yuksek risk skoru yakin deprem garantisi degildir.
Dusuk risk skoru guvenlik garantisi degildir.

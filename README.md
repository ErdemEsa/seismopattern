# SeismoPattern

[![Backend](https://img.shields.io/badge/Backend-Live-brightgreen)](https://seismopattern.onrender.com/api/status)
[![Model AUC](https://img.shields.io/badge/AUC-0.8943-blue)]()
[![Zone Coverage](https://img.shields.io/badge/Zones-59-orange)]()
[![Tests](https://img.shields.io/badge/Tests-12%2F12-brightgreen)]()

Kalibrated, cok katmanli, segment olcekli, olasiliksal deprem risk izleme sistemi.

**Deterministik deprem zaman tahmini degildir.**

## Canli Demo

- Backend API : https://seismopattern.onrender.com
- Status      : https://seismopattern.onrender.com/api/status
- Zones       : https://seismopattern.onrender.com/api/zones
- Mobil       : Flutter (Android APK + web build hazir)

> Not: Backend Render Free Tier uzerinde calisir. 15 dakika inaktivite
> sonrasi uykuya gecer, ilk istek 30-60 saniye surebilir. Sonrasi hizlidir.

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

OOS AUC-ROC (5-fold CV)   : 0.8943
PR-AUC                    : 0.8971
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
    POST /api/realtime        canli bolgesel analiz
    GET  /api/dual_risk       dual risk framework
    GET  /api/hazard_table    watchlist hazard tablosu
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


---
## Operasyonel Kapsam

- Izlenen watchlist bolge sayisi: 50+ (current: 58)
- Prospective izleme: immutable hash zinciri ile kayit altinda
- Bootstrap uncertainty: 50 model x 3 tip = 150 ensemble model

Ek endpoint ornekleri:

    GET  /api/zones           zone veritabani
    GET  /api/dual_risk_table watchlist risk tablosu
    GET  /api/uncertainty     bootstrap belirsizlik

---

## Mobil Uygulama

Flutter tabanli mobil arayuz, backend API ile canli calisir.

### Ozellikler

- 5 sekme: Ana Sayfa (dashboard) / Zones / Harita / Analiz / Hakkinda
- 59 bolgelik dunya haritasi, risk seviyesine gore renkli marker'lar
- Zone detay: uncertainty (bootstrap), fay, coupling, slip deficit
- Manuel koordinat analizi (istediginiz lat/lon icin)
- Renkli risk seviyesi kartlari (KRITIK / YUKSEK / ORTA / DIKKAT / DUSUK)
- Ayarlar ekrani: backend URL degistirilebilir

### Calistirmak

Web (herhangi bir tarayici):

    cd mobile_app
    flutter run -d web-server --web-port 8080

Android emulator:

    cd mobile_app
    flutter run

Android APK build:

    cd mobile_app
    flutter build apk --release

Uretilen APK: mobile_app/build/app/outputs/flutter-apk/app-release.apk

### Ekran Goruntuleri

Ekran goruntuleri icin bakiniz: mobile_app/docs/screenshots/

---

## Deployment

### Backend (Render.com Free Tier)

- Docker image olarak deploy edilir
- Otomatik SSL (Let's Encrypt)
- GitHub push -> otomatik build
- Health check: /healthz
- CORS acik (mobil app icin)

### Frontend (Flutter web)

- Static site olarak Render'a deploy edilebilir
- Ya da bagimsiz her yerde host edilebilir
- Backend URL runtime'da degistirilebilir (Ayarlar ekrani)

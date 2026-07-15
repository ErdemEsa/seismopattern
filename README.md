# SeismoPattern

**Kalibre edilmiş, çok katmanlı, segment ölçekli olasılıksal deprem risk izleme sistemi**

> ⚠️ Bu sistem deterministik deprem zaman tahmini değildir.  
> Uzun vadeli segment tehlikesini kısa vadeli sismik anomaliden ayrıştıran, karar destek amaçlı bir risk izleme sistemidir.

---

## İçerik

- [Genel Tanım](#genel-tanım)
- [Öne Çıkan Özellikler](#öne-çıkan-özellikler)
- [Bilimsel Çerçeve](#bilimsel-çerçeve)
- [Model Performansı](#model-performansı)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Kurulum](#kurulum)
- [Gerekli Büyük Veri Dosyaları](#gerekli-büyük-veri-dosyaları)
- [Kullanım](#kullanım)
- [Web Arayüzü](#web-arayüzü)
- [API Endpoint'leri](#api-endpointleri)
- [Dual Risk Framework](#dual-risk-framework)
- [İzlenen 25 Bölge](#izlenen-25-bölge)
- [Script Özeti](#script-özeti)
- [Dosya Yapısı](#dosya-yapısı)
- [Bilinen Sınırlamalar](#bilinen-sınırlamalar)
- [Bilimsel Uyarı](#bilimsel-uyarı)
- [Lisans](#lisans)

---

## Genel Tanım

**SeismoPattern**, büyük depremlerin öncesindeki sismik örüntüleri, fay geometrisini, GPS deformasyonunu, Coulomb gerilim transferini ve tarihsel öncü kayıtları birleştirerek **segment ölçekli olasılıksal risk değerlendirmesi** yapan bir sistemdir.

Sistem iki temel soruyu ayırır:

1. **Uzun vadede bu segment ne kadar tehlikeli?**
2. **Kısa vadede anormal bir sismik davranış var mı?**

Bu iki sorunun cevabı birleştirilerek:

- **Long-term Segment Risk**
- **Short-term Seismic Anomaly Risk**
- **Combined Operational Risk**
- **30 gün / 90 gün / 1 yıl / 5 yıl horizon skorları**

üretilir.

---

## Öne Çıkan Özellikler

- **Dual Risk Framework**
  - Long-term segment hazard
  - Short-term anomaly risk
  - Combined operational risk

- **Quality-Adjusted Risk**
  - Debiased model
  - Veri kalitesi cezası
  - Tektonik prior
  - Stabil bölgeleri bastırma

- **Çok Katmanlı Jeodinamik Sistem**
  - ISC + USGS
  - GEM aktif fay veritabanı
  - MIDAS GPS velocity field
  - Basitleştirilmiş Coulomb stres
  - NLP tarihsel öncü veri tabanı

- **Web Uygulaması**
  - Manuel analiz
  - Canlı / tarihsel analiz
  - Harita
  - Uyarı paneli
  - Bölge paneli
  - Dual Risk paneli
  - Swagger dokümantasyonu

- **Otomasyon**
  - Prospective tracker
  - Global scan
  - Alert system
  - MLOps versiyonlama

---

## Bilimsel Çerçeve

Bu sistem **deprem tahmini** yapmaz. Bunun yerine:

- geçmiş örüntülerden öğrenilen
- istatistiksel olarak kalibre edilmiş
- olasılıksal risk skorları üretir

### Temel bilimsel katmanlar

- **GCMT katalogu** → büyük ölçekli global sismisite
- **ISC + USGS** → daha ince bölgesel katalog
- **GEM Faults** → fay yakınlığı ve segment karmaşıklığı
- **GPS / MIDAS** → deformasyon ve strain
- **CFF** → statik gerilim transferi
- **NLP** → tarihsel precursor kayıtları

---

## Model Performansı

### Standart Model
- **OOS AUC-ROC**: `0.7054`
- **95% CI**: `[0.6788, 0.7330]`
- **PR-AUC**: `0.9101`
- **Brier Score**: `0.1791`
- **ECE (Isotonic calibrated)**: `0.0181`

### Debiased Model
- Magnitude feature’ları çıkarılmıştır
- Bölge imzası etkisini azaltır
- **Independent benchmark AUC**: `0.9219`

### Quality-Adjusted Model
- Debiased skor
- Veri kalitesi cezası
- Tektonik prior
- Stabil bölgelerde yanlış yüksek skor üretimini bastırır

### Audit Sonuçları
- Label shuffle testi: ✅ geçti
- Time-split testi: ✅ geçti
- Leave-one-region-out: ✅ geçti
- DeLong significance: ✅ 6/6 baseline p<0.05
- Calibration: ✅ isotonic ile güçlü iyileşme

---

## Sistem Mimarisi

```text
                    Kullanıcı / API
                          |
                          v
                ┌──────────────────┐
                │ Flask Web App    │
                └──────────────────┘
                          |
                          v
              ┌────────────────────────┐
              │ Dual Risk Framework    │
              └────────────────────────┘
                  /                \
                 /                  \
                v                    v
   ┌─────────────────────┐   ┌─────────────────────┐
   │ Long-term Risk      │   │ Short-term Risk     │
   │ - slip deficit      │   │ - debiased model    │
   │ - coupling          │   │ - quality penalty   │
   │ - fault geometry    │   │ - tectonic prior    │
   │ - GPS strain        │   │ - b-value, accel    │
   └─────────────────────┘   └─────────────────────┘
                 \                    /
                  \                  /
                   v                v
                ┌──────────────────────┐
                │ Combined Risk +      │
                │ Hazard Horizons      │
                │ 30d / 90d / 1y / 5y  │
                └──────────────────────┘
# SeismoPattern — Risk Modeli

**Versiyon:** SeismoPattern v2  
**Son güncelleme:** Temmuz 2026

---

## 1. Genel yapı

SeismoPattern risk çıktısı tek bir modelden değil,
çok katmanlı bir çerçeveden üretilir.

Ana katmanlar:
1. Short-Term anomaly risk
2. Long-Term segment risk
3. Combined operational risk
4. Hazard horizons

---

## 2. Short-Term risk

Kısa vadeli risk, son 3 yıllık sismik örüntüden üretilir.

Temel kaynak:
- debiased model skoru

Düzeltmeler:
- quality penalty
- tectonic penalty

Operasyonel kısa vadeli skor:
ST_adjusted = debiased_score - quality_penalty - tectonic_penalty

Amaç:
ham kısa vadeli skoru veri kalitesi ve tektonik bağlamla düzeltmek.

---

## 3. Long-Term risk

Uzun vadeli risk segmentin jeodinamik durumunu temsil eder.

Bileşenler:
- slip deficit
- coupling ratio
- recurrence phase
- unruptured segment ratio
- fay yakınlığı / karmaşıklığı
- GPS tabanlı deformasyon bilgisi

Bu katman kısa dönem sismik anomaliden bağımsızdır.

---

## 4. Combined operational risk

Birleşik skor LT ve ST katmanlarını birlikte kullanır.

Genel mantık:
- LT yüksekse segment zaten yapısal olarak önemli kabul edilir
- ST yüksekse yakın dönem anomali ağırlığı artar
- CFF ve NLP gibi ek bilgiler boost olarak eklenebilir

Bu skor karar destek amaçlıdır, kesin olay tahmini değildir.

---

## 5. Risk seviyeleri

Combined score tipik olarak kategorilere ayrılır:
- düşük
- orta
- yüksek
- çok yüksek

Bu seviyeler karar desteği ve görselleştirme kolaylığı sağlar.

---

## 6. Hazard horizons

Horizon bazlı tehlike çıktıları:
- 30 gün
- 90 gün
- 1 yıl
- 5 yıl

Çerçeve:
Poisson tabanlı yaklaşık olasılık

Biçim:
hazard = 1 - exp(-lambda * t * multiplier)

Multiplier kaynakları:
- segment multiplier
- anomaly multiplier
- CFF
- NLP
- quality factor

---

## 7. Standard ve debiased model farkı

Standard model magnitude feature'larını içerir.
Debiased model bu feature'ları çıkarır.

Operasyonel yorum:
- standard model araştırma karşılaştırması için yararlıdır
- debiased model kısa vadeli operasyonel sinyal için daha güvenlidir

---

## 8. Kullanım ilkesi

Risk skoru:
- deprem olacak demek değildir
- segment önceliklendirmesi sağlar
- bölgeler arası göreli kıyas üretir
- zaman içinde artış/azalış izlemeye yarar

---

## 9. Çıktıların yorumu

Yüksek LT + düşük ST:
- yapısal olarak tehlikeli ama kısa dönem anomali zayıf

Düşük LT + yüksek ST:
- kısa dönem anomali var ama uzun dönem bağlam zayıf

Yüksek LT + yüksek ST:
- operasyonel olarak en dikkat çekici kombinasyon

Düşük LT + düşük ST:
- göreli olarak düşük öncelik

---

## 10. Sonuç

Risk modeli tek bir makine öğrenmesi skoru değil,
jeodinamik bağlamla desteklenmiş çok katmanlı
bir segment ölçekli risk çerçevesidir.

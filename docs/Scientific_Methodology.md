# SeismoPattern — Bilimsel Metodoloji

**Versiyon:** SeismoPattern v2  
**Son güncelleme:** Temmuz 2026

---

## 1. Amaç

SeismoPattern'in amacı, büyük depremlerden önce gözlenebilen
istatistiksel sismik örüntüleri tespit etmek ve bunları
segment ölçekli olasılıksal risk izlemesine dönüştürmektir.

Ana soru şudur:

Belirli segmentlerde, büyük depremlerden önce küçük ve orta büyüklükteki
depremlerin dağılımında tekrarlayan örüntüler var mı?

---

## 2. Genel yaklaşım

Sistem dört ana katmanda çalışır:

1. Tarihsel örüntü çıkarımı
2. Kısa vadeli sismik anomali modelleme
3. Uzun vadeli segment tehlikesi modelleme
4. Birleşik operasyonel risk ve horizon hazard üretimi

---

## 3. Veri kaynakları

### 3.1 GCMT

- 1976-2025
- 69944 olay
- tarihsel büyük deprem öncü pencereleri ve eğitim seti için kullanılır

### 3.2 ISC + USGS

- canlı / yakın gerçek zamanlı bölgesel sismisite
- ISC birincil kaynak
- USGS yedek / tamamlayıcı kaynak

### 3.3 GEM fay verisi

- 16195 aktif fay segmenti
- segment yakınlığı ve yapısal bağlam için kullanılır

### 3.4 MIDAS GPS

- 20168 istasyon
- yatay hız ve yaklaşık strain bilgisi üretir

---

## 4. Ön işleme

### 4.1 Declustering

Ham katalog artçılardan arındırılır.
Kullanılan yöntem: Gardner-Knopoff 1974

- ham katalog: 69944
- declustered katalog: 40729

### 4.2 Bölgesel normalizasyon

Farklı bölgelerin doğal arka plan sismisitesi farklı olduğu için
ham feature'lar z-skoru ve bağlamsal normalizasyon ile desteklenir.

---

## 5. Feature engineering

Model 3 yıllık pencere üzerinde çalışır:

- 0-1 yıl
- 1-2 yıl
- 2-3 yıl

Feature grupları:
- olay sayısı ve trend
- quiescence / aktivasyon metrikleri
- büyüklük istatistikleri
- b-değeri
- derinlik istatistikleri
- mekânsal odak ve göç
- normalize z-skor feature'ları

Toplam standard feature sayısı: 35

Debiased sürümde magnitude ağırlıklı 4 feature çıkarılır.

---

## 6. İki aşamalı model mimarisi

### 6.1 Aşama 1: tip sınıflandırması

Kural tabanlı ön sınıflandırma yapılır:

- TIP_A: aktivasyon örüntüsü
- TIP_B: sessizlik örüntüsü
- TIP_C: belirsiz örüntü

### 6.2 Aşama 2: tip bazlı risk modeli

Her tip için ayrı XGBoost modeli eğitilir.

Birleşik skor:
- yüzde 70 birincil tip modeli
- yüzde 30 ensemble katkısı

---

## 7. Kalibrasyon

Ham model olasılıkları isotonic regression ile kalibre edilir.

Ana ölçüler:
- ECE: 0.0181
- Brier Score: 0.1791

Bu, çıktıların operasyonel yorumlanabilirliğini artırır.

---

## 8. Debiased model yaklaşımı

Magnitude feature'larının bölgesel katalog imzası taşıdığı görülmüştür.
Bu nedenle ayrı bir debiased model geliştirilmiştir.

Çıkarılan feature'lar:
- w3_max_mw
- w1_max_mw
- w3_mean_mw
- w1_mean_mw

Sonuç:
- standard model benchmark AUC: 0.42
- debiased model benchmark AUC: 0.92

Bu nedenle kısa vadeli operasyonel anomali katmanında
debiased model önceliklidir.

---

## 9. Dual Risk Framework

### 9.1 Long-Term risk

Uzun vadeli segment riski şu bileşenlerden oluşur:
- slip deficit
- coupling
- recurrence phase
- unruptured segment ratio
- fay yakınlığı / karmaşıklığı
- GPS hız ve strain bilgisi

### 9.2 Short-Term risk

Kısa vadeli anomali riski:
- debiased model skoru
- quality penalty
- tectonic penalty

### 9.3 Combined risk

Operasyonel risk:
- LT + ST birleşimi
- gerektiğinde CFF boost
- gerektiğinde NLP boost

### 9.4 Hazard horizons

Üretilen horizon'lar:
- 30 gün
- 90 gün
- 1 yıl
- 5 yıl

---

## 10. Açıklanabilirlik

Sistem şu yorumlanabilirlik araçlarını kullanır:
- SHAP importance
- counterfactual analiz
- kalite düzeltilmiş risk ayrımı

Magnitude feature baskınlığı bu analizlerle tespit edilmiştir.

---

## 11. Prospective kayıt mantığı

15 Temmuz 2026 itibarıyla haftalık tahminler immutable şekilde kaydedilmektedir.

Her run için:
- archive JSON
- SHA-256 hash
- run hash chain
- SQLite trigger koruması

oluşturulur.

Bu yapı hindsight eleştirisini azaltmayı hedefler.

---

## 12. Bilimsel konum

SeismoPattern:
- deterministik deprem tahmini değildir
- fizik + veri hibrit risk izleme sistemidir
- olasılıksal segment izleme yaklaşımıdır
- prospective doğrulama ile değerlendirilecek araştırma yazılımıdır


## Fraktal Boyut Feature'? (fractal_dim_36m)

### Motivasyon
B?y?k depremler ?ncesinde episentral b?lgedeki sismisitede
mek?nsal k?melenme de?i?imi beklenebilir. Fraktal boyut
(box-counting y?ntemi) bu mek?nsal da??l?m? ?l?er.

### Hesaplama
- Son 36 ayl?k penceredeki olaylar?n lat/lon konumlar? al?n?r
- 2?2, 4?4, 8?8, 16?16 kutu gridleri uygulan?r
- log(N_kutular) vs log(kutu_boyutu) e?imi fraktal boyutu verir
- De?er aral???: [0, 2] (0=tek nokta, 2=tamamen d?zlemsel da??l?m)

### Veri gereksinimleri
- Minimum 8 olay (200 km, 36 ay)
- Koordinat kolonlar?: eff_lat/eff_lon ?ncelikli

### Sonu?lar

| Metrik | De?er |
|---|---|
| REAL NaN oran? | 0.52 |
| CTRL NaN oran? | 0.57 |
| REAL medyan? | 0.638 |
| CTRL medyan? | 0.598 |
| Mann-Whitney p | 0.00356 |
| Model etkisi (PR-AUC) | +0.0013 |
| Model etkisi (F1@0.30) | +0.0081 |

### Yorum
Real pencereler (b?y?k deprem ?ncesi) hafif daha y?ksek fraktal
boyut g?steriyor. Bu, sismisitenin control'e k?yasla daha geni?
alana yay?ld???na i?aret edebilir ? veya sismik k?mele?menin
belirli bir d?zeni bozdu?una.

Sinyal istatistiksel olarak anlaml? (p<0.05) ancak etki b?y?kl???
orta d?zeyde. Bu nedenle model i?inde tamamlay?c? bir feature
olarak tutulmaktad?r.

### K?s?tlamalar
- NaN oran? y?ksek (%52-57): seyrek katalogda s?n?rl? kullan?m
- Sadece 36m penceresi yeterli veri sa?l?yor (12m ?ok sparse)
- GCMT sparse katalog i?in optimize edilmemi? box boyutlar?


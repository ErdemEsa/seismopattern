#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import textwrap

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)


def write_doc(filename: str, content: str) -> None:
    path = DOCS_DIR / filename
    text = textwrap.dedent(content).strip() + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"Yazildi: {path} ({len(text.splitlines())} satir)")


LIMITATIONS = """
# SeismoPattern — Sistem Sınırlamaları

**Versiyon:** SeismoPattern v2  
**Son güncelleme:** Temmuz 2026  
**Durum:** Aktif araştırma yazılımı

---

## 1. Bu sistem ne değildir

### 1.1 Deterministik deprem tahmini değildir

SeismoPattern, herhangi bir depremin tam zamanını, tam konumunu veya
büyüklüğünü tahmin ettiğini iddia etmez.

Sistem:
- segment ölçekli sismik örüntüleri izler
- tarihsel öncü pencerelerle benzerlik kurar
- olasılıksal risk skoru üretir

Sistem şunları yapmaz:
- “şu tarihte deprem olacak” demez
- kesin alarm vermez
- tahliye kararını tek başına desteklemez

### 1.2 Operasyonel deprem erken uyarı sistemi değildir

P-dalgası tabanlı saniyelik uyarı sistemlerinden farklıdır.
Aylar ve yıllar ölçeğinde sismik örüntü izler.

### 1.3 Prospective doğrulama henüz yenidir

Prospective kayıt sistemi 15 Temmuz 2026'da başlatılmıştır.
Bu nedenle gerçek ileriye dönük performans henüz kısa dönemlidir.

---

## 2. Model sınırlamaları

### 2.1 Standard model bağımsız benchmark'ta zayıftır

- Standard model CV AUC: 0.71
- Standard model independent benchmark AUC: 0.42
- Debiased model benchmark AUC: 0.92

Bu fark magnitude feature'larının bölgesel imza taşıyabildiğini göstermektedir.

### 2.2 Dominant feature sorunu vardır

SHAP analizinde `w3_max_mw` aşırı baskın görünmektedir.
Bu, fiziksel sinyal yerine bölgesel katalog karakterinin öğrenilmiş olabileceğini düşündürür.

### 2.3 Veri örneklemi sınırlıdır

- 642 Mw7+ öncü pencere
- 1137 kontrol pencere

Bu boyut ağaç tabanlı modeller için kullanılabilir olsa da,
daha karmaşık modeller için sınırlayıcıdır.

### 2.4 Sınıf dengesizliği vardır

Büyük depremler nadir olaylardır.
Bu nedenle model değerlendirmesinde yalnızca AUC değil,
PR-AUC, Brier Score ve kalibrasyon da dikkate alınmalıdır.

### 2.5 Zaman penceresi sınırlıdır

Model esas olarak son 1-3 yıllık pencereyi temsil eder.
Daha uzun gerilme birikim süreçleri doğrudan modellenmez.

---

## 3. Veri sınırlamaları

### 3.1 Katalog tamlığı bölgeden bölgeye değişir

Japonya ve Kaliforniya gibi bölgelerde küçük olaylar iyi izlenirken,
bazı intraplate veya az izlenen bölgelerde katalog eksik olabilir.

### 3.2 Declustering kusursuz değildir

Gardner-Knopoff tabanlı declustering uygulanmıştır,
ancak artçıların tümünü kusursuz ayırmak mümkün değildir.

### 3.3 GPS verisi güncel değildir

MIDAS GPS veritabanı 2024 sonrası için tam güncel değildir.
Bu nedenle en yeni deformasyon değişimleri eksik temsil edilebilir.

### 3.4 Kaynak kataloglar homojen değildir

GCMT, ISC ve USGS kataloglarının:
- tamlık eşiği
- derinlik kalitesi
- gecikme yapısı
- olay yoğunluğu

aynı değildir.

---

## 4. Coğrafi sınırlamalar

### 4.1 Sistem şu an 25 bölge ile sınırlıdır

Mevcut operasyonel değerlendirme önceden tanımlı 25 bölge üzerinde yapılmaktadır.

### 4.2 Koordinat seçimine duyarlılık vardır

Cascadia, Nankai ve Tohoku gibi geniş zonlarda tek merkez koordinat sonucu etkileyebilir.

### 4.3 Düşük sismisiteli bölgelerde kalite cezası baskın olabilir

New Madrid gibi bölgelerde düşük kısa vadeli skor,
gerçek güvenlikten çok veri azlığını yansıtıyor olabilir.

---

## 5. Metodolojik uyarılar

### 5.1 Retrospective sonuçlar iyimser olabilir

Geçmişe dönük testler hindsight bias nedeniyle prospective testten daha iyi görünebilir.

### 5.2 Multiple comparison etkisi vardır

Aynı anda birçok bölge izlendiğinde bazı yüksek skorlar şans eseri oluşabilir.

### 5.3 Hazard hesapları yaklaşık çerçevedir

30 gün, 90 gün, 1 yıl ve 5 yıl hazard çıktıları
Poisson tabanlı operasyonel yaklaşık değerlerdir.

---

## 6. Etik ve operasyonel sınırlar

Bu sistem:
- resmi tehlike haritasının yerine geçmez
- tahliye kararı aracı değildir
- sigorta veya mühendislik standardı belirleme aracı değildir

Yüksek skor kesin deprem anlamına gelmez.
Düşük skor da güvenlik garantisi vermez.

---

## 7. Doğru kullanım

Uygun:
- segment ölçekli risk izleme
- kısa vadeli sismik anomali takibi
- araştırma ve karar destek amaçlı kullanım

Uygun olmayan:
- tek bir olay için zaman tahmini
- resmi erken uyarı yerine kullanım
- uzman değerlendirmesi olmadan kamuya alarm üretimi

---

## 8. Sonuç

SeismoPattern, kalibre edilmiş, çok katmanlı, segment ölçekli,
olasılıksal deprem risk izleme sistemidir.

Deterministik deprem zaman tahmini sistemi değildir.
"""


SCIENTIFIC_METHODOLOGY = """
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
"""


VALIDATION_METHODOLOGY = """
# SeismoPattern — Doğrulama Metodolojisi

**Versiyon:** SeismoPattern v2  
**Son güncelleme:** Temmuz 2026

---

## 1. Temel ilke

Bir model yalnızca eğitim başarısıyla güvenilir kabul edilmez.
Bu nedenle SeismoPattern çok katmanlı doğrulama kullanır.

Amaç:
- leakage var mı?
- zaman genellemesi var mı?
- bölge genellemesi var mı?
- olasılıklar kalibre mi?
- gerçek prospective ortamda tutarlı mı?

---

## 2. Çapraz doğrulama

Ana retrospective değerlendirme:
- 5-fold cross validation

Temel sonuçlar:
- OOS AUC-ROC: 0.7054
- 95% CI: 0.6788 - 0.7330
- PR-AUC: 0.9101

---

## 3. Kalibrasyon kontrolü

Kalibrasyon sonrası:
- Brier Score: 0.1791
- ECE: 0.0181

ECE'nin düşük olması model skorlarının aşırı güvenli olmadığını gösterir.

---

## 4. Kaçak ve sahte sinyal testleri

### 4.1 Label shuffle test

Etiketler rastgele karıştırıldığında:
- AUC yaklaşık 0.50

Sonuç:
- leakage bulgusu yok

### 4.2 Time split test

Geçmişte eğitilip daha sonraki dönemde test edilmiştir:
- AUC: 0.67

### 4.3 Leave-region-out test

Bir bölge dışarıda bırakılıp o bölgede test edilmiştir:
- AUC: 0.77

### 4.4 Spatial leakage test

Mekânsal yakınlığa bağlı sızıntı aranmıştır:
- geçti

---

## 5. İstatistiksel anlamlılık

### 5.1 DeLong testi

Model, 6 baseline ile karşılaştırılmıştır.

Sonuç:
- 6/6 baseline'a karşı anlamlı üstünlük
- p < 0.05

### 5.2 Bootstrap güven aralığı

AUC için bootstrap güven aralığı hesaplanmıştır:
- 0.6788 - 0.7330

---

## 6. Harici benchmark

Bağımsız benchmark, standard modelde ciddi bir sorunu ortaya çıkarmıştır.

- Standard model AUC: 0.42
- Debiased model AUC: 0.92

Yorum:
- magnitude feature'ları bölgesel imza taşıyor olabilir
- debiased model daha güvenilir operasyonel adaydır

---

## 7. Prospective doğrulama

### 7.1 Neden gerekli

Retrospective testler her zaman sınırlıdır.
Gerçek kanıt, modelin tahmin yaptığı anda kaydedilip
daha sonra olaylarla karşılaştırılmasıdır.

### 7.2 Uygulama

Prospective sistem:
- haftalık run kaydı
- SHA-256 hash
- JSON arşiv
- hash chain
- DB seviyesinde update/delete engeli

### 7.3 Başlangıç

- İlk immutable kayıt: 15 Temmuz 2026
- Otomatik haftalık zamanlayıcı: aktif

### 7.4 Beklenen değerlendirme takvimi

- 90 günlük ilk okuma: Ekim 2026
- 12 aylık daha güçlü analiz: Temmuz 2027

---

## 8. Prospective yorumlama çerçevesi

Önerilen yorum:
- AUC > 0.70: umut verici
- 0.60 - 0.70: zayıf ama gerçek sinyal olabilir
- 0.50 - 0.60: dikkatli yorum
- < 0.50: model revizyonu gerekir

Erken dönem sonuçları küçük olay sayısı nedeniyle belirsiz olacaktır.

---

## 9. Sonuç

Doğrulama metodolojisi tek metrikli değildir.
SeismoPattern şu dört soruya birlikte cevap arar:

1. retrospective başarı var mı?
2. calibration iyi mi?
3. leakage testi temiz mi?
4. prospective performans korunuyor mu?

Asıl nihai değerlendirme prospective sonuçlarla yapılacaktır.
"""


FEATURE_DEFINITIONS = """
# SeismoPattern — Feature Tanımları

**Versiyon:** SeismoPattern v2  
**Son güncelleme:** Temmuz 2026

---

## 1. Zaman pencereleri

Model her analiz noktası için geriye dönük 3 yıllık pencere kullanır:

- w1: son 0-12 ay
- w2: 12-24 ay
- w3: 24-36 ay

---

## 2. Aktivite sayımı feature'ları

### count_0_1y
Son 12 aydaki bağımsız olay sayısı.

### count_1_2y
12-24 ay önceki olay sayısı.

### count_2_3y
24-36 ay önceki olay sayısı.

### count_linear_trend
Üç penceredeki olay sayılarının doğrusal trend katsayısı.

### count_accel_ratio
Yakın dönem / uzak dönem olay sayısı oranı.

### w1_n_events
w1 toplam olay sayısı.

### w3_n_events
w3 toplam olay sayısı.

---

## 3. Zaman örüntüsü feature'ları

### quiescence_ratio
Görece sessizlik veya aktivasyon örüntüsünü temsil eder.

Kural tabanlı tip ayrımında kullanılır:
- >= 1.0 → TIP_A
- < 0.5 → TIP_B
- diğer → TIP_C

### accel_90d
Son 90 günlük aktivasyon ölçüsü.

### monthly_slope_36m
36 aylık aylık olay serisinin eğimi.

---

## 4. Büyüklük feature'ları

### w1_mean_mw
w1 ortalama büyüklüğü

### w1_std_mw
w1 büyüklük standart sapması

### w1_max_mw
w1 maksimum büyüklüğü

### w3_mean_mw
w3 ortalama büyüklüğü

### w3_max_mw
w3 maksimum büyüklüğü

Not:
`w3_max_mw` ve benzeri magnitude feature'ları,
bağımsız benchmark'ta bölgesel imza taşıma riski nedeniyle
debiased modelden çıkarılmıştır.

---

## 5. b-değeri feature'ları

### w1_b_value
Son yıl için Gutenberg-Richter b-değeri

### w3_b_value
3 yıllık uzak pencere b-değeri

### b_drop_w3_w1
b-değerindeki düşüş miktarı

Not:
b-değeri hesabı için minimum olay sayısı gerekir.
Düşük olaylı bölgelerde kalite cezası devreye girer.

---

## 6. Derinlik feature'ları

### w1_mean_depth_km
w1 ortalama derinlik

### w1_std_depth_km
w1 derinlik standart sapması

### w3_mean_depth_km
w3 ortalama derinlik

### depth_change_km
yakın ve uzak pencere derinlik farkı

---

## 7. Mekânsal feature'lar

### w1_mean_dist_km
w1 olaylarının merkeze ortalama uzaklığı

### w1_std_dist_km
w1 olaylarının merkeze uzaklık yayılımı

### w3_mean_dist_km
w3 ortalama mesafe

### spatial_focus_change
mekânsal odaklanma değişimi

### w1_migration_slope_km_day
yakın dönem sismisite göç eğimi

### w3_migration_slope_km_day
uzak dönem sismisite göç eğimi

---

## 8. Normalize feature'lar

### z_rate_1y
1 yıllık olay hızının z-skoru

### z_rate_3y
3 yıllık hızın z-skoru

### z_b_value_1y
1 yıllık b-değeri z-skoru

### z_b_value_3y
3 yıllık b-değeri z-skoru

### z_max_mw_1y
1 yıllık max magnitude z-skoru

### z_depth_1y
1 yıllık derinlik z-skoru

### z_dist_1y
1 yıllık mesafe z-skoru

---

## 9. Debiased feature seti

Debiased model magnitude ağırlıklı 4 feature'ı çıkarır:

- w3_max_mw
- w1_max_mw
- w3_mean_mw
- w1_mean_mw

Amaç:
bölgesel katalog imzası taşıyan feature'ların etkisini azaltmak.

---

## 10. Feature dışı risk katmanları

Aşağıdaki bileşenler model feature'ı değil,
dual risk sisteminin ek katmanlarıdır:

- GPS velocity / strain
- fault distance
- CFF
- NLP precursor signals
- quality penalty
- tectonic penalty

Bunlar operasyonel risk üretiminde kullanılır.
"""


RISK_MODEL = """
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
"""


CALIBRATION = """
# SeismoPattern — Kalibrasyon

**Versiyon:** SeismoPattern v2  
**Son güncelleme:** Temmuz 2026

---

## 1. Kalibrasyon neden gerekli?

Bir sınıflandırma modeli yüksek AUC üretebilir,
ancak olasılık çıktıları iyi kalibre edilmemiş olabilir.

Örnek:
- model 0.80 veriyorsa
- benzer örneklerin gerçekten yaklaşık yüzde 80'inde olay görülmelidir

Kalibrasyonun amacı budur.

---

## 2. Kullanılan yöntem

SeismoPattern'de ham model çıktıları
Isotonic Regression ile kalibre edilmiştir.

Neden isotonic?
- monoton ilişkiyi korur
- ağaç tabanlı modeller için uygundur
- doğrusal olmayan düzeltme yapabilir

---

## 3. Uygulama noktası

Kalibrasyon, birleşik model skorunun
olasılık olarak raporlanmasından önce uygulanır.

Yani akış:
- model tahmini
- birleşik skor
- isotonic kalibrasyon
- son olasılık

---

## 4. Ölçütler

Kalibrasyon kalitesi şu ölçülerle izlenir:

### Brier Score
Tahmin edilen olasılıkla gerçekleşen sonuç arasındaki ortalama kare hata.

Mevcut sonuç:
- 0.1791

### ECE
Expected Calibration Error

Mevcut sonuç:
- 0.0181

Bu değer düşük olduğu için kalibrasyon güçlü kabul edilir.

---

## 5. Operasyonel anlamı

Kalibre model:
- risk skorlarını daha yorumlanabilir hale getirir
- bölgeler arası göreli kıyası güçlendirir
- overconfidence riskini azaltır

---

## 6. Sınırlamalar

Kalibrasyon retrospective veride öğrenilir.
Bu nedenle prospective ortamda drift oluşursa
kalibrasyon kalitesi zamanla bozulabilir.

Bu yüzden düzenli yeniden kontrol gerekir.

---

## 7. Gelecek geliştirmeler

Planlanan iyileştirmeler:
- horizon bazlı ayrı kalibrasyon
- uncertainty band ile birlikte raporlama
- prospective verilerle yeniden kalibrasyon
- zone-group bazlı kalibrasyon karşılaştırması

---

## 8. Sonuç

Kalibrasyon, SeismoPattern'in sadece sıralama yapan değil,
yorumlanabilir olasılık üreten bir sistem olmasını sağlar.

Mevcut ECE değeri operasyonel kullanım açısından güçlüdür,
ancak prospective performans ile birlikte izlenmeye devam edilmelidir.
"""


DOCUMENTS = {
    "Limitations.md": LIMITATIONS,
    "Scientific_Methodology.md": SCIENTIFIC_METHODOLOGY,
    "Validation_Methodology.md": VALIDATION_METHODOLOGY,
    "Feature_Definitions.md": FEATURE_DEFINITIONS,
    "Risk_Model.md": RISK_MODEL,
    "Calibration.md": CALIBRATION,
}


def main():
    for filename, content in DOCUMENTS.items():
        write_doc(filename, content)

    print()
    print("Tum docs dosyalari olusturuldu.")
    print("UTF-8 kontrol icin ornek:")
    print(
        "python -c \"print(open('docs/Scientific_Methodology.md', encoding='utf-8').read()[:200])\""
    )


if __name__ == "__main__":
    main()
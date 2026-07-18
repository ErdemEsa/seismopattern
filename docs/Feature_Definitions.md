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

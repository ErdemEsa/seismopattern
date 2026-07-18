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

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

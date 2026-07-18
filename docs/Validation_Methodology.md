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

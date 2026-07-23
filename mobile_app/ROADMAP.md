# SeismoPattern Mobile — Roadmap

## Yakın vadeli
- [ ] E adımı: Manuel koordinat analizi ekranı
- [ ] F adımı: Backend deployment (VPS + HTTPS)
- [ ] D adımı: UI iyileştirmeleri

## Otomatik backend keşfi (F sonrası artık gerek kalmayacak)
Şu an: Kullanıcı Ayarlar ekranından IP girmek zorunda.
F adımı tamamlanınca uygulama sabit bir public URL'ye bağlanacak
(örn. https://api.seismopattern.io), IP girme derdi ortadan kalkacak.

Deployment tamamlanınca:
- config.dart varsayılan URL production adresine değişecek
- Ayarlar ekranı "Gelişmiş" bölümüne taşınacak (hâlâ opsiyonel kalacak)
- lk kullanıcı deneyimi: aç → çalış

## Uzun vadeli
- [ ] Offline cache (last-known zone data)
- [ ] Push notification (KRITIK zone'da yeni Mw6+)
- [ ] Kullanıcı bookmarks (favori zone'lar)
- [ ] i18n (EN / TR toggle)
- [ ] Dark mode

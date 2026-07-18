# SeismoPattern - Kurulum ve Calistirma Kilavuzu

Guncelleme: Temmuz 2026

---

## Gereksinimler

- Python 3.12+
- Docker Desktop (Windows) veya Docker Engine (Linux)
- 8 GB RAM onerilir
- 10 GB disk alani

---

## Yontem 1: Docker ile Calistirma (Onerilen)

Bu yontem tum bagimlilikları otomatik olarak yonetir.

Adim 1: Projeyi indirin

    git clone <repo>
    cd SeismoPattern

Adim 2: Compose ile baslatin

    docker compose -f docker-compose.runtime.yml up

Ilk baslangicta pip install yapacagi icin 3-5 dakika surer.
Sonraki baslatmalarda paketler zaten kurulu oldugu icin hizlidir.

Adim 3: Erisim

    Web arayuzu : http://127.0.0.1:5000
    API docs    : http://127.0.0.1:5000/docs

Arka planda calistirmak icin:

    docker compose -f docker-compose.runtime.yml up -d

Durdurmak icin:

    docker compose -f docker-compose.runtime.yml down

---

## Yontem 2: Yerel Python ile Calistirma

Adim 1: Sanal ortam olusturun

    python -m venv venv
    venv\Scripts\activate   (Windows)
    source venv/bin/activate  (Linux/Mac)

Adim 2: Bagimliliklar

    pip install -r requirements.txt
    pip install --no-deps xgboost==3.3.0

Adim 3: Calistirin

    python app.py

---

## Kalici Docker Image Olusturma

Ilk basarili calistirma sonrasi container kaydedilebilir:

    docker commit seismopattern_app seismopattern:stable

Sonra docker-compose.runtime.yml icinde degistirin:

    image: seismopattern:stable

Bu sekilde sonraki baslangiclar pip install atlar (~30 saniye).

---

## Bagimlilik Notlari

Runtime bagimliliklar (requirements.txt):

    Flask==3.0.3
    Werkzeug==3.0.6
    pandas, numpy, scipy, requests, matplotlib
    scikit-learn, joblib, reportlab, folium
    SQLAlchemy, lxml, networkx, shapely

xgboost ayrica kurulur:

    pip install --no-deps xgboost==3.3.0

Egitim bagimlilikları (requirements_train.txt):

    lightgbm, catboost, imbalanced-learn, shap, torch

Bu paketler sadece model egitimi icin gereklidir.
Runtime API icin gerekmez.

---

## Flask/Werkzeug Uyum Notu

Flask==3.0.3 ve Werkzeug==3.0.6 birlikte kullanilmalidir.
Flask==3.1.x ile Werkzeug==3.1.8 kombinasyonu
bu sistemde LocalProxy uyumsuzluguna yol acmaktadir.

---

## Bilinen Sorunlar

Docker BuildKit EOF hatasi:

    docker buildx build --output type=docker,dest=image.tar -t seismopattern:test .

ISC baglanti gecikmesi:

    ISC mirror (iris.washington.edu) her zaman basarisiz olur.
    Sistem otomatik olarak www.isc.ac.uk adresine gecer.
    Ilk sorgu 7-15 saniye surebilir.
    Onbellek TTL: 2 saat.

---

## Prospective Izleme

Haftalik tahmin kaydi:

    python scripts/prospective_tracker.py --record
    python scripts/prospective_tracker.py --check-events
    python scripts/prospective_tracker.py --verify
    python scripts/prospective_tracker.py --status

Windows Task Scheduler kurulumu projeyle birlikte gelir.

---

## API Hizli Basvuru

    Durum         GET  /api/status
    Ana arayuz    GET  /
    API docs      GET  /docs

    Risk analizi  POST /api/realtime
                  POST /api/historical
                  GET  /api/geodynamic

    Bolgeler      GET  /api/zones
                  GET  /api/dual_risk
                  GET  /api/hazard_table
                  GET  /api/hazard

    Fiziksel      GET  /api/faults
                  GET  /api/gps
                  GET  /api/cff

    Raporlar      GET  /api/pdf
                  GET  /map

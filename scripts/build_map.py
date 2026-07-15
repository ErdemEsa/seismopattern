#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Seviye 4.2: Gelismis Harita Olusturucu
=======================================================
Leaflet.js tabanli interaktif harita olusturur.
Tum katmanlari (sismik, fay, GPS, CFF) gorsellestirir.

Kullanim:
  python scripts/build_map.py
  Sonra: http://127.0.0.1:5000/map
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("output")
MAP_FILE = OUTPUT_DIR / "seismo_map.html"


def get_monitor_data():
    """Son tarama verilerini al."""
    db_path = OUTPUT_DIR / "monitor.db"
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("""
        SELECT zone_key, zone_name, lat, lon,
               risk_score, risk_level, pattern_type,
               n_events, b_value, quiescence, cff_total, gps_vh,
               source, timestamp
        FROM scan_results
        WHERE id IN (
            SELECT MAX(id) FROM scan_results GROUP BY zone_key
        )
        ORDER BY risk_score DESC
    """).fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "key": r[0], "name": r[1],
            "lat": r[2], "lon": r[3],
            "score": r[4], "level": r[5], "pattern": r[6],
            "n_events": r[7], "b_value": r[8],
            "quiescence": r[9], "cff": r[10], "gps_vh": r[11],
            "source": r[12], "timestamp": r[13],
        })
    return results


def get_scored_events():
    """Tarihsel buyuk deprem skorlarini al."""
    path = OUTPUT_DIR / "real_risk_scored.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path, low_memory=False)
    if "risk_score" not in df.columns:
        return []

    lat_col = next((c for c in ["lat", "main_lat"] if c in df.columns), None)
    lon_col = next((c for c in ["lon", "main_lon"] if c in df.columns), None)
    mw_col = next((c for c in ["mw", "main_mw"] if c in df.columns), None)
    dt_col = next((c for c in ["datetime_utc", "main_datetime_utc"]
                   if c in df.columns), None)
    ft_col = next((c for c in ["fault_type", "main_fault_type"]
                   if c in df.columns), None)
    pt_col = "pattern_type" if "pattern_type" in df.columns else None

    if not lat_col or not lon_col:
        return []

    events = []
    for _, row in df.iterrows():
        lat = row.get(lat_col)
        lon = row.get(lon_col)
        if pd.isna(lat) or pd.isna(lon):
            continue

        score = row.get("risk_score", 0)
        if pd.isna(score):
            continue

        events.append({
            "lat": float(lat),
            "lon": float(lon),
            "score": float(score),
            "mw": float(row.get(mw_col, 0)) if mw_col else 0,
            "date": str(row.get(dt_col, ""))[:10] if dt_col else "",
            "fault_type": str(row.get(ft_col, "")) if ft_col else "",
            "pattern": str(row.get(pt_col, "")) if pt_col else "",
        })

    return events


def build_map():
    """Leaflet.js harita HTML dosyasi olustur."""
    monitor_data = get_monitor_data()
    scored_events = get_scored_events()

    # Risk rengini belirle
    def risk_color(score):
        if score >= 0.75: return "#f85149"
        if score >= 0.60: return "#e3b341"
        if score >= 0.45: return "#d29922"
        if score >= 0.30: return "#c69026"
        return "#3fb950"

    def risk_label(score):
        if score >= 0.75: return "KRITIK"
        if score >= 0.60: return "YUKSEK"
        if score >= 0.45: return "ORTA"
        if score >= 0.30: return "DIKKAT"
        return "DUSUK"

    # Monitor marker'lari JSON
    monitor_js = json.dumps([{
        "lat": d["lat"], "lon": d["lon"],
        "name": d["name"], "key": d["key"],
        "score": d["score"],
        "level": d["level"],
        "pattern": d["pattern"],
        "n_events": d["n_events"],
        "b_value": d["b_value"],
        "quiescence": d["quiescence"],
        "cff": d["cff"],
        "gps_vh": d["gps_vh"],
        "source": d["source"],
        "timestamp": d["timestamp"],
        "color": risk_color(d["score"]),
    } for d in monitor_data], default=str)

    # Tarihsel olaylar JSON (en onemli 200)
    top_events = sorted(scored_events, key=lambda x: -x["score"])[:200]
    events_js = json.dumps([{
        "lat": e["lat"], "lon": e["lon"],
        "score": e["score"],
        "mw": e["mw"],
        "date": e["date"],
        "fault_type": e["fault_type"],
        "pattern": e["pattern"],
        "color": risk_color(e["score"]),
    } for e in top_events], default=str)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SeismoPattern - Harita</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:Arial,sans-serif;background:#0d1117;color:#e6edf3}}
#map{{width:100%;height:calc(100vh - 120px)}}
.header{{background:#161b22;padding:12px 20px;border-bottom:1px solid #30363d;display:flex;justify-content:space-between;align-items:center}}
.header h1{{font-size:18px;color:#fff}}
.header a{{color:#388bfd;text-decoration:none;font-size:13px}}
.controls{{background:#161b22;padding:8px 20px;border-bottom:1px solid #30363d;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.controls label{{font-size:12px;color:#8b949e;cursor:pointer;display:flex;align-items:center;gap:4px}}
.controls input[type=checkbox]{{cursor:pointer}}
.controls button{{background:#21262d;color:#fff;border:1px solid #30363d;padding:5px 12px;border-radius:6px;cursor:pointer;font-size:12px}}
.controls button:hover{{background:#30363d}}
.legend{{position:absolute;bottom:30px;left:12px;z-index:1000;background:rgba(13,17,23,0.95);padding:12px;border-radius:8px;border:1px solid #30363d;font-size:12px;color:#e6edf3}}
.legend div{{margin:3px 0;display:flex;align-items:center;gap:6px}}
.legend span{{display:inline-block;width:14px;height:14px;border-radius:50%}}
.info-panel{{position:absolute;top:60px;right:12px;z-index:1000;background:rgba(13,17,23,0.95);padding:14px;border-radius:8px;border:1px solid #30363d;font-size:12px;color:#e6edf3;max-width:320px;display:none}}
.info-panel.show{{display:block}}
.info-panel h3{{font-size:14px;margin-bottom:8px;color:#f0f6fc}}
.info-panel .close{{position:absolute;top:8px;right:10px;cursor:pointer;color:#8b949e;font-size:16px}}
.stat{{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #21262d}}
.stat .label{{color:#8b949e}}
.stat .value{{font-weight:bold}}
</style>
</head>
<body>

<div class="header">
  <h1>SeismoPattern - Dunya Risk Haritasi</h1>
  <a href="/">Ana Sayfaya Don</a>
</div>

<div class="controls">
  <label><input type="checkbox" id="layerMonitor" checked onchange="toggleLayer('monitor')"> Izlenen Bolgeler</label>
  <label><input type="checkbox" id="layerEvents" onchange="toggleLayer('events')"> Tarihsel Mw7+ ({len(top_events)})</label>
  <button onclick="zoomToAll()">Tum Bolgeler</button>
  <button onclick="location.reload()">Yenile</button>
  <span style="font-size:11px;color:#484f58;margin-left:auto">
    Son guncelleme: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
  </span>
</div>

<div id="map"></div>

<div class="legend">
  <b>Risk Seviyeleri</b>
  <div><span style="background:#f85149"></span> KRITIK (75%+)</div>
  <div><span style="background:#e3b341"></span> YUKSEK (60-75%)</div>
  <div><span style="background:#d29922"></span> ORTA (45-60%)</div>
  <div><span style="background:#c69026"></span> DIKKAT (30-45%)</div>
  <div><span style="background:#3fb950"></span> DUSUK (&lt;30%)</div>
  <div style="margin-top:6px;font-size:11px;color:#484f58">
    Daire buyuklugu = risk skoru
  </div>
</div>

<div class="info-panel" id="infoPanel">
  <span class="close" onclick="closePanel()">&times;</span>
  <h3 id="panelTitle">-</h3>
  <div id="panelBody"></div>
</div>

<script>
// Harita olustur
var map = L.map('map', {{
  center: [25, 30],
  zoom: 3,
  maxZoom: 12,
  minZoom: 2
}});

// Koyu tema harita
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: 'CartoDB',
  subdomains: 'abcd',
  maxZoom: 19
}}).addTo(map);

// Veriler
var monitorData = {monitor_js};
var eventsData = {events_js};

// Katmanlar
var monitorLayer = L.layerGroup().addTo(map);
var eventsLayer = L.layerGroup();

// Monitor marker'lari
monitorData.forEach(function(d) {{
  var radius = Math.max(12, d.score * 30);

  var circle = L.circleMarker([d.lat, d.lon], {{
    radius: radius,
    color: d.color,
    fillColor: d.color,
    fillOpacity: 0.6,
    weight: 3,
    opacity: 0.9,
  }}).addTo(monitorLayer);

  // Pulsing efekti icin ek halka
  L.circleMarker([d.lat, d.lon], {{
    radius: radius + 8,
    color: d.color,
    fillColor: 'transparent',
    fillOpacity: 0,
    weight: 1,
    opacity: 0.3,
    dashArray: '5,5'
  }}).addTo(monitorLayer);

  // Label
  var label = L.divIcon({{
    html: '<div style="color:' + d.color + ';font-size:11px;font-weight:bold;text-shadow:0 0 4px #000;white-space:nowrap">'
      + d.name + ' ' + Math.round(d.score*100) + '%</div>',
    className: '',
    iconAnchor: [-radius-4, 0]
  }});
  L.marker([d.lat, d.lon], {{icon: label}}).addTo(monitorLayer);

  circle.on('click', function() {{ showMonitorPanel(d); }});
}});

// Tarihsel olaylar
eventsData.forEach(function(e) {{
  var radius = Math.max(3, e.score * 10);
  var circle = L.circleMarker([e.lat, e.lon], {{
    radius: radius,
    color: e.color,
    fillColor: e.color,
    fillOpacity: 0.5,
    weight: 1,
  }}).addTo(eventsLayer);

  circle.bindTooltip(
    'Mw ' + e.mw.toFixed(1) + ' | ' + e.date + ' | ' +
    Math.round(e.score*100) + '%'
  );
}});

// Panel fonksiyonlari
function showMonitorPanel(d) {{
  document.getElementById('panelTitle').textContent = d.name;

  var fmt = function(v, suffix) {{
    if (v === null || v === undefined) return '-';
    return typeof v === 'number' ? v.toFixed(2) + (suffix || '') : v;
  }};

  var html = '';
  html += '<div class="stat"><span class="label">Risk Skoru</span><span class="value" style="color:'+d.color+'">' + Math.round(d.score*100) + '% ' + d.level + '</span></div>';
  html += '<div class="stat"><span class="label">Sablon</span><span class="value">' + d.pattern + '</span></div>';
  html += '<div class="stat"><span class="label">Toplam Olay</span><span class="value">' + (d.n_events || '-') + '</span></div>';
  html += '<div class="stat"><span class="label">b-degeri</span><span class="value">' + fmt(d.b_value) + '</span></div>';
  html += '<div class="stat"><span class="label">Quiescence</span><span class="value">' + fmt(d.quiescence) + '</span></div>';
  html += '<div class="stat"><span class="label">CFF (bar)</span><span class="value">' + fmt(d.cff) + '</span></div>';
  html += '<div class="stat"><span class="label">GPS Vh (mm/yr)</span><span class="value">' + fmt(d.gps_vh) + '</span></div>';
  html += '<div class="stat"><span class="label">Kaynak</span><span class="value">' + (d.source || '-') + '</span></div>';
  html += '<div class="stat"><span class="label">Tarih</span><span class="value">' + (d.timestamp || '').substring(0,16) + '</span></div>';
  html += '<div style="margin-top:8px"><a href="/api/geodynamic?lat='+d.lat+'&lon='+d.lon+'" target="_blank" style="color:#388bfd;font-size:11px">Tam JSON rapor</a></div>';

  document.getElementById('panelBody').innerHTML = html;
  document.getElementById('infoPanel').classList.add('show');
}}

function closePanel() {{
  document.getElementById('infoPanel').classList.remove('show');
}}

function toggleLayer(name) {{
  if (name === 'monitor') {{
    if (map.hasLayer(monitorLayer)) map.removeLayer(monitorLayer);
    else map.addLayer(monitorLayer);
  }} else if (name === 'events') {{
    if (map.hasLayer(eventsLayer)) map.removeLayer(eventsLayer);
    else map.addLayer(eventsLayer);
  }}
}}

function zoomToAll() {{
  if (monitorData.length > 0) {{
    var bounds = L.latLngBounds(monitorData.map(function(d) {{ return [d.lat, d.lon]; }}));
    map.fitBounds(bounds, {{padding: [50,50]}});
  }}
}}

// Haritaya tiklandiginda koordinat goster
map.on('click', function(e) {{
  var lat = e.latlng.lat.toFixed(3);
  var lon = e.latlng.lng.toFixed(3);
  L.popup()
    .setLatLng(e.latlng)
    .setContent(
      '<div style="font-size:12px">' +
      lat + ', ' + lon + '<br>' +
      '<a href="/api/geodynamic?lat='+lat+'&lon='+lon+'" target="_blank" style="color:#388bfd">Analiz Et</a>' +
      '</div>'
    )
    .openOn(map);
}});
</script>
</body>
</html>"""

    MAP_FILE.write_text(html, encoding="utf-8")
    print(f"Harita olusturuldu: {MAP_FILE}")
    print(f"  Monitor noktasi: {len(monitor_data)}")
    print(f"  Tarihsel olay: {len(top_events)}")
    return str(MAP_FILE)


if __name__ == "__main__":
    build_map()
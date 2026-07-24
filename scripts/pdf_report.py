#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Seviye 4.4: PDF Rapor Uretimi
================================================
Bir bolge icin profesyonel PDF rapor olusturur.

Kullanim:
  python scripts/pdf_report.py --lat 41.01 --lon 28.98
  python scripts/pdf_report.py --lat 38.30 --lon 142.37 --refdate 2011-02-11
"""

import json
import time
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                  Table, TableStyle, PageBreak,
                                  HRFlowable)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path("output/reports")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# RENK TANIMLARI
# =========================================================

COLORS = {
    "KRITIK": colors.HexColor("#f85149"),
    "YUKSEK": colors.HexColor("#e3b341"),
    "ORTA":   colors.HexColor("#d29922"),
    "DIKKAT": colors.HexColor("#c69026"),
    "DUSUK":  colors.HexColor("#3fb950"),
    "bg":     colors.HexColor("#161b22"),
    "border": colors.HexColor("#30363d"),
    "text":   colors.HexColor("#333333"),
    "muted":  colors.HexColor("#666666"),
    "accent": colors.HexColor("#388bfd"),
}


def risk_color(level):
    return COLORS.get(level, COLORS["DUSUK"])


# =========================================================
# STILLER
# =========================================================

def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="Title2",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=6,
        textColor=colors.HexColor("#1a1a1a"),
    ))

    styles.add(ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=COLORS["muted"],
        spaceAfter=12,
    ))

    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=16,
        spaceAfter=8,
        textColor=COLORS["accent"],
        borderWidth=0,
        borderPadding=0,
    ))

    styles.add(ParagraphStyle(
        name="BodyText2",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
        textColor=COLORS["text"],
    ))

    styles.add(ParagraphStyle(
        name="SmallText",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=COLORS["muted"],
    ))

    styles.add(ParagraphStyle(
        name="Warning",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#994400"),
        backColor=colors.HexColor("#fff3e0"),
        borderWidth=1,
        borderColor=colors.HexColor("#e3b341"),
        borderPadding=6,
    ))

    styles.add(ParagraphStyle(
        name="BigScore",
        parent=styles["Normal"],
        fontSize=28,
        alignment=TA_CENTER,
        spaceAfter=8,
        spaceBefore=8,
        leading=34,
    ))
    return styles


# =========================================================
# VERI TOPLAMA
# =========================================================

def collect_data(lat, lon, ref_date=None):
    """Tum katmanlardan veri topla."""
    data = {"lat": lat, "lon": lon, "ref_date": ref_date,
            "timestamp": datetime.utcnow().isoformat()}

    # Sismik
    try:
        from isc_fetch_v2 import fetch_and_analyze
        feats, meta, err = fetch_and_analyze(
            lat, lon, 300, 2.5, ref_date=ref_date, use_cache=True
        )
        if feats:
            data["seismic_features"] = feats
            data["seismic_meta"] = meta
    except Exception as e:
        data["seismic_error"] = str(e)

    # Risk tahmini
    try:
        import joblib
        import numpy as np
        import pandas as pd

        MODEL_DIR = Path("output/models")
        fl_path = MODEL_DIR / "feature_lists.json"
        if fl_path.exists():
            with open(fl_path) as f:
                FL = json.load(f)
            MODELS = {}
            for pt in ["TIP_A", "TIP_B", "TIP_C"]:
                p = MODEL_DIR / f"model_{pt}.joblib"
                if p.exists():
                    MODELS[pt] = (joblib.load(p), FL.get(pt, []))

            if MODELS and "seismic_features" in data:
                row = data["seismic_features"].copy()
                c0 = float(row.get("count_0_1y", 0) or 0)
                c1 = float(row.get("count_1_2y", 0) or 0)
                c2 = float(row.get("count_2_3y", 0) or 0)
                row["count_linear_trend"] = c0 - c2
                row["count_accel_ratio"] = c0 / ((c1+c2)/2 + 1e-6)
                try: row["b_drop_w3_w1"] = float(row.get("w3_b_value",0) or 0) - float(row.get("w1_b_value",0) or 0)
                except: row["b_drop_w3_w1"] = 0
                try: row["spatial_focus_change"] = float(row.get("w3_mean_dist_km",0) or 0) - float(row.get("w1_mean_dist_km",0) or 0)
                except: row["spatial_focus_change"] = 0
                try: row["depth_change_km"] = float(row.get("w1_mean_depth_km",0) or 0) - float(row.get("w3_mean_depth_km",0) or 0)
                except: row["depth_change_km"] = 0
                if not row.get("quiescence_ratio"):
                    prev = (c1+c2)/2
                    row["quiescence_ratio"] = c0/prev if prev > 0 else None
                if not row.get("w3_n_events"): row["w3_n_events"] = c0+c1+c2
                if not row.get("w1_n_events"): row["w1_n_events"] = c0

                # Tip
                qr = row.get("quiescence_ratio")
                acc = row.get("accel_90d")
                n3 = row.get("w3_n_events", 0) or 0
                if qr is None or pd.isna(qr) or n3 < 3: pt = "TIP_C"
                elif qr < 0.5: pt = "TIP_B"
                elif qr >= 1.0: pt = "TIP_A"
                else: pt = "TIP_A"

                comp = {}
                for tip, (pipe, feats_list) in MODELS.items():
                    try:
                        x = pd.DataFrame([{f: row.get(f, np.nan) for f in feats_list}])
                        comp[tip] = round(float(pipe.predict_proba(x)[0, 1]), 4)
                    except: pass

                W = {"TIP_A": 1.0, "TIP_B": 1.2, "TIP_C": 0.5}
                primary = comp.get(pt)
                valid = {t: s for t, s in comp.items() if s is not None}
                if valid:
                    ws = sum(W.get(t, 1)*s for t, s in valid.items())
                    wt = sum(W.get(t, 1) for t in valid)
                    ens = ws/wt if wt > 0 else 0.5
                    fin = 0.7*primary + 0.3*ens if primary else ens
                    if fin >= 0.75: lv = "KRITIK"
                    elif fin >= 0.60: lv = "YUKSEK"
                    elif fin >= 0.45: lv = "ORTA"
                    elif fin >= 0.30: lv = "DIKKAT"
                    else: lv = "DUSUK"
                    data["risk"] = {"score": round(fin, 4), "level": lv,
                                    "pattern": pt, "components": comp}
    except Exception as e:
        data["risk_error"] = str(e)

    # Fay
    try:
        from fault_distance import get_fault_info_for_app
        data["faults"] = get_fault_info_for_app(lat, lon)
    except: pass

    # CFF
    try:
        from coulomb_simple import compute_cff_for_app
        data["cff"] = compute_cff_for_app(lat, lon, ref_date)
    except: pass

    # GPS
    try:
        from gps_velocity import get_gps_info_for_app
        data["gps"] = get_gps_info_for_app(lat, lon)
    except: pass

    # NLP
    try:
        from nlp_scanner import get_nlp_info_for_app
        data["nlp"] = get_nlp_info_for_app(lat, lon)
    except: pass

    # Bolge
    ZONES = {
        "marmara": {"name":"Marmara","lat":40.77,"lon":29.00,"last_major":"1999-08-17","last_major_mw":7.6,"recurrence_years":250,"fault_name":"KAF","fault_type":"STRIKE_SLIP","expected_mw":"7.0-7.4","population_risk":"~16 milyon"},
        "kahramanmaras": {"name":"Kahramanmaras","lat":37.22,"lon":37.02,"last_major":"2023-02-06","last_major_mw":7.8,"recurrence_years":500,"fault_name":"DAF","fault_type":"STRIKE_SLIP","expected_mw":"7.0-7.8","population_risk":"~3 milyon"},
        "cascadia": {"name":"Cascadia","lat":45.50,"lon":-125.00,"last_major":"1700-01-26","last_major_mw":9.0,"recurrence_years":300,"fault_name":"Cascadia Megathrust","fault_type":"REVERSE","expected_mw":"8.5-9.2","population_risk":"~10 milyon"},
        "nankai": {"name":"Nankai","lat":33.00,"lon":135.00,"last_major":"1946-12-21","last_major_mw":8.1,"recurrence_years":140,"fault_name":"Nankai Trough","fault_type":"REVERSE","expected_mw":"8.0-9.1","population_risk":"~30 milyon"},
        "tohoku": {"name":"Tohoku","lat":38.30,"lon":142.37,"last_major":"2011-03-11","last_major_mw":9.1,"recurrence_years":600,"fault_name":"Japan Trench","fault_type":"REVERSE","expected_mw":"8.0-9.0+","population_risk":"~6 milyon"},
        "lima": {"name":"Lima","lat":-12.00,"lon":-77.00,"last_major":"1746-10-28","last_major_mw":8.8,"recurrence_years":300,"fault_name":"Nazca Plate","fault_type":"REVERSE","expected_mw":"8.5-9.0","population_risk":"~12 milyon"},
        "izmit": {"name":"Izmit","lat":40.75,"lon":29.86,"last_major":"1999-08-17","last_major_mw":7.6,"recurrence_years":250,"fault_name":"KAF-Izmit","fault_type":"STRIKE_SLIP","expected_mw":"7.0-7.6","population_risk":"~5 milyon"},
    }
    def hav(a,b,c,d):
        a,b,c,d=map(math.radians,[a,b,c,d])
        dl,dn=c-a,d-b
        x=math.sin(dl/2)**2+math.cos(a)*math.cos(c)*math.sin(dn/2)**2
        return 6371*2*math.asin(math.sqrt(x))

    best_zone = None
    best_dist = 999999
    for k, z in ZONES.items():
        d = hav(lat, lon, z["lat"], z["lon"])
        if d < best_dist and d < 500:
            best_zone = dict(z)
            best_zone["key"] = k
            best_dist = d
    data["zone"] = best_zone

    # Quality-adjusted + Hazard
    try:
        from quality_adjusted_risk import compute_quality_adjusted_risk
        qar = compute_quality_adjusted_risk(lat, lon, ref_date)
        if not qar.get("error"):
            data["quality_adjusted"] = qar
    except Exception:
        pass

    try:
        from survival_hazard_model import analyze_hazard
        hz_result = analyze_hazard(lat=lat, lon=lon, ref_date=ref_date)
        if hz_result.get("hazards"):
            data["hazards"] = hz_result["hazards"]
    except Exception:
        pass

    return data


# =========================================================
# PDF OLUSTUR
# =========================================================

def generate_pdf(data, output_path=None):
    """PDF rapor olustur."""
    lat = data["lat"]
    lon = data["lon"]
    ref = data.get("ref_date")
    ts = data.get("timestamp", "")[:16]

    zone = data.get("zone")
    zone_name = zone["name"] if zone else f"{lat:.2f}N, {lon:.2f}E"

    if output_path is None:
        safe_name = zone_name.replace(" ", "_").replace("/", "_")
        date_str = (ref or datetime.utcnow().strftime("%Y-%m-%d")).replace("-", "")
        output_path = OUTPUT_DIR / f"SeismoPattern_{safe_name}_{date_str}.pdf"

    styles = get_styles()
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )

    story = []

    # === BASLIK ===
    story.append(Paragraph("SeismoPattern", styles["Title2"]))
    story.append(Paragraph("Deprem Oncu Sablon Analiz Raporu", styles["Subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=COLORS["accent"]))
    story.append(Spacer(1, 8))

    # Meta tablo
    meta_data = [
        ["Bolge", zone_name],
        ["Koordinat", f"{lat:.4f} N, {lon:.4f} E"],
        ["Referans Tarih", ref or "Guncel"],
        ["Rapor Tarihi", ts],
        ["Model", "SeismoPattern v4 (AUC 0.9087)"],
    ]
    meta_table = Table(meta_data, colWidths=[120, 350])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), COLORS["muted"]),
        ("TEXTCOLOR", (1, 0), (1, -1), COLORS["text"]),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # === RISK SONUCU ===
    risk = data.get("risk")
    if risk:
        score = risk["score"]
        level = risk["level"]
        pattern = risk["pattern"]
        color = risk_color(level)

        story.append(Paragraph("RISK DEGERLENDIRMESI", styles["SectionTitle"]))

        story.append(Paragraph(
            f'<para alignment="center"><font size="24" color="{color.hexval()}"><b>{int(score*100)}%</b></font></para>',
            styles["BigScore"]
        ))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f'<para alignment="center"><font size="12" color="{color.hexval()}">{level}</font></para>',
            styles["BodyText2"]
        ))
        story.append(Spacer(1, 6))

        risk_data = [
            ["Risk Skoru", f"{score*100:.1f}%"],
            ["Risk Seviyesi", level],
            ["Sablon Tipi", pattern],
        ]
        comps = risk.get("components", {})
        for tip, val in comps.items():
            if val is not None:
                risk_data.append([f"{tip} Skoru", f"{val*100:.1f}%"])

        risk_table = Table(risk_data, colWidths=[150, 320])
        risk_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), COLORS["muted"]),
            ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, COLORS["border"]),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 8))

    # === SISMIK VERILER ===
    sf = data.get("seismic_features")
    sm = data.get("seismic_meta")
    if sf:
        story.append(Paragraph("SISMIK ANALIZ", styles["SectionTitle"]))

        seis_data = [
            ["Kaynak", sm.get("source", "?") if sm else "?"],
            ["Toplam Olay", str(sm.get("n_total", "?")) if sm else "?"],
            ["Mc", str(sm.get("mc", "?")) if sm else "?"],
            ["b Kalitesi", sm.get("b_quality", "?") if sm else "?"],
            ["Son 1 Yil Olay", str(sf.get("count_0_1y", "-"))],
            ["1-2 Yil Olay", str(sf.get("count_1_2y", "-"))],
            ["2-3 Yil Olay", str(sf.get("count_2_3y", "-"))],
            ["Quiescence Orani", f"{sf['quiescence_ratio']:.3f}" if sf.get("quiescence_ratio") else "-"],
            ["90 Gun Hizlanma", f"{sf['accel_90d']:.3f}" if sf.get("accel_90d") else "-"],
            ["b-degeri (1y)", f"{sf['w1_b_value']:.4f}" if sf.get("w1_b_value") else "-"],
            ["b-degeri (3y)", f"{sf['b_all_3y']:.4f}" if sf.get("b_all_3y") else "-"],
            ["b Trendi", f"{sf['b_trend']:.5f}" if sf.get("b_trend") is not None else "-"],
            ["b Azaliyor mu?", str(sf.get("b_decreasing", "-"))],
            ["Maks Mw (1y)", f"{sf['w1_max_mw']:.1f}" if sf.get("w1_max_mw") else "-"],
        ]
        seis_table = Table(seis_data, colWidths=[150, 320])
        seis_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), COLORS["muted"]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e0e0e0")),
        ]))
        story.append(seis_table)
        story.append(Spacer(1, 8))

    # === FAY BILGISI ===
    faults = data.get("faults")
    if faults:
        ff = faults.get("features", {})
        if ff:
            story.append(Paragraph("FAY BILGISI", styles["SectionTitle"]))
            fault_data = [
                ["En Yakin Fay", ff.get("nearest_fault_name", "-")],
                ["Mesafe", f"{ff.get('nearest_fault_dist_km', '-')} km"],
                ["Fay Tipi", ff.get("nearest_fault_type", "-")],
                ["300km Fay Sayisi", str(ff.get("n_faults_within_300km", "-"))],
                ["Toplam Fay Uzunlugu", f"{ff.get('total_fault_length_300km', '-')} km"],
                ["Karmasiklik", str(ff.get("fault_complexity", "-"))],
            ]
            f_table = Table(fault_data, colWidths=[150, 320])
            f_table.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), COLORS["muted"]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e0e0e0")),
            ]))
            story.append(f_table)
            story.append(Spacer(1, 8))

    # === COULOMB STRES ===
    cff = data.get("cff")
    if cff and not cff.get("error"):
        story.append(Paragraph("COULOMB STRES TRANSFERI", styles["SectionTitle"]))
        cff_data = [
            ["Toplam CFF", f"{cff.get('cff_total', 0):.6f} bar"],
            ["Kaynak Sayisi", str(cff.get("cff_n_sources", 0))],
            ["Maks Tekil", f"{cff.get('cff_max_single', 0):.6f} bar"],
            ["En Yakin Kaynak", f"{cff.get('cff_nearest_source_km', '-')} km (Mw {cff.get('cff_nearest_source_mw', '?'):.1f})" if cff.get("cff_nearest_source_km") else "-"],
        ]
        c_table = Table(cff_data, colWidths=[150, 320])
        c_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), COLORS["muted"]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e0e0e0")),
        ]))
        story.append(c_table)
        if cff.get("cff_total", 0) > 1.0:
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                "UYARI: Yuksek Coulomb stres transferi tespit edildi. "
                "Bu bolgedeki faylar uzerinde ek gerilim birikimi olabilir.",
                styles["Warning"]
            ))
        story.append(Spacer(1, 8))

    # === GPS ===
    gps = data.get("gps")
    if gps and not gps.get("error"):
        story.append(Paragraph("GPS DEFORMASYON", styles["SectionTitle"]))
        gps_data = [
            ["Istasyon Sayisi", str(gps.get("gps_n_stations", 0))],
            ["Ort. Yatay Hiz", f"{gps.get('gps_mean_vh_mm_yr', '-')} mm/yil"],
            ["Maks Yatay Hiz", f"{gps.get('gps_max_vh_mm_yr', '-')} mm/yil"],
            ["Strain Rate", str(gps.get("gps_strain_rate", "-"))],
            ["En Yakin Istasyon", f"{gps.get('gps_nearest_site', '-')} ({gps.get('gps_nearest_dist_km', '-')} km)"],
        ]
        g_table = Table(gps_data, colWidths=[150, 320])
        g_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), COLORS["muted"]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e0e0e0")),
        ]))
        story.append(g_table)
        story.append(Spacer(1, 8))

    # === TARIHSEL ONCU ===
    nlp = data.get("nlp")
    if nlp:
        hist = nlp.get("historical", [])
        if hist:
            story.append(Paragraph("TARIHSEL ONCU KAYITLARI", styles["SectionTitle"]))
            for h in hist[:3]:
                story.append(Paragraph(
                    f"<b>{h['event']}</b> ({h['distance_km']} km)",
                    styles["BodyText2"]
                ))
                for p in h["precursors"]:
                    story.append(Paragraph(
                        f"  [{p['type']}] {p['detail']}",
                        styles["SmallText"]
                    ))
                story.append(Paragraph(
                    f"  Kaynak: {h['source']}",
                    styles["SmallText"]
                ))
                story.append(Spacer(1, 4))

    # === DUAL RISK ===
    qar = data.get("quality_adjusted")
    if qar:
        story.append(Paragraph("QUALITY-ADJUSTED SHORT-TERM RISK", styles["SectionTitle"]))
        qar_data = [
            ["Ham Skor (standard)", f"{(qar.get('st_raw',0) or 0)*100:.1f}%"],
            ["Debiased Skor", f"{(qar.get('st_debiased',0) or 0)*100:.1f}%"],
            ["Ayarlanmis Skor", f"{(qar.get('st_adjusted',0) or 0)*100:.1f}%"],
            ["Kalite Cezasi", f"-{(qar.get('quality_penalty',0) or 0)*100:.1f}%"],
            ["Tektonik Cezasi", f"-{(qar.get('tectonic_penalty',0) or 0)*100:.1f}%"],
            ["Guven", str(qar.get('confidence','?'))],
            ["Tektonik Sinif", str(qar.get('tectonic_class','?'))],
        ]
        q_table = Table(qar_data, colWidths=[150, 320])
        q_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), COLORS["muted"]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e0e0e0")),
        ]))
        story.append(q_table)
        story.append(Spacer(1, 8))

    # === HAZARD SKORLARI ===
    hz = data.get("hazards")
    if hz:
        story.append(Paragraph("UFUK BAZLI TEHLIKE SKORLARI", styles["SectionTitle"]))
        story.append(Paragraph(
            "Bu skorlar deterministik zaman tahmini degildir. "
            "Segment riski, sismik anomali, CFF ve NLP faktorlerinin "
            "Poisson modeli uzerindeki carpani ile hesaplanmistir.",
            styles["SmallText"]
        ))
        story.append(Spacer(1, 4))
        hz_data = [
            ["30 gun icinde", f"%{hz.get('30d','?')}"],
            ["90 gun icinde", f"%{hz.get('90d','?')}"],
            ["1 yil icinde", f"%{hz.get('1y','?')}"],
            ["5 yil icinde", f"%{hz.get('5y','?')}"],
        ]
        if hz.get("multiplier"):
            hz_data.append(["Toplam carpan", str(hz.get("multiplier"))])

        h_table = Table(hz_data, colWidths=[150, 320])
        h_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), COLORS["muted"]),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e0e0e0")),
        ]))
        story.append(h_table)
        story.append(Spacer(1, 8))

    # === BOLGE BAGLAMI ===
    if zone:
        story.append(Paragraph("BOLGE BAGLAMI", styles["SectionTitle"]))
        zone_data = [
            ["Bolge", zone["name"]],
            ["Fay", f"{zone['fault_name']} ({zone['fault_type']})"],
            ["Beklenen Mw", zone["expected_mw"]],
            ["Son Buyuk Deprem", f"{zone['last_major']} (Mw {zone['last_major_mw']})"],
            ["Tekrarlama", f"~{zone['recurrence_years']} yil"],
            ["Nufus Riski", zone.get("population_risk", "-")],
        ]

        try:
            last_dt = datetime.strptime(zone["last_major"], "%Y-%m-%d")
            yrs = (datetime.utcnow() - last_dt).days / 365.25
            zone_data.append(["Gecen Sure", f"{yrs:.1f} yil"])

            rec = zone["recurrence_years"]
            ratio = yrs / rec
            if ratio > 0.5: phase = "GEC"
            elif ratio > 0.3: phase = "ORTA"
            else: phase = "ERKEN"
            zone_data.append(["Sismik Faz", f"{phase} (oran: {ratio:.3f})"])

            lam = 1.0 / rec
            for w in [1, 5, 10, 30]:
                prob = (1 - math.exp(-lam * w)) * 100
                zone_data.append([f"Poisson {w} yil", f"%{prob:.1f}"])
        except: pass

        z_table = Table(zone_data, colWidths=[150, 320])
        z_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), COLORS["muted"]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e0e0e0")),
        ]))
        story.append(z_table)

    # === YASAL UYARI ===
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=COLORS["border"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "YASAL UYARI: Bu rapor deprem tahmini degildir. Hicbir bilimsel "
        "model depremin ne zaman olacagini kesin olarak soyleyemez. "
        "Bu arac, sismik oruntulerdeki anormallikleri tespit ederek "
        "farkindalik saglar. Afet hazirlik kararlari icin resmi kurumlarin "
        "(AFAD, USGS, JMA vb.) rehberligine basvurunuz.",
        styles["Warning"]
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"SeismoPattern v4 | Model AUC: 0.9087 | "
        f"Rapor: {ts} | "
        f"GCMT 1976-2025 | ISC+USGS",
        styles["SmallText"]
    ))

    # PDF olustur
    doc.build(story)
    print(f"PDF olusturuldu: {output_path}")
    return str(output_path)


# =========================================================
# APP ENTEGRASYONU
# =========================================================

def get_pdf_cache_path(lat, lon, ref_date=None):
    import re as _re

    tag = f"{lat:.4f}_{lon:.4f}"
    if ref_date:
        tag += f"_{ref_date}"

    safe_tag = _re.sub(r"[^a-zA-Z0-9_\-]", "_", tag)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"SeismoPattern_cache_{safe_tag}.pdf"


_PDF_CACHE_TTL = 3600  # 1 saat

def generate_pdf_for_app(lat, lon, ref_date=None):
    """app.py'den cagirilacak fonksiyon. Deterministik disk cache ile."""
    import re as _re

    tag = f"{lat:.4f}_{lon:.4f}"
    if ref_date:
        tag += f"_{ref_date}"

    safe_tag = _re.sub(r"[^a-zA-Z0-9_\-]", "_", tag)
    cache_path = get_pdf_cache_path(lat, lon, ref_date)

    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < _PDF_CACHE_TTL:
            return str(cache_path)
        try:
            cache_path.unlink()
        except Exception:
            pass

    data = collect_data(lat, lon, ref_date)
    path = generate_pdf(data, str(cache_path))
    return str(path)


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--refdate", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    print(f"Veri toplanıyor: {args.lat}, {args.lon}")
    data = collect_data(args.lat, args.lon, args.refdate)

    print("PDF olusturuluyor...")
    path = generate_pdf(data, args.output)

    print(f"\nTamamlandi: {path}")


if __name__ == "__main__":
    main()
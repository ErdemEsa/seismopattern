#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Seviye 4.3: Uyari Sistemi
==========================================
Risk degisimlerini izler, uyari uretir, loglar.

Uyari tipleri:
  1. LEVEL_UP    : Risk seviyesi yukseldi
  2. SCORE_SPIKE : Skor %10+ artti
  3. NEW_HIGH    : Ilk kez YUKSEK/KRITIK
  4. B_DROP      : b-degeri belirgin dustu
  5. CFF_HIGH    : Coulomb stres yuksek

Kullanim:
  python scripts/alert_system.py --check
  python scripts/alert_system.py --history
  python scripts/alert_system.py --clear
"""

import json
import sqlite3
import smtplib
import argparse
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path("output/monitor.db")
ALERT_LOG = Path("output/alerts.json")
CONFIG_PATH = Path("output/alert_config.json")


# =========================================================
# DEFAULT CONFIG
# =========================================================

DEFAULT_CONFIG = {
    "score_spike_threshold": 0.10,
    "high_levels": ["KRITIK", "YUKSEK"],
    "b_drop_threshold": 0.15,
    "cff_threshold": 1.0,
    "email_enabled": False,
    "email_smtp": "smtp.gmail.com",
    "email_port": 587,
    "email_user": "",
    "email_pass": "",
    "email_to": [],
    "max_alerts_per_day": 20,
}


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        # Eksik alanlari default ile tamamla
        for k, v in DEFAULT_CONFIG.items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# =========================================================
# UYARI KONTROL
# =========================================================

def check_alerts():
    """Monitor veritabanindan uyarilari kontrol et."""
    if not DB_PATH.exists():
        print("Monitor veritabani yok. Once auto_monitor.py --once calistirin.")
        return []

    cfg = load_config()
    conn = sqlite3.connect(str(DB_PATH))

    # Her bolge icin son iki tarama sonucunu al
    zones = conn.execute(
        "SELECT DISTINCT zone_key FROM scan_results"
    ).fetchall()

    alerts = []
    ts = datetime.utcnow().isoformat()

    for (zone_key,) in zones:
        rows = conn.execute("""
            SELECT zone_name, risk_score, risk_level, pattern_type,
                   b_value, cff_total, n_events, timestamp
            FROM scan_results
            WHERE zone_key=?
            ORDER BY id DESC LIMIT 2
        """, (zone_key,)).fetchall()

        if not rows:
            continue

        curr = rows[0]
        prev = rows[1] if len(rows) > 1 else None

        name = curr[0]
        score = curr[1] or 0
        level = curr[2] or "DUSUK"
        pattern = curr[3]
        b_val = curr[4]
        cff = curr[5] or 0
        n_ev = curr[6]
        scan_ts = curr[7]

        # 1. LEVEL_UP: Seviye yukseldi
        if prev:
            old_level = prev[2] or "DUSUK"
            level_order = {"DUSUK": 0, "DIKKAT": 1, "ORTA": 2,
                          "YUKSEK": 3, "KRITIK": 4}
            if level_order.get(level, 0) > level_order.get(old_level, 0):
                alerts.append({
                    "timestamp": ts,
                    "zone": zone_key,
                    "zone_name": name,
                    "type": "LEVEL_UP",
                    "severity": "HIGH" if level in cfg["high_levels"] else "MEDIUM",
                    "message": f"{name}: Seviye yukseldi ({old_level} -> {level})",
                    "old_score": prev[1],
                    "new_score": score,
                })

        # 2. SCORE_SPIKE: Skor ani artis
        if prev and prev[1] is not None:
            delta = score - prev[1]
            if delta >= cfg["score_spike_threshold"]:
                alerts.append({
                    "timestamp": ts,
                    "zone": zone_key,
                    "zone_name": name,
                    "type": "SCORE_SPIKE",
                    "severity": "HIGH" if delta >= 0.20 else "MEDIUM",
                    "message": f"{name}: Risk skoru {delta*100:.1f}% artti "
                              f"({prev[1]*100:.1f}% -> {score*100:.1f}%)",
                    "old_score": prev[1],
                    "new_score": score,
                })

        # 3. NEW_HIGH: Ilk kez yuksek seviye
        if level in cfg["high_levels"]:
            if prev is None or prev[2] not in cfg["high_levels"]:
                alerts.append({
                    "timestamp": ts,
                    "zone": zone_key,
                    "zone_name": name,
                    "type": "NEW_HIGH",
                    "severity": "CRITICAL",
                    "message": f"ALARM: {name} {level} seviyesine ulasti! "
                              f"({score*100:.1f}%)",
                    "new_score": score,
                })

        # 4. CFF_HIGH: Yuksek Coulomb stres
        if cff >= cfg["cff_threshold"]:
            alerts.append({
                "timestamp": ts,
                "zone": zone_key,
                "zone_name": name,
                "type": "CFF_HIGH",
                "severity": "MEDIUM",
                "message": f"{name}: Yuksek Coulomb stres ({cff:.4f} bar)",
                "cff_value": cff,
            })

        # 5. B_DROP: b-degeri dususu
        if prev and b_val is not None and prev[4] is not None:
            b_drop = prev[4] - b_val
            if b_drop >= cfg["b_drop_threshold"]:
                alerts.append({
                    "timestamp": ts,
                    "zone": zone_key,
                    "zone_name": name,
                    "type": "B_DROP",
                    "severity": "HIGH",
                    "message": f"{name}: b-degeri belirgin dustu "
                              f"({prev[4]:.3f} -> {b_val:.3f})",
                    "old_b": prev[4],
                    "new_b": b_val,
                })

    # Uyarilari veritabanina kaydet
    for alert in alerts:
        conn.execute("""
            INSERT INTO alerts (timestamp, zone_key, zone_name,
                               alert_type, message, old_score, new_score)
            VALUES (?,?,?,?,?,?,?)
        """, (
            alert["timestamp"],
            alert["zone"],
            alert["zone_name"],
            alert["type"],
            alert["message"],
            alert.get("old_score"),
            alert.get("new_score"),
        ))
    conn.commit()

    # JSON log
    if alerts:
        existing = []
        if ALERT_LOG.exists():
            try:
                existing = json.loads(ALERT_LOG.read_text())
            except Exception:
                pass
        existing.extend(alerts)
        # Son 1000 uyariyi tut
        existing = existing[-1000:]
        ALERT_LOG.write_text(json.dumps(existing, indent=2, default=str))

    conn.close()
    return alerts


def send_email_alert(alerts, cfg):
    """E-posta uyarisi gonder (opsiyonel)."""
    if not cfg.get("email_enabled"):
        return

    if not cfg.get("email_user") or not cfg.get("email_to"):
        return

    body = "SeismoPattern Uyari Raporu\n"
    body += f"Tarih: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    body += "=" * 50 + "\n\n"

    for a in alerts:
        sev = a.get("severity", "?")
        body += f"[{sev}] {a['message']}\n"

    msg = MIMEText(body)
    msg["Subject"] = f"SeismoPattern: {len(alerts)} uyari"
    msg["From"] = cfg["email_user"]
    msg["To"] = ", ".join(cfg["email_to"])

    try:
        with smtplib.SMTP(cfg["email_smtp"], cfg["email_port"]) as s:
            s.starttls()
            s.login(cfg["email_user"], cfg["email_pass"])
            s.send_message(msg)
        print(f"  E-posta gonderildi: {len(alerts)} uyari")
    except Exception as e:
        print(f"  E-posta hatasi: {e}")


def get_recent_alerts(days=7):
    """Son N gunun uyarilarini getir."""
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(str(DB_PATH))
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    rows = conn.execute("""
        SELECT timestamp, zone_key, zone_name, alert_type,
               message, old_score, new_score
        FROM alerts
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
    """, (cutoff,)).fetchall()

    conn.close()

    return [{
        "timestamp": r[0],
        "zone": r[1],
        "zone_name": r[2],
        "type": r[3],
        "message": r[4],
        "old_score": r[5],
        "new_score": r[6],
    } for r in rows]


def get_alerts_for_app():
    """app.py'den cagirilacak fonksiyon."""
    return {
        "recent": get_recent_alerts(7),
        "total_7d": len(get_recent_alerts(7)),
        "total_30d": len(get_recent_alerts(30)),
    }


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="Uyarilari kontrol et")
    ap.add_argument("--history", action="store_true",
                    help="Uyari gecmisini goster")
    ap.add_argument("--clear", action="store_true",
                    help="Uyari gecmisini temizle")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--setup-email", action="store_true",
                    help="E-posta ayarlarini yapilandir")
    args = ap.parse_args()

    if args.check:
        print("Uyari kontrolu...")
        alerts = check_alerts()
        if alerts:
            print(f"\n{len(alerts)} UYARI:")
            for a in alerts:
                sev = a.get("severity", "?")
                print(f"  [{sev}] {a['message']}")

            cfg = load_config()
            if cfg.get("email_enabled"):
                send_email_alert(alerts, cfg)
        else:
            print("Uyari yok.")

    elif args.history:
        alerts = get_recent_alerts(args.days)
        print(f"Son {args.days} gun: {len(alerts)} uyari")
        for a in alerts:
            print(f"  [{a['timestamp'][:16]}] [{a['type']}] {a['message']}")

    elif args.clear:
        if DB_PATH.exists():
            conn = sqlite3.connect(str(DB_PATH))
            conn.execute("DELETE FROM alerts")
            conn.commit()
            conn.close()
        if ALERT_LOG.exists():
            ALERT_LOG.write_text("[]")
        print("Uyari gecmisi temizlendi.")

    elif args.setup_email:
        cfg = load_config()
        print("E-posta ayarlari:")
        cfg["email_enabled"] = input("  Aktif? (e/h): ").lower() == "e"
        if cfg["email_enabled"]:
            cfg["email_smtp"] = input(f"  SMTP [{cfg['email_smtp']}]: ") or cfg["email_smtp"]
            cfg["email_port"] = int(input(f"  Port [{cfg['email_port']}]: ") or cfg["email_port"])
            cfg["email_user"] = input("  Kullanici: ")
            cfg["email_pass"] = input("  Sifre: ")
            to = input("  Alici (virgul ile): ")
            cfg["email_to"] = [t.strip() for t in to.split(",")]
        save_config(cfg)
        print("Ayarlar kaydedildi.")

    else:
        print("Kullanim:")
        print("  python scripts/alert_system.py --check")
        print("  python scripts/alert_system.py --history")
        print("  python scripts/alert_system.py --setup-email")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
SeismoPattern - GCMT Catalog Parser
====================================
Global CMT kataloğunu parse eder (İngilizce + Türkçe çeviri formatları)
Çıktı: Yapılandırılmış DataFrame + SQLite veritabanı

Kullanım:
    python seismo_parser.py --input data/ --output seismo.db
"""

import re
import os
import sys
import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class GCMTParser:
    """
    Global CMT Kataloğu Parser
    - Orijinal İngilizce format
    - Türkçe çevrilmiş format
    - Karma formatlar
    """
    
    def __init__(self):
        self.records = []
        self.parse_errors = []
        self.duplicate_count = 0
        
        # Türkçe → İngilizce eşleme tablosu
        self.tr_to_en = {
            'MERKEZ NOKTASI:': 'CENTROID:',
            'ÜCRETSİZ': 'FREE',
            'DÜZELTME': 'FIX',
            'SERBEST': 'FREE',
        }
        
    def normalize_line(self, line: str) -> str:
        """Türkçe terimleri İngilizce'ye çevir ve ondalık ayırıcıları düzelt"""
        normalized = line
        
        # Türkçe terimleri değiştir
        for tr, en in self.tr_to_en.items():
            normalized = normalized.replace(tr, en)
        
        return normalized
    
    def fix_decimal_separators(self, line: str, line_type: str) -> str:
        """
        Türkçe formattaki virgüllü ondalık sayıları noktaya çevir.
        DİKKAT: Sadece sayısal satırlarda (Satır 4, 5) yapılmalı.
        Satır 2'deki virgüller alan ayırıcısı olabilir.
        """
        if line_type in ('tensor', 'eigen'):
            # Sayılar arasındaki virgülleri tespit et
            # Patern: rakam,rakam (örn: 3,084)
            normalized = re.sub(r'(\d),(\d)', r'\1.\2', line)
            return normalized
        return line
    
    def detect_line_type(self, line: str) -> str:
        """Satır tipini belirle"""
        stripped = line.strip()
        
        if not stripped:
            return 'empty'
        
        # Satır 1: Hypocenter (MLI, PDEW, SWEQ, PDEC vb. ile başlar)
        if re.match(r'^[A-Z]{2,5}\s+\d{4}/', stripped):
            return 'hypocenter'
        
        # Satır 2: Event ID (B: ile body wave bilgisi içerir)
        if re.search(r'B:\s*\d+', stripped) and re.search(r'CMT:', stripped):
            return 'event_info'
        
        # Satır 3: Centroid
        if 'CENTROID:' in stripped or 'MERKEZ NOKTASI:' in stripped:
            return 'centroid'
        
        # Satır 4: Moment tensor (üs ile başlar: 23, 24, 25, 26, 27, 28)
        if re.match(r'^\s*2[3-8]\s+[-]?\d+', stripped):
            return 'tensor'
        
        # Satır 5: Eigenvalue (V10 ile başlar)
        if re.match(r'^\s*V10', stripped):
            return 'eigen'
        
        return 'unknown'
    
    def parse_hypocenter_line(self, line: str) -> Dict:
        """
        Satır 1: Hypocenter bilgileri
        Örnek: MLI  1976/01/01 01:29:39.6 -28.61 -177.64  59.0 6.2 0.0 KERMADEC ISLANDS REGION
        """
        data = {}
        try:
            # Kaynak tipi
            data['source_type'] = line[0:5].strip()
            
            # Tarih ve saat
            date_str = line[5:16].strip()  # 1976/01/01
            time_str = line[17:27].strip()  # 01:29:39.6
            
            try:
                data['datetime_utc'] = datetime.strptime(
                    f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S.%f"
                )
            except ValueError:
                # Bazı kayıtlarda farklı format olabilir
                try:
                    # Virgüllü format: 16:49:44,2
                    time_str_fixed = time_str.replace(',', '.')
                    data['datetime_utc'] = datetime.strptime(
                        f"{date_str} {time_str_fixed}", "%Y/%m/%d %H:%M:%S.%f"
                    )
                except ValueError:
                    data['datetime_utc'] = datetime.strptime(
                        date_str, "%Y/%m/%d"
                    )
            
            # Koordinatlar ve büyüklükler
            # Pozisyon bazlı parsing (sabit genişlik format)
            rest = line[27:]
            parts = rest.split()
            
            if len(parts) >= 4:
                # Lat, Lon, Depth, Mb, Ms
                lat_str = parts[0].replace(',', '.')
                lon_str = parts[1].replace(',', '.')
                dep_str = parts[2].replace(',', '.')
                
                data['hypo_lat'] = float(lat_str)
                data['hypo_lon'] = float(lon_str)
                data['hypo_depth_km'] = float(dep_str)
                
                if len(parts) >= 5:
                    data['mb'] = float(parts[3].replace(',', '.'))
                if len(parts) >= 6:
                    data['ms'] = float(parts[4].replace(',', '.'))
                
                # Bölge adı (kalan kısım)
                if len(parts) >= 7:
                    data['region'] = ' '.join(parts[5:]).strip()
                else:
                    data['region'] = ''
            
        except Exception as e:
            data['parse_error_line1'] = str(e)
        
        return data
    
    def parse_event_info_line(self, line: str) -> Dict:
        """
        Satır 2: Event ID ve istasyon bilgileri
        Örnek: M010176A  B:  0  0  0  S:  0  0  0  M: 12  30 135 CMT: 1 BOXHD:  9.4
        """
        data = {}
        try:
            # Event ID
            data['event_id'] = line[0:16].strip()
            
            # Half duration
            hd_match = re.search(r'(?:BOXHD|TRIHD):\s*([\d.]+)', line)
            if hd_match:
                data['half_duration'] = float(hd_match.group(1))
                data['source_function'] = 'BOX' if 'BOXHD' in line else 'TRI'
            
            # Body wave istasyon sayısı
            b_match = re.search(r'B:\s*(\d+)\s+(\d+)\s+(\d+)', line)
            if b_match:
                data['n_body_stations'] = int(b_match.group(1))
                data['n_body_components'] = int(b_match.group(2))
                data['body_period'] = int(b_match.group(3))
            
            # Surface wave istasyon sayısı
            s_match = re.search(r'S:\s*(\d+)\s+(\d+)\s+(\d+)', line)
            if s_match:
                data['n_surface_stations'] = int(s_match.group(1))
            
            # Mantle wave istasyon sayısı
            m_match = re.search(r'M:\s*(\d+)\s+(\d+)\s+(\d+)', line)
            if m_match:
                data['n_mantle_stations'] = int(m_match.group(1))
                
        except Exception as e:
            data['parse_error_line2'] = str(e)
        
        return data
    
    def parse_centroid_line(self, line: str) -> Dict:
        """
        Satır 3: Centroid bilgileri
        Örnek: CENTROID:  13.8 0.2 -29.25 0.02 -176.96 0.01  47.8  0.6 FREE
        """
        data = {}
        try:
            normalized = self.normalize_line(line)
            
            # CENTROID: sonrasını al
            centroid_match = re.search(
                r'CENTROID:\s+([-\d.,]+)\s+([-\d.,]+)\s+([-\d.,]+)\s+'
                r'([-\d.,]+)\s+([-\d.,]+)\s+([-\d.,]+)\s+([-\d.,]+)\s+'
                r'([-\d.,]+)\s+(FREE|FIX)',
                normalized
            )
            
            if centroid_match:
                data['time_shift'] = float(centroid_match.group(1).replace(',', '.'))
                data['time_shift_err'] = float(centroid_match.group(2).replace(',', '.'))
                data['centroid_lat'] = float(centroid_match.group(3).replace(',', '.'))
                data['centroid_lat_err'] = float(centroid_match.group(4).replace(',', '.'))
                data['centroid_lon'] = float(centroid_match.group(5).replace(',', '.'))
                data['centroid_lon_err'] = float(centroid_match.group(6).replace(',', '.'))
                data['centroid_depth_km'] = float(centroid_match.group(7).replace(',', '.'))
                data['centroid_depth_err'] = float(centroid_match.group(8).replace(',', '.'))
                data['depth_type'] = centroid_match.group(9)  # FREE or FIX
                
        except Exception as e:
            data['parse_error_line3'] = str(e)
        
        return data
    
    def parse_tensor_line(self, line: str) -> Dict:
        """
        Satır 4: Moment tensor bileşenleri
        Örnek: 26  7.680 0.090  0.090 0.060 -7.770 0.070  1.390 0.160  4.520 0.160 -3.260 0.060
        """
        data = {}
        try:
            # Virgülleri noktaya çevir
            fixed_line = self.fix_decimal_separators(line, 'tensor')
            parts = fixed_line.split()
            
            if len(parts) >= 13:
                data['exponent'] = int(parts[0])
                data['mrr'] = float(parts[1])
                data['mrr_err'] = float(parts[2])
                data['mtt'] = float(parts[3])
                data['mtt_err'] = float(parts[4])
                data['mpp'] = float(parts[5])
                data['mpp_err'] = float(parts[6])
                data['mrt'] = float(parts[7])
                data['mrt_err'] = float(parts[8])
                data['mrp'] = float(parts[9])
                data['mrp_err'] = float(parts[10])
                data['mtp'] = float(parts[11])
                data['mtp_err'] = float(parts[12])
                
                # Scalar Moment hesapla (dyne·cm)
                exp = data['exponent']
                mrr = data['mrr'] * (10 ** exp)
                mtt = data['mtt'] * (10 ** exp)
                mpp = data['mpp'] * (10 ** exp)
                mrt = data['mrt'] * (10 ** exp)
                mrp = data['mrp'] * (10 ** exp)
                mtp = data['mtp'] * (10 ** exp)
                
                # M0 = sqrt(sum of squares / 2)
                m0 = np.sqrt(0.5 * (mrr**2 + mtt**2 + mpp**2 + 
                                     2*(mrt**2 + mrp**2 + mtp**2)))
                
                data['scalar_moment_dyncm'] = m0
                
                # Mw hesapla: Mw = (2/3) * log10(M0) - 10.7
                # M0 dyne·cm cinsinden
                if m0 > 0:
                    data['mw'] = (2.0/3.0) * np.log10(m0) - 10.7
                else:
                    data['mw'] = None
                    
        except Exception as e:
            data['parse_error_line4'] = str(e)
        
        return data
    
    def parse_eigen_line(self, line: str) -> Dict:
        """
        Satır 5: Eigenvalue ve Nodal düzlem bilgileri
        Örnek: V10   8.940 75 283   1.260  2  19 -10.190 15 110   9.560 202 30   93  18 60   88
        """
        data = {}
        try:
            fixed_line = self.fix_decimal_separators(line, 'eigen')
            parts = fixed_line.split()
            
            if len(parts) >= 17:
                # Version
                data['version'] = parts[0]
                
                # T-axis (tension)
                data['t_val'] = float(parts[1])
                data['t_plunge'] = int(parts[2])
                data['t_azimuth'] = int(parts[3])
                
                # N-axis (null)
                data['n_val'] = float(parts[4])
                data['n_plunge'] = int(parts[5])
                data['n_azimuth'] = int(parts[6])
                
                # P-axis (pressure)
                data['p_val'] = float(parts[7])
                data['p_plunge'] = int(parts[8])
                data['p_azimuth'] = int(parts[9])
                
                # Scalar Moment (from eigenvalues)
                exp_from_parent = data.get('exponent', 0)
                data['scalar_moment_eigen'] = float(parts[10])
                
                # Nodal Plane 1
                data['np1_strike'] = int(parts[11])
                data['np1_dip'] = int(parts[12])
                data['np1_rake'] = int(parts[13])
                
                # Nodal Plane 2
                data['np2_strike'] = int(parts[14])
                data['np2_dip'] = int(parts[15])
                data['np2_rake'] = int(parts[16])
                
                # FAY TİPİNİ BELİRLE (Rake açısından)
                data['fault_type'] = self.classify_fault(data['np1_rake'])
                
        except Exception as e:
            data['parse_error_line5'] = str(e)
        
        return data
    
    @staticmethod
    def classify_fault(rake: int) -> str:
        """
        Rake açısından fay tipini belirle
        
        Rake açısı kuralları:
        -  -30° < rake <  30°  → Sol yanal doğrultu atımlı (Left-lateral SS)
        -  150° < rake < 210° → Sol yanal doğrultu atımlı (Left-lateral SS)  
        -   30° < rake < 150° → Ters fay (Reverse/Thrust)
        - -150° < rake < -30° → Normal fay (Normal)
        -  150° < rake        veya rake < -150° → Sağ yanal doğrultu atımlı (Right-lateral SS)
        
        Daha detaylı sınıflandırma:
        """
        # Rake'i -180 ile 180 arasına normalize et
        while rake > 180:
            rake -= 360
        while rake < -180:
            rake += 360
        
        abs_rake = abs(rake)
        
        if abs_rake <= 30:
            return 'STRIKE_SLIP'  # Doğrultu atımlı
        elif abs_rake >= 150:
            return 'STRIKE_SLIP'  # Doğrultu atımlı
        elif 30 < rake < 150:
            return 'REVERSE'  # Ters fay (bindirme)
        elif -150 < rake < -30:
            return 'NORMAL'  # Normal fay
        elif 30 < abs_rake < 60:
            return 'OBLIQUE_SS'  # Eğik doğrultu atımlı
        elif 120 < abs_rake < 150:
            return 'OBLIQUE_SS'  # Eğik doğrultu atımlı
        else:
            return 'UNKNOWN'
    
    def parse_file(self, filepath: str) -> List[Dict]:
        """Tek bir dosyayı parse et"""
        print(f"\n{'='*60}")
        print(f"DOSYA OKUNUYOR: {filepath}")
        print(f"{'='*60}")
        
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        print(f"  Toplam satır sayısı: {len(lines):,}")
        
        records = []
        current_record = {}
        line_sequence = 0  # 0-4 arası döngü (5 satırlık bloklar)
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip('\n')
            
            # Boş satırları atla
            if not line.strip():
                i += 1
                continue
            
            line_type = self.detect_line_type(line)
            
            if line_type == 'hypocenter':
                # Yeni kayıt başlıyor
                if current_record and 'datetime_utc' in current_record:
                    records.append(current_record)
                
                current_record = self.parse_hypocenter_line(line)
                line_sequence = 1
                
            elif line_type == 'event_info' and line_sequence == 1:
                current_record.update(self.parse_event_info_line(line))
                line_sequence = 2
                
            elif line_type == 'centroid' and line_sequence == 2:
                current_record.update(self.parse_centroid_line(line))
                line_sequence = 3
                
            elif line_type == 'tensor' and line_sequence == 3:
                current_record.update(self.parse_tensor_line(line))
                line_sequence = 4
                
            elif line_type == 'eigen' and line_sequence == 4:
                # Exponent bilgisini eigen'a aktar
                eigen_data = self.parse_eigen_line(line)
                if 'exponent' in current_record:
                    eigen_data['exponent'] = current_record['exponent']
                current_record.update(eigen_data)
                line_sequence = 0  # Kayıt tamamlandı
                
            else:
                # Beklenmeyen satır - hata kaydı
                if line.strip():
                    self.parse_errors.append({
                        'file': filepath,
                        'line_number': i + 1,
                        'expected_sequence': line_sequence,
                        'detected_type': line_type,
                        'content': line[:80]
                    })
            
            i += 1
        
        # Son kaydı ekle
        if current_record and 'datetime_utc' in current_record:
            records.append(current_record)
        
        print(f"  Parse edilen kayıt sayısı: {len(records):,}")
        print(f"  Hata sayısı: {len(self.parse_errors):,}")
        
        return records
    
    def parse_all_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Tüm dosyaları parse et ve DataFrame oluştur"""
        all_records = []
        
        for filepath in file_paths:
            records = self.parse_file(filepath)
            all_records.extend(records)
        
        print(f"\n{'='*60}")
        print(f"TOPLAM PARSE EDİLEN KAYIT: {len(all_records):,}")
        
        # DataFrame oluştur
        df = pd.DataFrame(all_records)
        
        # Tarih sütununu index yap
        if 'datetime_utc' in df.columns:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
            df = df.sort_values('datetime_utc').reset_index(drop=True)
        
        # Duplikasyon kontrolü (event_id bazında)
        if 'event_id' in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset='event_id', keep='first')
            self.duplicate_count = before - len(df)
            print(f"  Kaldırılan duplike kayıt: {self.duplicate_count:,}")
        
        print(f"  Final kayıt sayısı: {len(df):,}")
        print(f"  Tarih aralığı: {df['datetime_utc'].min()} → {df['datetime_utc'].max()}")
        
        if 'mw' in df.columns:
            print(f"  Mw aralığı: {df['mw'].min():.1f} → {df['mw'].max():.1f}")
            print(f"  Mw 7.0+ deprem sayısı: {len(df[df['mw'] >= 7.0]):,}")
        
        return df
    

class SeismoDatabase:
    """SQLite veritabanı yönetimi"""
    
    def __init__(self, db_path: str = 'seismo_pattern.db'):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        print(f"Veritabanı bağlantısı: {self.db_path}")
    
    def create_tables(self):
        """Ana tabloları oluştur"""
        self.conn.executescript("""
            -- Ana deprem kataloğu
            CREATE TABLE IF NOT EXISTS earthquakes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE,
                datetime_utc TIMESTAMP,
                
                -- Hypocenter bilgileri
                hypo_lat REAL,
                hypo_lon REAL,
                hypo_depth_km REAL,
                mb REAL,
                ms REAL,
                region TEXT,
                source_type TEXT,
                
                -- Centroid bilgileri
                centroid_lat REAL,
                centroid_lon REAL,
                centroid_depth_km REAL,
                depth_type TEXT,
                time_shift REAL,
                
                -- Büyüklükler
                mw REAL,
                scalar_moment_dyncm REAL,
                exponent INTEGER,
                half_duration REAL,
                
                -- Moment tensor bileşenleri
                mrr REAL, mtt REAL, mpp REAL,
                mrt REAL, mrp REAL, mtp REAL,
                
                -- Eigenvalues ve eksenler
                t_val REAL, t_plunge INTEGER, t_azimuth INTEGER,
                n_val REAL, n_plunge INTEGER, n_azimuth INTEGER,
                p_val REAL, p_plunge INTEGER, p_azimuth INTEGER,
                
                -- Nodal düzlemler
                np1_strike INTEGER, np1_dip INTEGER, np1_rake INTEGER,
                np2_strike INTEGER, np2_dip INTEGER, np2_rake INTEGER,
                
                -- Fay sınıflandırması
                fault_type TEXT,
                
                -- Meta
                n_body_stations INTEGER,
                n_surface_stations INTEGER,
                n_mantle_stations INTEGER
            );
            
            -- Mw 7.0+ büyük depremler (ayrı tablo)
            CREATE TABLE IF NOT EXISTS major_earthquakes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                earthquake_id INTEGER REFERENCES earthquakes(id),
                event_id TEXT,
                datetime_utc TIMESTAMP,
                mw REAL,
                centroid_lat REAL,
                centroid_lon REAL,
                centroid_depth_km REAL,
                region TEXT,
                fault_type TEXT,
                
                -- Analiz pencereleri (sonra doldurulacak)
                precursor_count_1yr INTEGER,
                precursor_count_2yr INTEGER,
                precursor_count_3yr INTEGER,
                
                -- Şablon etiketleri (sonra doldurulacak)
                pattern_type TEXT,
                quiescence_detected BOOLEAN,
                foreshock_detected BOOLEAN,
                b_value_anomaly BOOLEAN,
                migration_detected BOOLEAN
            );
            
            -- Öncü deprem analiz tablosu
            CREATE TABLE IF NOT EXISTS precursor_windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                major_eq_id INTEGER REFERENCES major_earthquakes(id),
                precursor_eq_id INTEGER REFERENCES earthquakes(id),
                
                -- İlişki bilgileri
                days_before INTEGER,
                distance_km REAL,
                azimuth_deg REAL,
                depth_difference_km REAL,
                magnitude_difference REAL,
                
                -- Zaman penceresi
                window_years INTEGER  -- 1, 2 veya 3
            );
            
            -- İndeksler
            CREATE INDEX IF NOT EXISTS idx_eq_datetime ON earthquakes(datetime_utc);
            CREATE INDEX IF NOT EXISTS idx_eq_mw ON earthquakes(mw);
            CREATE INDEX IF NOT EXISTS idx_eq_location ON earthquakes(centroid_lat, centroid_lon);
            CREATE INDEX IF NOT EXISTS idx_eq_event ON earthquakes(event_id);
            CREATE INDEX IF NOT EXISTS idx_major_datetime ON major_earthquakes(datetime_utc);
            CREATE INDEX IF NOT EXISTS idx_precursor_major ON precursor_windows(major_eq_id);
        """)
        print("Veritabanı tabloları oluşturuldu.")
    
    def insert_dataframe(self, df: pd.DataFrame):
        """DataFrame'i veritabanına yaz"""
        # Sütun isimlerini eşle
        columns = [col for col in df.columns if col in self._get_table_columns()]
        
        df_to_insert = df[columns].copy()
        df_to_insert.to_sql('earthquakes', self.conn, if_exists='replace', index=False)
        
        # Büyük depremleri ayrı tabloya kopyala
        major = df[df['mw'] >= 7.0].copy()
        if not major.empty:
            major_records = major[['event_id', 'datetime_utc', 'mw', 
                                    'centroid_lat', 'centroid_lon', 
                                    'centroid_depth_km', 'region', 
                                    'fault_type']].copy()
            major_records.to_sql('major_earthquakes', self.conn, 
                                if_exists='replace', index=False)
        
        self.conn.commit()
        print(f"Veritabanına yazıldı: {len(df):,} kayıt")
        print(f"Büyük deprem (Mw 7.0+): {len(major):,} kayıt")
    
    def _get_table_columns(self):
        """earthquakes tablosundaki sütunları getir"""
        return [
            'event_id', 'datetime_utc', 'hypo_lat', 'hypo_lon', 
            'hypo_depth_km', 'mb', 'ms', 'region', 'source_type',
            'centroid_lat', 'centroid_lon', 'centroid_depth_km', 
            'depth_type', 'time_shift', 'mw', 'scalar_moment_dyncm',
            'exponent', 'half_duration', 'mrr', 'mtt', 'mpp', 
            'mrt', 'mrp', 'mtp', 't_val', 't_plunge', 't_azimuth',
            'n_val', 'n_plunge', 'n_azimuth', 'p_val', 'p_plunge',
            'p_azimuth', 'scalar_moment_eigen', 'np1_strike', 
            'np1_dip', 'np1_rake', 'np2_strike', 'np2_dip', 
            'np2_rake', 'fault_type', 'n_body_stations',
            'n_surface_stations', 'n_mantle_stations'
        ]
    
    def close(self):
        if self.conn:
            self.conn.close()


class DataQualityReport:
    """Veri kalitesi raporu"""
    
    @staticmethod
    def generate(df: pd.DataFrame) -> str:
        report = []
        report.append("\n" + "="*70)
        report.append("VERİ KALİTESİ RAPORU")
        report.append("="*70)
        
        # Genel istatistikler
        report.append(f"\nToplam kayıt: {len(df):,}")
        report.append(f"Tarih aralığı: {df['datetime_utc'].min()} → {df['datetime_utc'].max()}")
        report.append(f"Toplam yıl: {(df['datetime_utc'].max() - df['datetime_utc'].min()).days / 365.25:.1f}")
        
        # Büyüklük dağılımı
        if 'mw' in df.columns:
            report.append(f"\nBÜYÜKLÜK DAĞILIMI:")
            report.append(f"  Mw min: {df['mw'].min():.2f}")
            report.append(f"  Mw max: {df['mw'].max():.2f}")
            report.append(f"  Mw ortalama: {df['mw'].mean():.2f}")
            report.append(f"  Mw medyan: {df['mw'].median():.2f}")
            
            bins = [(5.0, 5.5), (5.5, 6.0), (6.0, 6.5), (6.5, 7.0), 
                    (7.0, 7.5), (7.5, 8.0), (8.0, 8.5), (8.5, 9.5)]
            report.append(f"\n  Büyüklük aralıkları:")
            for low, high in bins:
                count = len(df[(df['mw'] >= low) & (df['mw'] < high)])
                bar = '█' * (count // 100)
                report.append(f"    Mw {low:.1f}-{high:.1f}: {count:>6,} {bar}")
        
        # Derinlik dağılımı
        if 'centroid_depth_km' in df.columns:
            report.append(f"\nDERİNLİK DAĞILIMI:")
            report.append(f"  Min: {df['centroid_depth_km'].min():.1f} km")
            report.append(f"  Max: {df['centroid_depth_km'].max():.1f} km")
            report.append(f"  Ortalama: {df['centroid_depth_km'].mean():.1f} km")
            
            shallow = len(df[df['centroid_depth_km'] <= 70])
            intermediate = len(df[(df['centroid_depth_km'] > 70) & (df['centroid_depth_km'] <= 300)])
            deep = len(df[df['centroid_depth_km'] > 300])
            report.append(f"  Sığ (≤70 km): {shallow:,}")
            report.append(f"  Orta (70-300 km): {intermediate:,}")
            report.append(f"  Derin (>300 km): {deep:,}")
        
        # Fay tipi dağılımı
        if 'fault_type' in df.columns:
            report.append(f"\nFAY TİPİ DAĞILIMI:")
            fault_counts = df['fault_type'].value_counts()
            for ftype, count in fault_counts.items():
                pct = count / len(df) * 100
                report.append(f"  {ftype}: {count:,} ({pct:.1f}%)")
        
        # Eksik veri analizi
        report.append(f"\nEKSİK VERİ ANALİZİ:")
        critical_cols = ['mw', 'centroid_lat', 'centroid_lon', 
                         'centroid_depth_km', 'fault_type',
                         'np1_strike', 'np1_dip', 'np1_rake']
        for col in critical_cols:
            if col in df.columns:
                missing = df[col].isna().sum()
                pct = missing / len(df) * 100
                status = "✅" if pct < 1 else "⚠️" if pct < 5 else "❌"
                report.append(f"  {status} {col}: {missing:,} eksik ({pct:.2f}%)")
        
        # Yıllık dağılım
        report.append(f"\nYILLIK DEPREM SAYISI (Mw 7.0+):")
        if 'mw' in df.columns:
            major = df[df['mw'] >= 7.0].copy()
            major['year'] = major['datetime_utc'].dt.year
            yearly = major.groupby('year').size()
            report.append(f"  Ortalama: {yearly.mean():.1f}/yıl")
            report.append(f"  Min: {yearly.min()} ({yearly.idxmin()})")
            report.append(f"  Max: {yearly.max()} ({yearly.idxmax()})")
        
        return '\n'.join(report)


def main():
    """Ana çalıştırma fonksiyonu"""
    parser = argparse.ArgumentParser(description='SeismoPattern - GCMT Parser')
    parser.add_argument('--input', '-i', nargs='+', required=True,
                        help='Girdi dosya yolları (text1.txt text2.txt text3.txt)')
    parser.add_argument('--output', '-o', default='seismo_pattern.db',
                        help='Çıktı veritabanı dosyası')
    parser.add_argument('--report', '-r', action='store_true',
                        help='Kalite raporu oluştur')
    parser.add_argument('--csv', '-c', default=None,
                        help='CSV olarak da kaydet')
    
    args = parser.parse_args()
    
    # Parser oluştur
    gcmt_parser = GCMTParser()
    
    # Dosyaları parse et
    df = gcmt_parser.parse_all_files(args.input)
    
    # Kalite raporu
    if args.report:
        report = DataQualityReport.generate(df)
        print(report)
        
        # Raporu dosyaya kaydet
        with open('data_quality_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("\nRapor kaydedildi: data_quality_report.txt")
    
    # Veritabanına yaz
    db = SeismoDatabase(args.output)
    db.connect()
    db.create_tables()
    db.insert_dataframe(df)
    db.close()
    
    # CSV'ye de kaydet (opsiyonel)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"CSV kaydedildi: {args.csv}")
    
    # Parse hatalarını kaydet
    if gcmt_parser.parse_errors:
        errors_df = pd.DataFrame(gcmt_parser.parse_errors)
        errors_df.to_csv('parse_errors.csv', index=False)
        print(f"Parse hataları kaydedildi: parse_errors.csv ({len(errors_df)} hata)")
    
    print(f"\n{'='*60}")
    print("PARSER İŞLEMİ TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  Veritabanı: {args.output}")
    print(f"  Toplam kayıt: {len(df):,}")
    if 'mw' in df.columns:
        print(f"  Mw 7.0+ deprem: {len(df[df['mw'] >= 7.0]):,}")
    print(f"  Duplike kaldırılan: {gcmt_parser.duplicate_count:,}")
    print(f"  Parse hatası: {len(gcmt_parser.parse_errors):,}")


if __name__ == '__main__':
    main()
import 'package:flutter/material.dart';

class ZoneModel {
  final String id;
  final String name;
  final String tectonicType;
  final String faultName;
  final String faultType;
  final String region;
  final String expectedMw;
  final String populationRisk;
  final String riskLevel;
  final double? riskScore;
  final double? couplingRatio;
  final double? slipDeficitM;
  final int? lastMajorYear;
  final double? lastMajorMw;
  final double? lat;
  final double? lon;
  final Map<String, dynamic> raw;

  const ZoneModel({
    required this.id,
    required this.name,
    required this.tectonicType,
    required this.faultName,
    required this.faultType,
    required this.region,
    required this.expectedMw,
    required this.populationRisk,
    required this.riskLevel,
    required this.riskScore,
    required this.couplingRatio,
    required this.slipDeficitM,
    required this.lastMajorYear,
    required this.lastMajorMw,
    required this.lat,
    required this.lon,
    required this.raw,
  });

  String get displayName => name.isNotEmpty ? name : id;
  bool get hasCoordinates => lat != null && lon != null;

  /// Risk seviyesi metni (KRITIK / YUKSEK / ORTA / DIKKAT / DUSUK / -)
  String get riskLevelDisplay {
    if (riskLevel.isNotEmpty) return riskLevel;
    final s = riskScore ?? _derivedRiskScore();
    if (s >= 0.75) return 'KRITIK';
    if (s >= 0.50) return 'YUKSEK';
    if (s >= 0.30) return 'ORTA';
    if (s >= 0.15) return 'DIKKAT';
    return 'DUSUK';
  }

  /// Backend segment_risk_score vermediyse (yeni subduction zone'lari),
  /// coupling ve slip deficit'ten yaklasik bir risk skoru turet.
  double _derivedRiskScore() {
    final c = couplingRatio ?? 0.5;
    final s = slipDeficitM ?? 1.0;
    // Coupling agirlikli, slip deficit destekli
    final derived = (c * 0.5) + ((s / 10.0).clamp(0.0, 0.5));
    return derived.clamp(0.0, 1.0);
  }

  /// Risk seviyesine gÃ¶re kart rengi
  Color get riskColor {
    switch (riskLevelDisplay.toUpperCase()) {
      case 'KRITIK':
        return const Color(0xFFD32F2F); // koyu kÄ±rmÄ±zÄ±
      case 'YUKSEK':
        return const Color(0xFFF57C00); // turuncu
      case 'ORTA':
        return const Color(0xFFFBC02D); // sarÄ±
      case 'DIKKAT':
        return const Color(0xFF7CB342); // aÃ§Ä±k yeÅŸil
      case 'DUSUK':
        return const Color(0xFF388E3C); // yeÅŸil
      default:
        return const Color(0xFF757575); // gri (bilinmeyen)
    }
  }

  /// SÄ±ralama iÃ§in sayÄ±sal risk (yÃ¼ksekten dÃ¼ÅŸÃ¼ÄŸe)
  double get riskSortKey {
    if (riskScore != null) return riskScore!;
    switch (riskLevelDisplay.toUpperCase()) {
      case 'KRITIK':
        return 0.9;
      case 'YUKSEK':
        return 0.7;
      case 'ORTA':
        return 0.5;
      case 'DIKKAT':
        return 0.3;
      case 'DUSUK':
        return 0.1;
      default:
        return 0.0;
    }
  }

  factory ZoneModel.fromJson(Map<String, dynamic> json) {
    final id = _pickString(json, ['id', 'zone_id', 'key', 'slug', 'code']);
    final parsedName = _pickString(json, [
      'name',
      'label',
      'display_name',
      'title',
    ]);
    final name = parsedName.isNotEmpty ? parsedName : id;

    return ZoneModel(
      id: id,
      name: name,
      tectonicType: _pickString(json, [
        'tectonic_type',
        'tectonic',
        'type',
      ], fallback: ''),
      faultName: _pickString(json, ['fault_name'], fallback: ''),
      faultType: _pickString(json, ['fault_type'], fallback: ''),
      region: _pickString(json, [
        'region',
        'country',
        'area',
        'location',
      ], fallback: ''),
      expectedMw: _pickString(json, ['expected_mw'], fallback: ''),
      populationRisk: _pickString(json, ['population_risk'], fallback: ''),
      riskLevel: _pickString(json, ['segment_risk_level'], fallback: ''),
      riskScore: _pickDouble(json, ['segment_risk_score']),
      couplingRatio: _pickDouble(json, ['coupling_ratio', 'coupling']),
      slipDeficitM: _pickDouble(json, ['slip_deficit_m']),
      lastMajorYear: _pickInt(json, ['last_major_year']),
      lastMajorMw: _pickDouble(json, ['last_major_mw']),
      lat: _pickDouble(json, ['lat', 'latitude', 'center_lat', 'eff_lat']),
      lon: _pickDouble(json, [
        'lon',
        'lng',
        'longitude',
        'center_lon',
        'eff_lon',
      ]),
      raw: json,
    );
  }

  static String _pickString(
    Map<String, dynamic> json,
    List<String> keys, {
    String fallback = '',
  }) {
    for (final key in keys) {
      final value = json[key];
      if (value == null) continue;
      final text = value.toString().trim();
      if (text.isNotEmpty && text.toLowerCase() != 'null') return text;
    }
    return fallback;
  }

  static double? _pickDouble(Map<String, dynamic> json, List<String> keys) {
    for (final key in keys) {
      final value = json[key];
      if (value == null) continue;
      if (value is num) return value.toDouble();
      final parsed = double.tryParse(value.toString());
      if (parsed != null) return parsed;
    }
    return null;
  }

  static int? _pickInt(Map<String, dynamic> json, List<String> keys) {
    for (final key in keys) {
      final value = json[key];
      if (value == null) continue;
      if (value is int) return value;
      if (value is num) return value.toInt();
      final parsed = int.tryParse(value.toString());
      if (parsed != null) return parsed;
    }
    return null;
  }
}

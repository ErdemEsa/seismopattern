class ZoneModel {
  final String id;
  final String name;
  final String tectonicType;
  final String region;
  final double? lat;
  final double? lon;
  final Map<String, dynamic> raw;

  const ZoneModel({
    required this.id,
    required this.name,
    required this.tectonicType,
    required this.region,
    required this.lat,
    required this.lon,
    required this.raw,
  });

  String get displayName => name.isNotEmpty ? name : id;
  bool get hasCoordinates => lat != null && lon != null;

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
        'fault_type',
      ], fallback: 'Unknown'),
      region: _pickString(json, [
        'region',
        'country',
        'area',
        'location',
      ], fallback: ''),
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
      if (text.isNotEmpty) return text;
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
}

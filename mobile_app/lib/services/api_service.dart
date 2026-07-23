import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config.dart';
import '../models/zone_model.dart';

class ApiService {
  Future<Map<String, dynamic>> fetchStatus() async {
    final uri = Uri.parse('${AppConfig.baseUrl}${AppConfig.statusPath}');
    final response = await http.get(uri).timeout(const Duration(seconds: 20));

    if (response.statusCode != 200) {
      throw Exception(
        'Status request failed (${response.statusCode}): ${response.body}',
      );
    }

    final decoded = jsonDecode(response.body);
    if (decoded is Map<String, dynamic>) {
      return decoded;
    }

    return {'data': decoded};
  }

  Future<List<ZoneModel>> fetchZones() async {
    final uri = Uri.parse('${AppConfig.baseUrl}${AppConfig.zonesPath}');
    final response = await http.get(uri).timeout(const Duration(seconds: 20));

    if (response.statusCode != 200) {
      throw Exception(
        'Zones request failed (${response.statusCode}): ${response.body}',
      );
    }

    final decoded = jsonDecode(response.body);
    final List<ZoneModel> zones = [];

    if (decoded is List) {
      for (final item in decoded) {
        if (item is Map) {
          zones.add(ZoneModel.fromJson(Map<String, dynamic>.from(item)));
        }
      }
    } else if (decoded is Map<String, dynamic>) {
      final nested = decoded['zones'] ?? decoded['data'] ?? decoded['items'];

      if (nested is List) {
        for (final item in nested) {
          if (item is Map) {
            zones.add(ZoneModel.fromJson(Map<String, dynamic>.from(item)));
          }
        }
      } else if (nested is Map) {
        _mapToZones(Map<String, dynamic>.from(nested), zones);
      } else {
        _mapToZones(decoded, zones);
      }
    }

    zones.sort(
      (a, b) =>
          a.displayName.toLowerCase().compareTo(b.displayName.toLowerCase()),
    );

    return zones;
  }

  void _mapToZones(Map<String, dynamic> map, List<ZoneModel> zones) {
    map.forEach((key, value) {
      if (value is Map) {
        final data = Map<String, dynamic>.from(value);
        data.putIfAbsent('id', () => key);
        zones.add(ZoneModel.fromJson(data));
      }
    });
  }

  Future<Map<String, dynamic>> fetchUncertainty({
    required double lat,
    required double lon,
  }) async {
    final uri = Uri.parse(
      '${AppConfig.baseUrl}${AppConfig.uncertaintyPath}',
    ).replace(queryParameters: {'lat': lat.toString(), 'lon': lon.toString()});

    final response = await http.get(uri).timeout(const Duration(seconds: 25));

    if (response.statusCode != 200) {
      throw Exception(
        'Uncertainty request failed (${response.statusCode}): ${response.body}',
      );
    }

    final decoded = jsonDecode(response.body);
    if (decoded is Map<String, dynamic>) {
      return decoded;
    }

    return {'data': decoded};
  }
}

import 'dart:convert';

import 'package:flutter/material.dart';

import '../models/zone_model.dart';
import '../services/api_service.dart';

class ZoneDetailScreen extends StatefulWidget {
  final ZoneModel zone;

  const ZoneDetailScreen({super.key, required this.zone});

  @override
  State<ZoneDetailScreen> createState() => _ZoneDetailScreenState();
}

class _ZoneDetailScreenState extends State<ZoneDetailScreen> {
  final ApiService _apiService = ApiService();
  late Future<Map<String, dynamic>> _future;

  @override
  void initState() {
    super.initState();
    _future = _loadData();
  }

  Future<Map<String, dynamic>> _loadData() async {
    if (!widget.zone.hasCoordinates) {
      return {'error': 'Bu zone için koordinat bulunamadı.'};
    }

    return _apiService.fetchUncertainty(
      lat: widget.zone.lat!,
      lon: widget.zone.lon!,
    );
  }

  String _fmt(dynamic value) {
    if (value == null) return '-';
    if (value is int) return value.toString();
    if (value is double) {
      if (value == value.roundToDouble() && value.abs() < 1e9) {
        return value.toInt().toString();
      }
      return value.toStringAsFixed(4);
    }
    return value.toString();
  }

  Widget _metricTile(String label, dynamic value) {
    return Card(
      child: ListTile(
        title: Text(label),
        subtitle: Text(
          _fmt(value),
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final zone = widget.zone;

    return Scaffold(
      appBar: AppBar(title: Text(zone.displayName)),
      body: FutureBuilder<Map<String, dynamic>>(
        future: _future,
        builder: (context, snapshot) {
          final isLoading = snapshot.connectionState == ConnectionState.waiting;

          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        zone.displayName,
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text('Zone ID: ${zone.id}'),
                      Text('Tektonik: ${zone.tectonicType}'),
                      if (zone.region.isNotEmpty) Text('Bölge: ${zone.region}'),
                      Text('Lat: ${zone.lat ?? '-'}'),
                      Text('Lon: ${zone.lon ?? '-'}'),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 12),
              if (isLoading)
                const Center(
                  child: Padding(
                    padding: EdgeInsets.all(24),
                    child: CircularProgressIndicator(),
                  ),
                )
              else if (snapshot.hasError)
                Card(
                  color: Colors.red.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Text('Hata: ${snapshot.error}'),
                  ),
                )
              else if (snapshot.hasData) ...[
                _metricTile('Mean', snapshot.data!['mean']),
                _metricTile('Std', snapshot.data!['std']),
                _metricTile('CI Lower', snapshot.data!['ci_lower']),
                _metricTile('CI Upper', snapshot.data!['ci_upper']),
                _metricTile('Model Count', snapshot.data!['n_models']),
                _metricTile('Pattern Type', snapshot.data!['pattern_type']),
                _metricTile('Method', snapshot.data!['method']),
                const SizedBox(height: 12),
                const Text(
                  'Raw JSON',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: SelectableText(
                      const JsonEncoder.withIndent('  ').convert(snapshot.data),
                      style: const TextStyle(fontFamily: 'monospace'),
                    ),
                  ),
                ),
              ],
            ],
          );
        },
      ),
    );
  }
}

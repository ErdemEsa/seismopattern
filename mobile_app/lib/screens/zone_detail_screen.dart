import '../widgets/pdf_download_button.dart';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/zone_model.dart';
import '../providers/app_provider.dart';
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
        dense: true,
        title: Text(label),
        subtitle: Text(
          _fmt(value),
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
    );
  }

  Widget _riskSummary(ZoneModel zone) {
    final color = zone.riskColor;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: color),
                  ),
                  child: Text(
                    zone.riskLevelDisplay,
                    style: TextStyle(color: color, fontWeight: FontWeight.bold),
                  ),
                ),
                const SizedBox(width: 12),
                if (zone.riskScore != null)
                  Text(
                    'Segment Risk: ${zone.riskScore!.toStringAsFixed(2)}',
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
              ],
            ),
            const SizedBox(height: 10),
            if (zone.faultName.isNotEmpty) _kv('Fay', zone.faultName),
            if (zone.faultType.isNotEmpty) _kv('Fay tipi', zone.faultType),
            if (zone.tectonicType.isNotEmpty)
              _kv('Tektonik', zone.tectonicType),
            if (zone.region.isNotEmpty) _kv('Bölge', zone.region),
            if (zone.expectedMw.isNotEmpty) _kv('Beklenen Mw', zone.expectedMw),
            if (zone.populationRisk.isNotEmpty)
              _kv('Nüfus riski', zone.populationRisk),
            if (zone.couplingRatio != null)
              _kv('Coupling ratio', zone.couplingRatio!.toStringAsFixed(2)),
            if (zone.slipDeficitM != null)
              _kv('Slip deficit', '${zone.slipDeficitM!.toStringAsFixed(2)} m'),
            if (zone.lastMajorYear != null || zone.lastMajorMw != null)
              _kv(
                'Son büyük deprem',
                [
                  if (zone.lastMajorYear != null) '${zone.lastMajorYear}',
                  if (zone.lastMajorMw != null)
                    'Mw ${zone.lastMajorMw!.toStringAsFixed(1)}',
                ].join(' • '),
              ),
            _kv(
              'Konum',
              '${zone.lat?.toStringAsFixed(3) ?? '-'}, ${zone.lon?.toStringAsFixed(3) ?? '-'}',
            ),

            if (zone.lat != null && zone.lon != null) ...[
              const SizedBox(height: 16),
              Center(
                child: PdfDownloadButton(
                  key: ValueKey(
                    'pdf-zone-${zone.name}-${zone.lat!.toStringAsFixed(4)}-${zone.lon!.toStringAsFixed(4)}',
                  ),
                  lat: zone.lat!,
                  lon: zone.lon!,
                  autoStart: true,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _kv(String k, String v) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 140,
            child: Text(k, style: const TextStyle(color: Colors.black54)),
          ),
          Expanded(
            child: Text(v, style: const TextStyle(fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final zone = widget.zone;

    return Scaffold(
      appBar: AppBar(
        title: Text(zone.displayName),
        backgroundColor: zone.riskColor.withValues(alpha: 0.15),
        actions: [
          if (zone.hasCoordinates)
            IconButton(
              icon: const Icon(Icons.map),
              tooltip: 'Haritada goster',
              onPressed: () {
                context.read<AppProvider>().focusOnMap(zone);
                Navigator.of(context).popUntil((r) => r.isFirst);
              },
            ),
        ],
      ),
      body: FutureBuilder<Map<String, dynamic>>(
        future: _future,
        builder: (context, snapshot) {
          final isLoading = snapshot.connectionState == ConnectionState.waiting;

          return ListView(
            padding: const EdgeInsets.all(12),
            children: [
              _riskSummary(zone),
              const SizedBox(height: 8),
              const Padding(
                padding: EdgeInsets.only(left: 4, bottom: 6, top: 6),
                child: Text(
                  'Kısa vadeli belirsizlik (bootstrap)',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ),
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
                if (snapshot.data!['error'] != null)
                  Card(
                    color: Colors.orange.shade50,
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: Text(snapshot.data!['error'].toString()),
                    ),
                  )
                else ...[
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
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 6),
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(10),
                      child: SelectableText(
                        const JsonEncoder.withIndent(
                          '  ',
                        ).convert(snapshot.data),
                        style: const TextStyle(fontFamily: 'monospace'),
                      ),
                    ),
                  ),
                ],
              ],
              const SizedBox(height: 16),
              const Card(
                color: Color(0xFFFFF3E0),
                child: Padding(
                  padding: EdgeInsets.all(12),
                  child: Text(
                    'Gösterilen skorlar araştırma amaçlı olasılıksal risk göstergeleridir. '
                    'Deterministik deprem tahmini veya erken uyarı sistemi değildir.',
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

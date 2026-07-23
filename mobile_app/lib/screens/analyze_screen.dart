import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../services/api_service.dart';

class AnalyzeScreen extends StatefulWidget {
  const AnalyzeScreen({super.key});

  @override
  State<AnalyzeScreen> createState() => _AnalyzeScreenState();
}

class _AnalyzeScreenState extends State<AnalyzeScreen> {
  final _formKey = GlobalKey<FormState>();
  final _latController = TextEditingController(text: '40.77');
  final _lonController = TextEditingController(text: '29.00');
  final _apiService = ApiService();

  bool _loading = false;
  String? _error;
  Map<String, dynamic>? _result;
  double? _lastLat;
  double? _lastLon;

  @override
  void dispose() {
    _latController.dispose();
    _lonController.dispose();
    super.dispose();
  }

  Future<void> _analyze() async {
    if (!_formKey.currentState!.validate()) return;

    final lat = double.parse(_latController.text.trim());
    final lon = double.parse(_lonController.text.trim());

    setState(() {
      _loading = true;
      _error = null;
      _result = null;
      _lastLat = lat;
      _lastLon = lon;
    });

    try {
      final data = await _apiService.fetchUncertainty(lat: lat, lon: lon);
      setState(() {
        _result = data;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  void _usePreset(String label, double lat, double lon) {
    _latController.text = lat.toString();
    _lonController.text = lon.toString();
    setState(() {});
  }

  String? _validateLat(String? v) {
    if (v == null || v.trim().isEmpty) return 'Enlem gerekli';
    final d = double.tryParse(v.trim());
    if (d == null) return 'Sayı olmalı';
    if (d < -90 || d > 90) return '-90 ile 90 arası';
    return null;
  }

  String? _validateLon(String? v) {
    if (v == null || v.trim().isEmpty) return 'Boylam gerekli';
    final d = double.tryParse(v.trim());
    if (d == null) return 'Sayı olmalı';
    if (d < -180 || d > 180) return '-180 ile 180 arası';
    return null;
  }

  Color _confidenceColor(double mean) {
    if (mean >= 0.75) return const Color(0xFFD32F2F);
    if (mean >= 0.50) return const Color(0xFFF57C00);
    if (mean >= 0.30) return const Color(0xFFFBC02D);
    if (mean >= 0.15) return const Color(0xFF7CB342);
    return const Color(0xFF388E3C);
  }

  String _confidenceLabel(double mean) {
    if (mean >= 0.75) return 'KRITIK';
    if (mean >= 0.50) return 'YUKSEK';
    if (mean >= 0.30) return 'ORTA';
    if (mean >= 0.15) return 'DIKKAT';
    return 'DUSUK';
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Manuel Koordinat Analizi',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 4),
                const Text(
                  'İstediğiniz herhangi bir noktada kısa vadeli belirsizlik skoru hesaplayın.',
                  style: TextStyle(color: Colors.black54),
                ),
                const SizedBox(height: 14),
                Form(
                  key: _formKey,
                  child: Row(
                    children: [
                      Expanded(
                        child: TextFormField(
                          controller: _latController,
                          decoration: const InputDecoration(
                            border: OutlineInputBorder(),
                            labelText: 'Enlem (Lat)',
                            hintText: '40.77',
                          ),
                          keyboardType: const TextInputType.numberWithOptions(
                            decimal: true,
                            signed: true,
                          ),
                          inputFormatters: [
                            FilteringTextInputFormatter.allow(
                              RegExp(r'[0-9\.\-]'),
                            ),
                          ],
                          validator: _validateLat,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: TextFormField(
                          controller: _lonController,
                          decoration: const InputDecoration(
                            border: OutlineInputBorder(),
                            labelText: 'Boylam (Lon)',
                            hintText: '29.00',
                          ),
                          keyboardType: const TextInputType.numberWithOptions(
                            decimal: true,
                            signed: true,
                          ),
                          inputFormatters: [
                            FilteringTextInputFormatter.allow(
                              RegExp(r'[0-9\.\-]'),
                            ),
                          ],
                          validator: _validateLon,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: _loading ? null : _analyze,
                    icon: _loading
                        ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.analytics),
                    label: Text(_loading ? 'Hesaplanıyor...' : 'Analiz Et'),
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Hızlı örnekler',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: [
                    _preset('İstanbul', 41.01, 28.98),
                    _preset('Tokyo', 35.68, 139.69),
                    _preset('San Francisco', 37.77, -122.42),
                    _preset('Los Angeles', 34.05, -118.24),
                    _preset('Lima', -12.05, -77.04),
                    _preset('Katmandu', 27.72, 85.32),
                    _preset('Santiago', -33.45, -70.66),
                    _preset('Anchorage', 61.22, -149.90),
                  ],
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),
        if (_error != null)
          Card(
            color: Colors.red.shade50,
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                children: [
                  const Icon(Icons.error, color: Colors.red),
                  const SizedBox(width: 8),
                  Expanded(child: Text(_error!)),
                ],
              ),
            ),
          ),
        if (_result != null) _buildResult(_result!),
        const SizedBox(height: 16),
        const Card(
          color: Color(0xFFFFF3E0),
          child: Padding(
            padding: EdgeInsets.all(12),
            child: Text(
              'Skorlar araştırma amaçlı olasılıksal göstergelerdir. '
              'Deterministik deprem tahmini değildir.',
            ),
          ),
        ),
      ],
    );
  }

  Widget _preset(String label, double lat, double lon) {
    return ActionChip(
      label: Text(label),
      onPressed: () => _usePreset(label, lat, lon),
    );
  }

  Widget _buildResult(Map<String, dynamic> data) {
    if (data['error'] != null) {
      return Card(
        color: Colors.orange.shade50,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Text(data['error'].toString()),
        ),
      );
    }

    final mean = (data['mean'] as num?)?.toDouble() ?? 0.0;
    final std = (data['std'] as num?)?.toDouble() ?? 0.0;
    final ciLower = (data['ci_lower'] as num?)?.toDouble() ?? 0.0;
    final ciUpper = (data['ci_upper'] as num?)?.toDouble() ?? 0.0;
    final nModels = data['n_models'];
    final patternType = data['pattern_type']?.toString() ?? '-';
    final nEvents = data['n_events'];

    final color = _confidenceColor(mean);
    final label = _confidenceLabel(mean);

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
                    color: color.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: color),
                  ),
                  child: Text(
                    label,
                    style: TextStyle(
                      color: color,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Text(
                  '${_lastLat?.toStringAsFixed(3)}, ${_lastLon?.toStringAsFixed(3)}',
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ],
            ),
            const SizedBox(height: 14),
            const Text('Mean (ortalama skor)'),
            const SizedBox(height: 4),
            _bar(mean, color),
            const SizedBox(height: 4),
            Text(
              mean.toStringAsFixed(4),
              style: TextStyle(
                color: color,
                fontWeight: FontWeight.bold,
                fontSize: 20,
              ),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(child: _stat('Std', std.toStringAsFixed(4))),
                Expanded(
                  child: _stat(
                    'CI [%95]',
                    '${ciLower.toStringAsFixed(3)} - ${ciUpper.toStringAsFixed(3)}',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(child: _stat('Model sayısı', nModels?.toString() ?? '-')),
                Expanded(child: _stat('Pattern', patternType)),
                Expanded(child: _stat('Olay sayısı', nEvents?.toString() ?? '-')),
              ],
            ),
            const SizedBox(height: 12),
            ExpansionTile(
              title: const Text('Raw JSON'),
              tilePadding: EdgeInsets.zero,
              childrenPadding: EdgeInsets.zero,
              children: [
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(10),
                  color: Colors.grey.shade100,
                  child: SelectableText(
                    const JsonEncoder.withIndent('  ').convert(data),
                    style: const TextStyle(fontFamily: 'monospace'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _bar(double value, Color color) {
    final clamped = value.clamp(0.0, 1.0);
    return ClipRRect(
      borderRadius: BorderRadius.circular(6),
      child: LinearProgressIndicator(
        value: clamped,
        minHeight: 12,
        backgroundColor: color.withOpacity(0.15),
        valueColor: AlwaysStoppedAnimation<Color>(color),
      ),
    );
  }

  Widget _stat(String k, String v) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(k, style: const TextStyle(color: Colors.black54, fontSize: 12)),
        Text(v, style: const TextStyle(fontWeight: FontWeight.bold)),
      ],
    );
  }
}

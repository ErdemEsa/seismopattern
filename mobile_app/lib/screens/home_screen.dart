import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../config.dart';
import '../models/zone_model.dart';
import '../providers/app_provider.dart';
import 'zone_detail_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<AppProvider>(
      builder: (context, app, _) {
        final loading = (app.isLoadingStatus && app.status == null) ||
            (app.isLoadingZones && app.zones.isEmpty);

        if (loading) {
          return Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: const [
                CircularProgressIndicator(),
                SizedBox(height: 12),
                Text('Backend uyanıyor...'),
                SizedBox(height: 4),
                Text(
                  'İlk yükleme 30 saniye kadar sürebilir',
                  style: TextStyle(color: Colors.black54, fontSize: 12),
                ),
              ],
            ),
          );
        }

        return RefreshIndicator(
          onRefresh: () async {
            await Future.wait([app.loadStatus(), app.loadZones()]);
          },
          child: ListView(
            physics: const AlwaysScrollableScrollPhysics(),
            padding: const EdgeInsets.all(12),
            children: [
              _HeroBanner(status: app.status),
              const SizedBox(height: 12),
              _StatusBadge(
                status: app.status,
                hasError: app.statusError != null,
                errorMessage: app.statusError,
                onRetry: app.loadStatus,
              ),
              const SizedBox(height: 12),
              _StatCards(status: app.status, zones: app.zones),
              const SizedBox(height: 12),
              _CriticalZones(zones: app.zones),
              const SizedBox(height: 12),
              const _DisclaimerCard(),
              const SizedBox(height: 12),
              if (app.status != null) _RawJsonExpander(status: app.status!),
            ],
          ),
        );
      },
    );
  }
}

class _HeroBanner extends StatelessWidget {
  final Map<String, dynamic>? status;
  const _HeroBanner({required this.status});

  @override
  Widget build(BuildContext context) {
    final version = status?['version']?.toString() ?? '4.0';
    return Card(
      color: Colors.deepOrange.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.deepOrange,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Icon(
                    Icons.public,
                    color: Colors.white,
                    size: 24,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'SeismoPattern',
                        style: TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        'v$version - Kalibre olasılıksal risk izleme',
                        style: const TextStyle(
                          color: Colors.black54,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _StatusBadge extends StatelessWidget {
  final Map<String, dynamic>? status;
  final bool hasError;
  final String? errorMessage;
  final Future<void> Function() onRetry;

  const _StatusBadge({
    required this.status,
    required this.hasError,
    required this.errorMessage,
    required this.onRetry,
  });

  @override
  Widget build(BuildContext context) {
    final ok = status != null && !hasError;
    final color = ok ? Colors.green : Colors.red;
    final label = ok ? 'Backend online' : 'Backend offline';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(
                color: color,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: color.withValues(alpha: 0.5),
                    blurRadius: 6,
                  ),
                ],
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    label,
                    style: TextStyle(
                      color: color,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    AppConfig.baseUrl,
                    style: const TextStyle(
                      fontSize: 11,
                      color: Colors.black54,
                    ),
                  ),
                  if (hasError && errorMessage != null)
                    Text(
                      errorMessage!,
                      style: const TextStyle(fontSize: 11, color: Colors.red),
                    ),
                ],
              ),
            ),
            if (hasError)
              IconButton(
                icon: const Icon(Icons.refresh),
                onPressed: () => onRetry(),
              ),
          ],
        ),
      ),
    );
  }
}

class _StatCards extends StatelessWidget {
  final Map<String, dynamic>? status;
  final List<ZoneModel> zones;

  const _StatCards({required this.status, required this.zones});

  @override
  Widget build(BuildContext context) {
    final auc = status?['auc']?.toString() ?? '-';
    final zoneCount = zones.length;
    final critical = zones.where((z) => z.riskLevelDisplay == 'KRITIK').length;

    return Row(
      children: [
        Expanded(
          child: _statCard(
            icon: Icons.public,
            label: 'İzlenen Zone',
            value: zoneCount.toString(),
            color: Colors.blue,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _statCard(
            icon: Icons.warning_amber,
            label: 'KRITIK',
            value: critical.toString(),
            color: Colors.red,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _statCard(
            icon: Icons.analytics,
            label: 'Model AUC',
            value: auc,
            color: Colors.deepOrange,
          ),
        ),
      ],
    );
  }

  Widget _statCard({
    required IconData icon,
    required String label,
    required String value,
    required Color color,
  }) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: color, size: 22),
            const SizedBox(height: 6),
            Text(
              value,
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            Text(
              label,
              style: const TextStyle(fontSize: 11, color: Colors.black54),
            ),
          ],
        ),
      ),
    );
  }
}

class _CriticalZones extends StatelessWidget {
  final List<ZoneModel> zones;
  const _CriticalZones({required this.zones});

  @override
  Widget build(BuildContext context) {
    final sorted = List<ZoneModel>.from(zones)
      ..sort((a, b) => b.riskSortKey.compareTo(a.riskSortKey));
    final top = sorted.take(5).toList();

    if (top.isEmpty) return const SizedBox.shrink();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: const [
                Icon(Icons.local_fire_department, color: Colors.red),
                SizedBox(width: 6),
                Text(
                  'En yüksek riskli 5 bölge',
                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 6),
            for (final zone in top) _row(context, zone),
          ],
        ),
      ),
    );
  }

  Widget _row(BuildContext context, ZoneModel zone) {
    final color = zone.riskColor;
    return InkWell(
      onTap: () {
        Navigator.of(context).push(
          MaterialPageRoute(builder: (_) => ZoneDetailScreen(zone: zone)),
        );
      },
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Row(
          children: [
            Container(
              width: 6,
              height: 32,
              decoration: BoxDecoration(
                color: color,
                borderRadius: BorderRadius.circular(3),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    zone.displayName,
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
                  if (zone.region.isNotEmpty)
                    Text(
                      zone.region,
                      style:
                          const TextStyle(fontSize: 11, color: Colors.black54),
                    ),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: color),
              ),
              child: Text(
                zone.riskLevelDisplay,
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.bold,
                  fontSize: 11,
                ),
              ),
            ),
            const Icon(Icons.chevron_right, size: 18),
          ],
        ),
      ),
    );
  }
}

class _DisclaimerCard extends StatelessWidget {
  const _DisclaimerCard();

  @override
  Widget build(BuildContext context) {
    return const Card(
      color: Color(0xFFFFF3E0),
      child: Padding(
        padding: EdgeInsets.all(12),
        child: Row(
          children: [
            Icon(Icons.info_outline, color: Colors.orange),
            SizedBox(width: 8),
            Expanded(
              child: Text(
                'Bu uygulama deterministik deprem tahmini veya erken uyarı '
                'sistemi değildir. Skorlar araştırma amaçlı olasılıksal '
                'risk göstergeleridir.',
                style: TextStyle(fontSize: 12),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _RawJsonExpander extends StatelessWidget {
  final Map<String, dynamic> status;
  const _RawJsonExpander({required this.status});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ExpansionTile(
        leading: const Icon(Icons.code),
        title: const Text('Teknik detaylar (Status JSON)'),
        children: [
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            color: Colors.grey.shade100,
            child: SelectableText(
              const JsonEncoder.withIndent('  ').convert(status),
              style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }
}

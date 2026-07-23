import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/zone_model.dart';
import '../providers/app_provider.dart';
import 'zone_detail_screen.dart';

enum ZoneSortMode { risk, name }

class ZonesScreen extends StatefulWidget {
  const ZonesScreen({super.key});

  @override
  State<ZonesScreen> createState() => _ZonesScreenState();
}

class _ZonesScreenState extends State<ZonesScreen> {
  ZoneSortMode _sortMode = ZoneSortMode.risk;
  String _query = '';

  List<ZoneModel> _applyFilter(List<ZoneModel> zones) {
    Iterable<ZoneModel> filtered = zones;

    if (_query.trim().isNotEmpty) {
      final q = _query.trim().toLowerCase();
      filtered = filtered.where((z) {
        return z.displayName.toLowerCase().contains(q) ||
            z.region.toLowerCase().contains(q) ||
            z.faultName.toLowerCase().contains(q) ||
            z.tectonicType.toLowerCase().contains(q) ||
            z.faultType.toLowerCase().contains(q);
      });
    }

    final list = filtered.toList();
    if (_sortMode == ZoneSortMode.risk) {
      list.sort((a, b) => b.riskSortKey.compareTo(a.riskSortKey));
    } else {
      list.sort(
        (a, b) =>
            a.displayName.toLowerCase().compareTo(b.displayName.toLowerCase()),
      );
    }
    return list;
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<AppProvider>(
      builder: (context, app, _) {
        if (app.isLoadingZones && app.zones.isEmpty) {
          return const Center(child: CircularProgressIndicator());
        }

        if (app.zonesError != null && app.zones.isEmpty) {
          return Center(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.error_outline, size: 48, color: Colors.red),
                  const SizedBox(height: 12),
                  Text(app.zonesError!, textAlign: TextAlign.center),
                  const SizedBox(height: 12),
                  ElevatedButton.icon(
                    onPressed: app.loadZones,
                    icon: const Icon(Icons.refresh),
                    label: const Text('Tekrar dene'),
                  ),
                ],
              ),
            ),
          );
        }

        final visible = _applyFilter(app.zones);

        return Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 12, 12, 4),
              child: TextField(
                decoration: InputDecoration(
                  prefixIcon: const Icon(Icons.search),
                  hintText: 'Zone / bölge / fay ara...',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  isDense: true,
                ),
                onChanged: (v) => setState(() => _query = v),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: Row(
                children: [
                  Text(
                    '${visible.length} zone',
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
                  const Spacer(),
                  const Text('Sırala: '),
                  DropdownButton<ZoneSortMode>(
                    value: _sortMode,
                    isDense: true,
                    onChanged: (v) {
                      if (v != null) setState(() => _sortMode = v);
                    },
                    items: const [
                      DropdownMenuItem(
                        value: ZoneSortMode.risk,
                        child: Text('Risk (yüksek → düşük)'),
                      ),
                      DropdownMenuItem(
                        value: ZoneSortMode.name,
                        child: Text('sim (A → Z)'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const Divider(height: 1),
            Expanded(
              child: RefreshIndicator(
                onRefresh: app.loadZones,
                child: ListView.separated(
                  physics: const AlwaysScrollableScrollPhysics(),
                  padding: const EdgeInsets.all(12),
                  itemCount: visible.length,
                  separatorBuilder: (context, index) =>
                      const SizedBox(height: 8),
                  itemBuilder: (context, index) {
                    final zone = visible[index];
                    return _ZoneCard(zone: zone);
                  },
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _ZoneCard extends StatelessWidget {
  final ZoneModel zone;
  const _ZoneCard({required this.zone});

  @override
  Widget build(BuildContext context) {
    final riskColor = zone.riskColor;
    final subtitleParts = <String>[
      if (zone.region.isNotEmpty) zone.region,
      if (zone.faultType.isNotEmpty)
        zone.faultType
      else if (zone.tectonicType.isNotEmpty)
        zone.tectonicType,
      if (zone.expectedMw.isNotEmpty) 'Mw ${zone.expectedMw}',
    ];

    return Card(
      clipBehavior: Clip.hardEdge,
      child: InkWell(
        onTap: () {
          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (_) => ZoneDetailScreen(zone: zone),
            ),
          );
        },
        child: IntrinsicHeight(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Container(width: 8, color: riskColor),
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: Text(
                              zone.displayName,
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                              ),
                            ),
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 4,
                            ),
                            decoration: BoxDecoration(
                              color: riskColor.withValues(alpha: 0.15),
                              borderRadius: BorderRadius.circular(6),
                              border: Border.all(color: riskColor),
                            ),
                            child: Text(
                              zone.riskLevelDisplay,
                              style: TextStyle(
                                color: riskColor,
                                fontWeight: FontWeight.bold,
                                fontSize: 12,
                              ),
                            ),
                          ),
                        ],
                      ),
                      if (subtitleParts.isNotEmpty) ...[
                        const SizedBox(height: 4),
                        Text(
                          subtitleParts.join('  •  '),
                          style: const TextStyle(color: Colors.black54),
                        ),
                      ],
                      const SizedBox(height: 6),
                      Wrap(
                        spacing: 6,
                        runSpacing: 4,
                        children: [
                          if (zone.riskScore != null)
                            _chip(
                              'Skor ${zone.riskScore!.toStringAsFixed(2)}',
                              riskColor,
                            ),
                          if (zone.couplingRatio != null)
                            _chip(
                              'Coupling ${zone.couplingRatio!.toStringAsFixed(2)}',
                              Colors.blueGrey,
                            ),
                          if (zone.slipDeficitM != null)
                            _chip(
                              'Slip ${zone.slipDeficitM!.toStringAsFixed(1)}m',
                              Colors.indigo,
                            ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
              const Padding(
                padding: EdgeInsets.only(right: 8),
                child: Center(child: Icon(Icons.chevron_right)),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _chip(String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: color,
          fontSize: 11,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

# -*- coding: utf-8 -*-
"""Regenerates map_screen with focused-zone auto-zoom support."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib"

CONTENT = r'''
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:provider/provider.dart';

import '../models/zone_model.dart';
import '../providers/app_provider.dart';
import 'zone_detail_screen.dart';

class MapScreen extends StatefulWidget {
  const MapScreen({super.key});

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  final MapController _mapController = MapController();
  ZoneModel? _selected;

  double _markerRadius(ZoneModel z) {
    final s = z.riskSortKey;
    if (s >= 0.75) return 18;
    if (s >= 0.50) return 15;
    if (s >= 0.30) return 12;
    if (s >= 0.15) return 10;
    return 8;
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<AppProvider>(
      builder: (context, app, _) {
        final zones = app.zones.where((z) => z.hasCoordinates).toList();

        // Focus istegi varsa uygula (Zone Detail -> Haritada goster)
        final focused = app.focusedZone;
        if (focused != null && focused.hasCoordinates) {
          WidgetsBinding.instance.addPostFrameCallback((_) {
            if (!mounted) return;
            setState(() => _selected = focused);
            try {
              _mapController.move(
                LatLng(focused.lat!, focused.lon!),
                5,
              );
            } catch (_) {}
            app.clearFocus();
          });
        }

        if (app.isLoadingZones && zones.isEmpty) {
          return const Center(child: CircularProgressIndicator());
        }

        if (zones.isEmpty) {
          return Center(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.map_outlined, size: 48, color: Colors.grey),
                  const SizedBox(height: 12),
                  const Text('Koordinatli zone bulunamadi.'),
                  const SizedBox(height: 12),
                  ElevatedButton.icon(
                    onPressed: app.loadZones,
                    icon: const Icon(Icons.refresh),
                    label: const Text('Yeniden yukle'),
                  ),
                ],
              ),
            ),
          );
        }

        return Stack(
          children: [
            FlutterMap(
              mapController: _mapController,
              options: const MapOptions(
                initialCenter: LatLng(20, 30),
                initialZoom: 2,
                minZoom: 2,
                maxZoom: 12,
              ),
              children: [
                TileLayer(
                  urlTemplate:
                      'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                  userAgentPackageName: 'com.seismopattern.mobile_app',
                ),
                MarkerLayer(
                  markers: zones.map((z) {
                    final radius = _markerRadius(z);
                    final isSelected =
                        _selected != null && _selected!.id == z.id;
                    return Marker(
                      point: LatLng(z.lat!, z.lon!),
                      width: radius * 2 + 8,
                      height: radius * 2 + 8,
                      child: GestureDetector(
                        onTap: () => setState(() => _selected = z),
                        child: Container(
                          decoration: BoxDecoration(
                            color: z.riskColor.withValues(alpha: 0.75),
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: isSelected
                                  ? Colors.blueAccent
                                  : Colors.white,
                              width: isSelected ? 3 : 2,
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withValues(alpha: 0.25),
                                blurRadius: 4,
                                offset: const Offset(0, 2),
                              ),
                            ],
                          ),
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ],
            ),
            const Positioned(
              left: 12,
              top: 12,
              child: _Legend(),
            ),
            if (_selected != null)
              Positioned(
                left: 12,
                right: 12,
                bottom: 12,
                child: _SelectedCard(
                  zone: _selected!,
                  onClose: () => setState(() => _selected = null),
                  onOpen: () {
                    final z = _selected!;
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => ZoneDetailScreen(zone: z),
                      ),
                    );
                  },
                ),
              ),
          ],
        );
      },
    );
  }
}

class _Legend extends StatelessWidget {
  const _Legend();

  static const _levels = [
    ('KRITIK', Color(0xFFD32F2F)),
    ('YUKSEK', Color(0xFFF57C00)),
    ('ORTA', Color(0xFFFBC02D)),
    ('DIKKAT', Color(0xFF7CB342)),
    ('DUSUK', Color(0xFF388E3C)),
  ];

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.white.withValues(alpha: 0.92),
      elevation: 3,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Risk seviyesi',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12),
            ),
            const SizedBox(height: 6),
            for (final level in _levels)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 2),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 12,
                      height: 12,
                      decoration: BoxDecoration(
                        color: level.$2,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 6),
                    Text(level.$1, style: const TextStyle(fontSize: 11)),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _SelectedCard extends StatelessWidget {
  final ZoneModel zone;
  final VoidCallback onClose;
  final VoidCallback onOpen;

  const _SelectedCard({
    required this.zone,
    required this.onClose,
    required this.onOpen,
  });

  @override
  Widget build(BuildContext context) {
    final color = zone.riskColor;
    return Card(
      elevation: 6,
      clipBehavior: Clip.hardEdge,
      child: InkWell(
        onTap: onOpen,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              Container(width: 6, height: 48, color: color),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      zone.displayName,
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      [
                        if (zone.region.isNotEmpty) zone.region,
                        if (zone.faultType.isNotEmpty)
                          zone.faultType
                        else if (zone.tectonicType.isNotEmpty)
                          zone.tectonicType,
                        zone.riskLevelDisplay,
                      ].join('  -  '),
                      style: TextStyle(
                        color: color,
                        fontWeight: FontWeight.w600,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
              IconButton(icon: const Icon(Icons.close), onPressed: onClose),
              const Icon(Icons.chevron_right),
            ],
          ),
        ),
      ),
    );
  }
}
'''

path = LIB / "screens" / "map_screen.dart"
path.write_text(CONTENT.lstrip("\n"), encoding="utf-8", newline="\n")
print(f"[OK] wrote {path}")

# -*- coding: utf-8 -*-
"""Regenerates Flutter mobile screens with proper UTF-8 encoding."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib"

FILES = {}

# ============================================================
# lib/main.dart
# ============================================================
FILES["main.dart"] = r'''
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'providers/app_provider.dart';
import 'screens/about_screen.dart';
import 'screens/home_screen.dart';
import 'screens/map_screen.dart';
import 'screens/zones_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  runApp(
    ChangeNotifierProvider(
      create: (_) => AppProvider()..loadInitialData(),
      child: const SeismoPatternApp(),
    ),
  );
}

class SeismoPatternApp extends StatelessWidget {
  const SeismoPatternApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SeismoPattern',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepOrange),
        useMaterial3: true,
      ),
      home: const RootShell(),
    );
  }
}

class RootShell extends StatefulWidget {
  const RootShell({super.key});

  @override
  State<RootShell> createState() => _RootShellState();
}

class _RootShellState extends State<RootShell> {
  int _index = 0;

  final _screens = const [
    HomeScreen(),
    ZonesScreen(),
    MapScreen(),
    AboutScreen(),
  ];

  final _titles = const [
    'SeismoPattern',
    'Zones',
    'Harita',
    'Hakkında',
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(_titles[_index])),
      body: IndexedStack(index: _index, children: _screens),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (value) {
          setState(() => _index = value);
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.home_outlined),
            selectedIcon: Icon(Icons.home),
            label: 'Ana Sayfa',
          ),
          NavigationDestination(
            icon: Icon(Icons.public_outlined),
            selectedIcon: Icon(Icons.public),
            label: 'Zones',
          ),
          NavigationDestination(
            icon: Icon(Icons.map_outlined),
            selectedIcon: Icon(Icons.map),
            label: 'Harita',
          ),
          NavigationDestination(
            icon: Icon(Icons.info_outline),
            selectedIcon: Icon(Icons.info),
            label: 'Hakkında',
          ),
        ],
      ),
    );
  }
}
'''

# ============================================================
# lib/screens/map_screen.dart
# ============================================================
FILES["screens/map_screen.dart"] = r'''
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
                  const Text('Koordinatlı zone bulunamadı.'),
                  const SizedBox(height: 12),
                  ElevatedButton.icon(
                    onPressed: app.loadZones,
                    icon: const Icon(Icons.refresh),
                    label: const Text('Yeniden yükle'),
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
                    return Marker(
                      point: LatLng(z.lat!, z.lon!),
                      width: radius * 2 + 4,
                      height: radius * 2 + 4,
                      child: GestureDetector(
                        onTap: () => setState(() => _selected = z),
                        child: Container(
                          decoration: BoxDecoration(
                            color: z.riskColor.withOpacity(0.75),
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: Colors.white,
                              width: 2,
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.25),
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
            Positioned(
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
      color: Colors.white.withOpacity(0.92),
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
                    Text(
                      level.$1,
                      style: const TextStyle(fontSize: 11),
                    ),
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
              Container(
                width: 6,
                height: 48,
                color: color,
              ),
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
                      ].join('  •  '),
                      style: TextStyle(
                        color: color,
                        fontWeight: FontWeight.w600,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
              IconButton(
                icon: const Icon(Icons.close),
                onPressed: onClose,
              ),
              const Icon(Icons.chevron_right),
            ],
          ),
        ),
      ),
    );
  }
}
'''

# ============================================================
# lib/screens/zones_screen.dart (unchanged from previous run)
# ============================================================
FILES["screens/zones_screen.dart"] = r'''
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
                              color: riskColor.withOpacity(0.15),
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
        color: color.withOpacity(0.10),
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
'''

# ============================================================
# lib/screens/zone_detail_screen.dart (unchanged)
# ============================================================
FILES["screens/zone_detail_screen.dart"] = r'''
import 'dart:convert';

import 'package:flutter/material.dart';

import '../models/zone_model.dart';
import '../services/api_service.dart';

class ZoneDetailScreen extends StatefulWidget {
  final ZoneModel zone;

  const ZoneDetailScreen({
    super.key,
    required this.zone,
  });

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
                    color: color.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: color),
                  ),
                  child: Text(
                    zone.riskLevelDisplay,
                    style: TextStyle(
                      color: color,
                      fontWeight: FontWeight.bold,
                    ),
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
        backgroundColor: zone.riskColor.withOpacity(0.15),
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
                        const JsonEncoder.withIndent('  ')
                            .convert(snapshot.data),
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
'''

for rel, content in FILES.items():
    path = LIB / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    text = content.lstrip("\n")
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"[OK] wrote {path}")

print("Done.")

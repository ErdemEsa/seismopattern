import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/app_provider.dart';
import 'zone_detail_screen.dart';

class ZonesScreen extends StatelessWidget {
  const ZonesScreen({super.key});

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

        return RefreshIndicator(
          onRefresh: app.loadZones,
          child: ListView.separated(
            physics: const AlwaysScrollableScrollPhysics(),
            padding: const EdgeInsets.all(12),
            itemCount: app.zones.length,
            separatorBuilder: (context, index) => const SizedBox(height: 8),
            itemBuilder: (context, index) {
              final zone = app.zones[index];

              return Card(
                child: ListTile(
                  leading: const CircleAvatar(child: Icon(Icons.public)),
                  title: Text(zone.displayName),
                  subtitle: Text(
                    [
                      if (zone.id.isNotEmpty) 'ID: ${zone.id}',
                      'Tektonik: ${zone.tectonicType}',
                      if (zone.region.isNotEmpty) 'Bölge: ${zone.region}',
                      if (zone.hasCoordinates)
                        'Koordinat: ${zone.lat}, ${zone.lon}',
                    ].join('\n'),
                  ),
                  isThreeLine: true,
                  trailing: const Icon(Icons.chevron_right),
                  onTap: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => ZoneDetailScreen(zone: zone),
                      ),
                    );
                  },
                ),
              );
            },
          ),
        );
      },
    );
  }
}

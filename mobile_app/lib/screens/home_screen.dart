import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../config.dart';
import '../providers/app_provider.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<AppProvider>(
      builder: (context, app, _) {
        if (app.isLoadingStatus && app.status == null) {
          return const Center(child: CircularProgressIndicator());
        }

        return RefreshIndicator(
          onRefresh: app.loadStatus,
          child: ListView(
            physics: const AlwaysScrollableScrollPhysics(),
            padding: const EdgeInsets.all(16),
            children: [
              Card(
                child: ListTile(
                  leading: const Icon(Icons.cloud_done),
                  title: const Text('Backend bağlantısı'),
                  subtitle: Text(AppConfig.baseUrl),
                ),
              ),
              const SizedBox(height: 12),
              Card(
                child: ListTile(
                  leading: Icon(
                    app.statusError == null ? Icons.check_circle : Icons.error,
                    color: app.statusError == null ? Colors.green : Colors.red,
                  ),
                  title: Text(
                    app.statusError == null
                        ? 'API erişilebilir'
                        : 'API erişim hatası',
                  ),
                  subtitle: Text(
                    app.statusError ?? 'Status endpoint başarıyla okundu.',
                  ),
                ),
              ),
              const SizedBox(height: 12),
              if (app.status != null) ...[
                const Text(
                  'Status JSON',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: SelectableText(
                      const JsonEncoder.withIndent('  ').convert(app.status),
                      style: const TextStyle(fontFamily: 'monospace'),
                    ),
                  ),
                ),
              ],
              if (app.status == null && app.statusError != null) ...[
                ElevatedButton.icon(
                  onPressed: app.loadStatus,
                  icon: const Icon(Icons.refresh),
                  label: const Text('Tekrar dene'),
                ),
              ],
              const SizedBox(height: 16),
              const Card(
                child: Padding(
                  padding: EdgeInsets.all(12),
                  child: Text(
                    'Bu uygulama deterministik deprem tahmini veya resmi erken uyarı sistemi değildir. '
                    'Gösterilen skorlar araştırma amaçlı olasılıksal risk göstergeleridir.',
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

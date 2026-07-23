import 'package:flutter/material.dart';

import '../config.dart';
import 'settings_screen.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: ListTile(
            leading: const Icon(Icons.settings),
            title: const Text('Ayarlar'),
            subtitle: Text('Backend URL: ${AppConfig.baseUrl}'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const SettingsScreen()),
              );
            },
          ),
        ),
        const SizedBox(height: 12),
        const Card(
          child: Padding(
            padding: EdgeInsets.all(16),
            child: Text(
              'SeismoPattern v4\n\n'
              'Kalibre edilmiş, çok katmanlı, segment ölçekli, '
              'olasılıksal deprem risk izleme ve karar destek sistemi.',
              style: TextStyle(fontSize: 16),
            ),
          ),
        ),
        const SizedBox(height: 12),
        const Card(
          child: Padding(
            padding: EdgeInsets.all(16),
            child: Text(
              'Önemli Uyarı\n\n'
              'Bu uygulama deterministik deprem tahmini değildir.\n'
              'Bu uygulama resmi erken uyarı sistemi değildir.\n'
              'Gösterilen skorlar yalnızca araştırma amaçlı olasılıksal risk göstergeleridir.',
            ),
          ),
        ),
        const SizedBox(height: 12),
        const Card(
          child: Padding(
            padding: EdgeInsets.all(16),
            child: Text(
              'Backend: Flask API\n'
              'Mobil: Flutter\n'
              'Model: Two-stage XGBoost + isotonic calibration\n'
              'Bootstrap: 150 model\n'
              'Watchlist: 59 zone',
            ),
          ),
        ),
      ],
    );
  }
}

# -*- coding: utf-8 -*-
"""Regenerates Flutter mobile files with proper UTF-8 encoding."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib"

FILES = {}

# ============================================================
# lib/config.dart
# ============================================================
FILES["config.dart"] = r'''
import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Runtime editable API config. Kullanıcı ayarlar ekranından değiştirebilir.
class AppConfig {
  static const String _kStorageKey = 'api_base_url';

  /// Şu an geçerli olan backend base URL.
  static String _currentBaseUrl = _defaultBaseUrl();

  /// Uygulama açılışında SharedPreferences'ten yüklenir.
  static Future<void> load() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final saved = prefs.getString(_kStorageKey);
      if (saved != null && saved.trim().isNotEmpty) {
        _currentBaseUrl = _normalize(saved);
      }
    } catch (_) {
      // sessizce varsayılana düş
    }
  }

  /// Kullanıcı ayarlar ekranından yeni URL kaydettiğinde.
  static Future<void> setBaseUrl(String url) async {
    final normalized = _normalize(url);
    _currentBaseUrl = normalized;
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_kStorageKey, normalized);
    } catch (_) {}
  }

  /// Varsayılana dön.
  static Future<void> resetToDefault() async {
    _currentBaseUrl = _defaultBaseUrl();
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(_kStorageKey);
    } catch (_) {}
  }

  static String get baseUrl => _currentBaseUrl;
  static String get defaultBaseUrl => _defaultBaseUrl();

  static String _defaultBaseUrl() {
    // 1. --dart-define=API_BASE=... build parametresi
    const override = String.fromEnvironment('API_BASE', defaultValue: '');
    if (override.isNotEmpty) return _normalize(override);

    // 2. Platforma göre varsayılan
    if (kIsWeb) return 'http://localhost:5000';
    try {
      if (Platform.isAndroid) return 'http://10.0.2.2:5000';
    } catch (_) {}
    return 'http://localhost:5000';
  }

  static String _normalize(String raw) {
    var url = raw.trim();
    if (url.endsWith('/')) url = url.substring(0, url.length - 1);
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'http://$url';
    }
    return url;
  }

  static const String statusPath = '/api/status';
  static const String zonesPath = '/api/zones';
  static const String uncertaintyPath = '/api/uncertainty';
}
'''

# ============================================================
# lib/main.dart
# ============================================================
FILES["main.dart"] = r'''
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'config.dart';
import 'providers/app_provider.dart';
import 'screens/about_screen.dart';
import 'screens/home_screen.dart';
import 'screens/map_screen.dart';
import 'screens/zones_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await AppConfig.load();

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
# lib/screens/settings_screen.dart
# ============================================================
FILES["screens/settings_screen.dart"] = r'''
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../config.dart';
import '../providers/app_provider.dart';
import '../services/api_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _controller;
  bool _testing = false;
  String? _testResult;
  bool _testOk = false;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: AppConfig.baseUrl);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _testConnection() async {
    setState(() {
      _testing = true;
      _testResult = null;
    });

    final url = _controller.text.trim();
    if (url.isEmpty) {
      setState(() {
        _testing = false;
        _testResult = 'URL boş olamaz.';
        _testOk = false;
      });
      return;
    }

    // Geçici olarak deneme URL'sini set et
    final previous = AppConfig.baseUrl;
    await AppConfig.setBaseUrl(url);

    try {
      final api = ApiService();
      final status = await api.fetchStatus();
      setState(() {
        _testResult =
            'Bağlantı başarılı. Sürüm: ${status['version'] ?? 'unknown'}';
        _testOk = true;
        _testing = false;
      });
    } catch (e) {
      // Başarısızsa eski URL'ye dön
      await AppConfig.setBaseUrl(previous);
      _controller.text = AppConfig.baseUrl;
      setState(() {
        _testResult = 'Bağlantı başarısız: $e';
        _testOk = false;
        _testing = false;
      });
    }
  }

  Future<void> _save() async {
    final url = _controller.text.trim();
    if (url.isEmpty) return;
    await AppConfig.setBaseUrl(url);

    if (!mounted) return;
    final app = context.read<AppProvider>();
    await app.loadInitialData();

    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Backend güncellendi: ${AppConfig.baseUrl}')),
    );
    Navigator.of(context).pop();
  }

  Future<void> _reset() async {
    await AppConfig.resetToDefault();
    setState(() {
      _controller.text = AppConfig.baseUrl;
      _testResult = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Ayarlar')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text(
            'Backend URL',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _controller,
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              hintText: 'http://192.168.1.10:5000',
              helperText: 'Örnek: http://192.168.1.10:5000',
            ),
            keyboardType: TextInputType.url,
            autocorrect: false,
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _testing ? null : _testConnection,
                  icon: _testing
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.network_check),
                  label: const Text('Test Et'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _save,
                  icon: const Icon(Icons.save),
                  label: const Text('Kaydet'),
                ),
              ),
            ],
          ),
          if (_testResult != null) ...[
            const SizedBox(height: 12),
            Card(
              color: _testOk ? Colors.green.shade50 : Colors.red.shade50,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Row(
                  children: [
                    Icon(
                      _testOk ? Icons.check_circle : Icons.error,
                      color: _testOk ? Colors.green : Colors.red,
                    ),
                    const SizedBox(width: 8),
                    Expanded(child: Text(_testResult!)),
                  ],
                ),
              ),
            ),
          ],
          const SizedBox(height: 24),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Yardım',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Bilgisayarınızda çalışan Flask backend\'inin adresini girin. '
                    'Backend başlatıldığında terminaldeki "Running on http://192.168.x.x:5000" '
                    'satırındaki adresi kullanabilirsiniz. Telefonun aynı WiFi ağında '
                    'olması gerekir.',
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Varsayılan: ${AppConfig.defaultBaseUrl}',
                    style: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          TextButton.icon(
            onPressed: _reset,
            icon: const Icon(Icons.restore),
            label: const Text('Varsayılana dön'),
          ),
        ],
      ),
    );
  }
}
'''

# ============================================================
# lib/screens/about_screen.dart
# ============================================================
FILES["screens/about_screen.dart"] = r'''
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
'''

for rel, content in FILES.items():
    path = LIB / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    text = content.lstrip("\n")
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"[OK] wrote {path}")

print("Done.")

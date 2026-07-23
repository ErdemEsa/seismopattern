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

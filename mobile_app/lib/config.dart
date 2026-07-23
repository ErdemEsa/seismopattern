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

    // 2. Web'de: kendi origin'ini kullan (Flask ayni URL'de servis ediyor)
    if (kIsWeb) {
      final origin = Uri.base.origin;
      // localhost gelistirme icin fallback
      if (origin.contains('localhost') || origin.contains('127.0.0.1')) {
        return origin;
      }
      return origin;
    }

    // 3. Android emulator icin
    try {
      if (Platform.isAndroid) return 'https://seismopattern.onrender.com';
    } catch (_) {}

    return 'https://seismopattern.onrender.com';
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

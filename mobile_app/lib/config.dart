import 'package:flutter/foundation.dart';

class AppConfig {
  static String get baseUrl {
    if (kIsWeb) {
      return 'http://localhost:5000';
    }
    return 'http://10.0.2.2:5000';
  }

  static const String statusPath = '/api/status';
  static const String zonesPath = '/api/zones';
  static const String uncertaintyPath = '/api/uncertainty';
}

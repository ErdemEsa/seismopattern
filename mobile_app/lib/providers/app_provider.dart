import 'package:flutter/material.dart';

import '../models/zone_model.dart';
import '../services/api_service.dart';

class AppProvider extends ChangeNotifier {
  final ApiService _apiService = ApiService();

  Map<String, dynamic>? status;
  List<ZoneModel> zones = [];

  bool isLoadingStatus = false;
  bool isLoadingZones = false;

  String? statusError;
  String? zonesError;

  Future<void> loadInitialData() async {
    await Future.wait([loadStatus(), loadZones()]);
  }

  Future<void> loadStatus() async {
    isLoadingStatus = true;
    statusError = null;
    notifyListeners();

    try {
      status = await _apiService.fetchStatus();
    } catch (e) {
      statusError = e.toString();
    } finally {
      isLoadingStatus = false;
      notifyListeners();
    }
  }

  Future<void> loadZones() async {
    isLoadingZones = true;
    zonesError = null;
    notifyListeners();

    try {
      zones = await _apiService.fetchZones();
    } catch (e) {
      zonesError = e.toString();
    } finally {
      isLoadingZones = false;
      notifyListeners();
    }
  }
}

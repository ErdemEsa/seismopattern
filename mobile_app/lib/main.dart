import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'config.dart';
import 'providers/app_provider.dart';
import 'screens/about_screen.dart';
import 'screens/analyze_screen.dart';
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
  AppProvider? _prov;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final prov = context.read<AppProvider>();
    if (_prov != prov) {
      _prov?.removeListener(_handleTabRequest);
      _prov = prov;
      prov.addListener(_handleTabRequest);
    }
  }

  void _handleTabRequest() {
    final prov = _prov;
    if (prov == null) return;
    if (prov.desiredTabIndex != 0 && prov.desiredTabIndex != _index) {
      setState(() => _index = prov.desiredTabIndex);
      prov.consumeTabRequest();
    }
  }

  @override
  void dispose() {
    _prov?.removeListener(_handleTabRequest);
    super.dispose();
  }

  final _screens = const [
    HomeScreen(),
    ZonesScreen(),
    MapScreen(),
    AnalyzeScreen(),
    AboutScreen(),
  ];

  final _titles = const [
    'SeismoPattern',
    'Zones',
    'Harita',
    'Analiz',
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
            icon: Icon(Icons.analytics_outlined),
            selectedIcon: Icon(Icons.analytics),
            label: 'Analiz',
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

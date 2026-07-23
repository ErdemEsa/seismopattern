import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:mobile_app/main.dart';
import 'package:mobile_app/providers/app_provider.dart';

void main() {
  testWidgets('app shell renders', (WidgetTester tester) async {
    await tester.pumpWidget(
      ChangeNotifierProvider(
        create: (_) => AppProvider(),
        child: const SeismoPatternApp(),
      ),
    );

    expect(find.text('SeismoPattern'), findsOneWidget);
    expect(find.byType(NavigationBar), findsOneWidget);
  });
}

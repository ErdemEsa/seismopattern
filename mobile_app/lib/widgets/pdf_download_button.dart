import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../services/api_service.dart';

enum _PdfState { idle, starting, polling, ready, error }

class PdfDownloadButton extends StatefulWidget {
  final double lat;
  final double lon;
  final String? refDate;

  const PdfDownloadButton({
    super.key,
    required this.lat,
    required this.lon,
    this.refDate,
  });

  @override
  State<PdfDownloadButton> createState() => _PdfDownloadButtonState();
}

class _PdfDownloadButtonState extends State<PdfDownloadButton> {
  final _api = ApiService();
  _PdfState _state = _PdfState.idle;
  String? _errorMsg;
  Timer? _timer;
  int _checks = 0;

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _start() async {
    setState(() {
      _state = _PdfState.starting;
      _errorMsg = null;
      _checks = 0;
    });

    try {
      final r = await _api.startPdfGeneration(
        lat: widget.lat,
        lon: widget.lon,
        refDate: widget.refDate,
      );

      if (r['status'] == 'ready') {
        setState(() => _state = _PdfState.ready);
        await _download();
        return;
      }

      setState(() => _state = _PdfState.polling);
      _timer = Timer.periodic(const Duration(seconds: 12), (_) => _poll());
    } catch (e) {
      setState(() {
        _state = _PdfState.error;
        _errorMsg = e.toString();
      });
    }
  }

  Future<void> _poll() async {
    _checks++;
    if (_checks > 25) {
      _timer?.cancel();
      setState(() {
        _state = _PdfState.error;
        _errorMsg = 'PDF hazirlanamadi (zaman asimi)';
      });
      return;
    }

    try {
      final r = await _api.getPdfStatus(
        lat: widget.lat,
        lon: widget.lon,
        refDate: widget.refDate,
      );

      if (r['status'] == 'ready') {
        _timer?.cancel();
        setState(() => _state = _PdfState.ready);
        await _download();
      } else if (r['status'] == 'error') {
        _timer?.cancel();
        setState(() {
          _state = _PdfState.error;
          _errorMsg = r['error'] ?? 'PDF hatasi';
        });
      }
    } catch (_) {}
  }

  Future<void> _download() async {
    final url = _api.getPdfDownloadUrl(
      lat: widget.lat,
      lon: widget.lon,
      refDate: widget.refDate,
    );
    final uri = Uri.parse(url);

    try {
      final ok = await launchUrl(
        uri,
        mode: kIsWeb
            ? LaunchMode.platformDefault
            : LaunchMode.externalApplication,
        webOnlyWindowName: '_blank',
      );

      if (!ok) {
        throw Exception('Tarayici indirimi baslatamadi');
      }

      if (mounted) {
        setState(() => _state = _PdfState.idle);
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = _PdfState.error;
          _errorMsg = 'PDF acilamadi: $e';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return switch (_state) {
      _PdfState.idle => ElevatedButton.icon(
        onPressed: _start,
        icon: const Icon(Icons.picture_as_pdf),
        label: const Text('PDF Rapor Indir'),
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF388bfd),
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        ),
      ),
      _PdfState.starting || _PdfState.polling => ElevatedButton.icon(
        onPressed: null,
        icon: const SizedBox(
          width: 16,
          height: 16,
          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
        ),
        label: Text(
          _state == _PdfState.starting
              ? 'Baslatiyor...'
              : 'Hazirlaniyor... ($_checks)',
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF388bfd).withValues(alpha: 0.6),
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        ),
      ),
      _PdfState.ready => ElevatedButton.icon(
        onPressed: _download,
        icon: const Icon(Icons.download_done),
        label: const Text('Tekrar Indir'),
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF3fb950),
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        ),
      ),
      _PdfState.error => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ElevatedButton.icon(
            onPressed: _start,
            icon: const Icon(Icons.refresh),
            label: const Text('Tekrar Dene'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFf85149),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            ),
          ),
          if (_errorMsg != null)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                _errorMsg!,
                style: const TextStyle(fontSize: 11, color: Colors.red),
              ),
            ),
        ],
      ),
    };
  }
}

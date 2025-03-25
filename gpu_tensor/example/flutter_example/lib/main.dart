import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:gpu_tensor/gpu_tensor.dart'; // Your FFT extension is assumed to be available here.

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GPU Tensor FFT Demo',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const FFTDemoPage(),
    );
  }
}

class FFTDemoPage extends StatefulWidget {
  const FFTDemoPage({super.key});
  @override
  State<FFTDemoPage> createState() => _FFTDemoPageState();
}

class _FFTDemoPageState extends State<FFTDemoPage> {
  String _resultText = "";
  bool _isComputing = false;

  Future<void> _runFFT() async {
    setState(() {
      _isComputing = true;
      _resultText = "Computing FFT...";
    });

    // Create a 1D real tensor with 8 points.
    // The fft1d() method will automatically upgrade it to complex (interleaved) form.
    const int n = 8;
    final Float32List realData = Float32List(n);
    for (int i = 0; i < n; i++) {
      realData[i] = i.toDouble();
    }
    Tensor tensor = await Tensor.create([n], data: realData);

    // Compute the FFT.
    Tensor fftResult = await tensor.fft1d();

    // Retrieve FFT output data.
    Float32List resultData = await fftResult.getData();

    setState(() {
      _isComputing = false;
      _resultText = resultData.toString();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("GPU Tensor FFT Demo")),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                "Press the button to perform a 1D FFT on a sample tensor.",
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _isComputing ? null : _runFFT,
                child: const Text("Run FFT"),
              ),
              const SizedBox(height: 20),
              _isComputing
                  ? const CircularProgressIndicator()
                  : Text(_resultText, textAlign: TextAlign.center),
            ],
          ),
        ),
      ),
    );
  }
}

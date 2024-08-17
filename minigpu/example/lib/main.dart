import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:minigpu/minigpu.dart';
import 'package:minigpu/minigpu_flutter.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Minigpu Example',
      home: MinigpuExample(),
    );
  }
}

class MinigpuExample extends StatefulWidget {
  @override
  _MinigpuExampleState createState() => _MinigpuExampleState();
}

class _MinigpuExampleState extends State<MinigpuExample> {
  late Minigpu _minigpu;
  late ComputeShader _shader;
  late Buffer _buffer;
  List<double> _result = [];

  @override
  void initState() {
    super.initState();
    _initMinigpu();
  }

  Future<void> _initMinigpu() async {
    _minigpu = Minigpu();
    //await _minigpu.init();

    //_shader = _minigpu.createComputeShader();
    //_shader.loadKernelString('''
    //  kernel void myKernel(global float* buffer) {
    //    const int index = get_global_id(0);
    //    buffer[index] = index * 2.0f;
    //  }
    // ''');

    final bufferSize = 1024;
    final memSize = bufferSize * Float32List.bytesPerElement;
    //_buffer = _minigpu.createBuffer(_shader, bufferSize, memSize);

    //_shader.setBuffer('myKernel', 'buffer', _buffer);
    //_shader.dispatch('myKernel', 1024, 1, 1);

    //final outputData = Float32List(bufferSize);
    //_buffer.readSync(outputData, memSize);

    setState(() {
      //_result = outputData.map((value) => value.toDouble()).toList();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Minigpu Example'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
                onPressed: () async {
                  await _minigpu.init();
                },
                child: Text('Initialize')),
            Text(
              'Result:',
              style: TextStyle(fontSize: 24),
            ),
            SizedBox(height: 16),
            Text(
              _result.toString(),
              style: TextStyle(fontSize: 18),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _buffer.destroy();
    _shader.destroy();
    super.dispose();
  }
}

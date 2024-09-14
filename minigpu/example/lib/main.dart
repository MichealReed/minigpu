import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:minigpu/minigpu.dart';

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
  late Buffer _inputBuffer;
  late Buffer _outputBuffer;
  List<double> _result = [];

  @override
  void initState() {
    super.initState();
    _minigpu = Minigpu();
    _initMinigpu();
  }

  Future<void> _initMinigpu() async {
    await _minigpu.init();
  }

  Future<void> _runKernel() async {
    _shader = _minigpu.createComputeShader();
    _shader.loadKernelString('''(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR
                 * (x + .044715 * x * x * x))), x, x > 10.0);
    }
}
   )''');

    final bufferSize = 10000;
    final memSize = bufferSize * Float32List.bytesPerElement;
    _inputBuffer = _minigpu.createBuffer(bufferSize, memSize);
    _outputBuffer = _minigpu.createBuffer(bufferSize, memSize);

    final inputData = Float32List.fromList(
      List<double>.generate(bufferSize, (i) => i / 10.0),
    );
    _inputBuffer.setData(inputData.buffer, memSize);

    _shader.setBuffer('main', 'inp', _inputBuffer);
    _shader.setBuffer('main', 'out', _outputBuffer);
    _shader.dispatch('main', (bufferSize / 256).ceil(), 1, 1);

    final outputData = Float32List(bufferSize);
    _outputBuffer.readSync(outputData.buffer, memSize);

    setState(() {
      _result =
          outputData.sublist(0, 16).map((value) => value.toDouble()).toList();
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
            ElevatedButton(onPressed: _runKernel, child: Text('Run GELU')),
            SizedBox(height: 16),
            Text(
              'Result:',
              style: TextStyle(fontSize: 24),
            ),
            SizedBox(height: 16),
            Text(
              _result.map((value) => value.toStringAsFixed(2)).join(', '),
              style: TextStyle(fontSize: 18),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _inputBuffer.destroy();
    _outputBuffer.destroy();
    _shader.destroy();
    super.dispose();
  }
}

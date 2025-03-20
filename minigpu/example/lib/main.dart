import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:minigpu/buffer.dart';
import 'package:minigpu/compute_shader.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  runApp(MyApp());
}

int roundToBytesize(double value, {int byteSize = 4}) {
  return ((value / byteSize).round()) * byteSize;
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Interactive Minigpu Example',
      home: InteractiveShaderExample(),
    );
  }
}

class InteractiveShaderExample extends StatefulWidget {
  @override
  _InteractiveShaderExampleState createState() =>
      _InteractiveShaderExampleState();
}

class _InteractiveShaderExampleState extends State<InteractiveShaderExample> {
  late Minigpu _minigpu;
  late ComputeShader _shader;
  late Buffer _inputBuffer;
  late Buffer _outputBuffer;
  List<double> _result = [];
  final TextEditingController _shaderController = TextEditingController();

  int _bufferSize = 100; // current buffer size
  final int floatSize = 4; // 4 bytes per f32
  final Random _random = Random();
  int _previewLength = 16; // number of preview values to show

  // Loop configuration:
  bool _loopMode = false; // when checked, loop mode is available
  bool _isLoopRunning = false; // whether the loop is actively running
  double _loopDelay = 1.0; // seconds delay between shader executions

  @override
  void initState() {
    super.initState();
    _minigpu = Minigpu();
    _initMinigpu();
    // default WGSL kernel code (similar to a GELU operation)
    _shaderController.text = '''
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        //out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR * (x + .044715 * x * x * x))), x, x > 10.0);
        out[i] = inp[i] + .2;
    }
}
''';
    _initializeBuffer();
  }

  Future<void> _initMinigpu() async {
    await _minigpu.init();
    _createBuffers();
    _setBufferData();
    _shader = _minigpu.createComputeShader();
  }

  void _createBuffers() {
    int memSize = _bufferSize * floatSize;
    _inputBuffer = _minigpu.createBuffer(memSize);
    _outputBuffer = _minigpu.createBuffer(memSize);
  }

  void _setBufferData() {
    List<double> data = List.generate(_bufferSize, (i) => i / 10.0);
    final Float32List floatData = Float32List.fromList(data);
    _inputBuffer.setData(floatData, _bufferSize);
    setState(() {
      _result = data.sublist(0, _previewLength.clamp(0, data.length));
    });
  }

  void _initializeBuffer() {
    List<double> data = List.generate(_bufferSize, (i) => i / 10.0);
    final Float32List floatData = Float32List.fromList(data);
    if (_minigpu.isInitialized) {
      _inputBuffer.setData(floatData, _bufferSize);
    }
    setState(() {
      _result = data.sublist(0, _previewLength.clamp(0, data.length));
    });
  }

  void _resetBuffer() {
    List<double> data = List.filled(_bufferSize, 0.0);
    final Float32List floatData = Float32List.fromList(data);
    _inputBuffer.setData(floatData, _bufferSize);
    setState(() {
      _result = data.sublist(0, _previewLength.clamp(0, data.length));
    });
  }

  void _randomizeBuffer() {
    List<double> data =
        List.generate(_bufferSize, (i) => -5 + _random.nextDouble() * 10);
    final Float32List floatData = Float32List.fromList(data);
    _inputBuffer.setData(floatData, _bufferSize);
    setState(() {
      _result = data.sublist(0, _previewLength.clamp(0, data.length));
    });
  }

  Future<void> _runKernel() async {
    _shader.loadKernelString(_shaderController.text);
    _shader.setBuffer('inp', _inputBuffer);
    _shader.setBuffer('out', _outputBuffer);

    int workgroups = ((_bufferSize + 255) / 256).floor();
    await _shader.dispatch(workgroups, 1, 1);

    final Float32List outputData = Float32List(_bufferSize);
    await _outputBuffer.read(outputData, _bufferSize);

    setState(() {
      List<double> data = outputData.map((v) => v.toDouble()).toList();
      _result = data.sublist(0, _previewLength.clamp(0, data.length));
      // Feed output as the next input
      _inputBuffer.setData(outputData, _bufferSize);
    });

    if (_loopMode && _isLoopRunning && mounted) {
      Future.delayed(Duration(milliseconds: (_loopDelay * 1000).toInt()), () {
        if (mounted && _isLoopRunning) _runKernel();
      });
    }
  }

  void _toggleLoop() {
    if (_loopMode) {
      if (_isLoopRunning) {
        // Stop loop
        setState(() {
          _isLoopRunning = false;
        });
      } else {
        // Start loop
        setState(() {
          _isLoopRunning = true;
        });
        _runKernel();
      }
    } else {
      // If loop mode is off, just execute once.
      _runKernel();
    }
  }

  @override
  void dispose() {
    _shader.destroy();
    _inputBuffer.destroy();
    _outputBuffer.destroy();
    _shaderController.dispose();
    super.dispose();
  }

  Widget _buildShaderEditor() {
    return Card(
      margin: EdgeInsets.all(8),
      child: Padding(
        padding: EdgeInsets.all(8),
        child: TextField(
          controller: _shaderController,
          maxLines: null,
          decoration: InputDecoration(
            border: OutlineInputBorder(),
            labelText: 'WGSL Shader Code',
          ),
          style: TextStyle(fontFamily: 'monospace'),
        ),
      ),
    );
  }

  Widget _buildBufferControls() {
    return SingleChildScrollView(
        child: Card(
      margin: EdgeInsets.all(8),
      child: Column(
        children: [
          // Config Panel: Wrap automatically wraps buttons when space is limited.
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Wrap(
              spacing: 8,
              runSpacing: 8,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                // Preview length slider
                Column(
                  children: [
                    Text('Preview Length: $_previewLength'),
                    SizedBox(
                      width: 150,
                      child: Slider(
                        min: 1,
                        max: _bufferSize.toDouble(),
                        value: _previewLength.toDouble(),
                        onChanged: (val) {
                          setState(() {
                            _previewLength = val.toInt();
                            // Update preview based on current data.
                            _result = _result.sublist(
                                0, _previewLength.clamp(0, _result.length));
                          });
                        },
                      ),
                    ),
                  ],
                ),
                // Buffer size slider
                Column(
                  children: [
                    Text('Buffer Size: $_bufferSize'),
                    SizedBox(
                      width: 150,
                      child: Slider(
                        min: 10,
                        max: 200,
                        divisions: 40,
                        value: _bufferSize.toDouble(),
                        onChanged: (val) {
                          setState(() {
                            _bufferSize = val.toInt();
                            if (_previewLength > _bufferSize) {
                              _previewLength = _bufferSize;
                            }
                            _createBuffers();
                            _initializeBuffer();
                            _shader.setBuffer('inp', _inputBuffer);
                            _shader.setBuffer('out', _outputBuffer);
                          });
                        },
                      ),
                    ),
                  ],
                ),
                // Loop mode switch
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text('Loop Mode'),
                    Switch(
                      value: _loopMode,
                      onChanged: (val) {
                        setState(() {
                          _loopMode = val;
                          // If turning off loop mode, also stop any active loop.
                          if (!_loopMode) {
                            _isLoopRunning = false;
                          }
                        });
                      },
                    ),
                  ],
                ),
                // Loop delay slider (visible only if loop mode is enabled)
                if (_loopMode)
                  Column(
                    children: [
                      Text('Loop Delay: ${_loopDelay.toStringAsFixed(1)} s'),
                      SizedBox(
                        width: 150,
                        child: Slider(
                          min: 0.1,
                          max: 5.0,
                          divisions: 49,
                          value: _loopDelay,
                          onChanged: (val) {
                            setState(() {
                              _loopDelay = val;
                            });
                          },
                        ),
                      ),
                    ],
                  ),
                // Execute / Loop Toggle button
                ElevatedButton(
                  onPressed: _toggleLoop,
                  child: Text(_loopMode
                      ? (_isLoopRunning ? 'Stop Loop' : 'Start Loop')
                      : 'Execute Shader'),
                ),
                // Other control buttons
                ElevatedButton(
                  onPressed: _resetBuffer,
                  child: Text('Reset'),
                ),
                ElevatedButton(
                  onPressed: _initializeBuffer,
                  child: Text('Initialize'),
                ),
                ElevatedButton(
                  onPressed: _randomizeBuffer,
                  child: Text('Randomize'),
                ),
              ],
            ),
          ),
          Divider(),
          // Data Preview
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Buffer Preview (first $_previewLength values):',
                  style: TextStyle(fontSize: 18),
                ),
                SizedBox(height: 8),
                Container(
                  width: double.infinity,
                  padding: EdgeInsets.all(8),
                  color: Colors.black12,
                  child: Text(
                    _result.map((v) => v.toStringAsFixed(2)).join(', '),
                    style: TextStyle(fontSize: 16),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    ));
  }

  // For wide and narrow layouts, shader editor remains unchanged.
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Interactive Shader Example'),
      ),
      body: LayoutBuilder(builder: (context, constraints) {
        if (constraints.maxWidth > 800) {
          return Column(
            children: [
              Expanded(
                child: Row(
                  children: [
                    Expanded(child: _buildShaderEditor()),
                    Expanded(child: _buildBufferControls()),
                  ],
                ),
              ),
            ],
          );
        } else {
          return Column(
            children: [
              Expanded(child: _buildShaderEditor()),
              Expanded(child: _buildBufferControls()),
            ],
          );
        }
      }),
    );
  }
}


# minigpu

 A Flutter library for cross-platform GPU compute shaders integrating WGSL, GPU.CPP, and WebGPU via Dawn.

https://docs.flutter.dev/get-started/install

```console
cd minigpu/example
flutter run -d Windows
flutter run -d Linux
flutter build apk
```

- [x] Windows
- [x] Linux
- [ ] Mac - Try it and open issue!
- [x] Web
- [x] Android
- [ ] iOS - Try it and open issue!

## Example

 ```dart

Future<void> _runGELU() async {
// Initialize the GPU.
    await _minigpu = Minigpu();
// Create the compute shader.
    _shader = _minigpu.createComputeShader();

// Load the compute kernel code as a string.
    _shader.loadKernelString('''
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>
) {
  let i: u32 = GlobalInvocationID.x;
  if (i < arrayLength(&inp)) {
    let x: f32 = inp[i];
    out[i] = select(
      0.5 * x * (1.0 + tanh(
        GELU_SCALING_FACTOR * (x + .044715 * x * x * x)
      )),
      x,
      x > 10.0
    );
  }
}
''');

// Define the buffer size and generate input data.
  final bufferSize = 100;
  final inputData = Float32List.fromList(
    List<double>.generate(bufferSize, (i) => i / 10.0),
  );
  print('bufferSize: ${inputData.lengthInBytes}');
  final memSize = bufferSize * 4; // 4 bytes per float32

// Create GPU buffers for input and output data.
  _inputBuffer = _minigpu.createBuffer(bufferSize, memSize);
  _outputBuffer = _minigpu.createBuffer(bufferSize, memSize);

// Upload the input data.
  _inputBuffer.setData(inputData, bufferSize);

// Bind the buffers to the shader.
  _shader.setBuffer('inp', _inputBuffer);
  _shader.setBuffer('out', _outputBuffer);

// Calculate the number of workgroups required.
  final workgroups = ((bufferSize + 255) / 256).floor();

// Dispatch the compute shader.
  await _shader.dispatch(workgroups, 1, 1);

// Read the output data.
  final outputData = Float32List(bufferSize);
  await _outputBuffer.read(outputData, bufferSize);

// Update the UI 
  setState(() {
    _result = outputData.sublist(0, 16).map((value) => value.toDouble()).toList();
    print('Result: $_result');
  });
}
  ```

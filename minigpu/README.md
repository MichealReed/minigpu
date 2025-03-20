
# minigpu

A Flutter library for cross-platform GPU compute shaders integrating WGSL, GPU.CPP, and WebGPU via Dawn.

[minigpu.practicalxr.com](https://minigpu.practicalxr.com/)

[pub.dev](https://pub.dev/packages/minigpu)

## Three things to know

1. Dawn can take a while to build, so run with -v to see progress.
2. This uses dart native assets, you must use the flag on the dev channel `dart --enable-experiment=native-assets` or for flutter must be on the master channel and
`flutter config --enable-native-assets` is needed once.
3. The root makefile can be used to regenerate bindings or the web plugin.

## Getting Started

```console
# clone the repo
git clone https://github.com/PracticalXR/minigpu.git

# Try dart
cd minigpu
dart --enable-experiment=native-assets bin/example.dart -v
dart --enable-experiment=native-assets test

# Try flutter
cd minigpu/example
flutter config --enable-native-assets
flutter run -d windows/linux/macos/chrome/...
```

- [x] Windows
- [x] Linux
- [x] Mac
- [x] Web
- [x] Android
- [x] iOS

## Example

 ```dart

Future<void> _runGELU() async {
  // Initialize the GPU.
  final gpu = Minigpu();
  await gpu.init();
  // Create the compute shader.
  final shader = gpu.createComputeShader();

// Load the compute kernel code as a string.
  shader.loadKernelString('''
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
  final inputBuffer = gpu.createBuffer(bufferSize, memSize);
  final outputBuffer = gpu.createBuffer(bufferSize, memSize);

// Upload the input data.
  inputBuffer.setData(inputData, bufferSize);

// Bind the buffers to the shader.
  shader.setBuffer('inp', inputBuffer);
  shader.setBuffer('out', outputBuffer);

// Calculate the number of workgroups required.
  final workgroups = ((bufferSize + 255) / 256).floor();

// Dispatch the compute shader.
  await shader.dispatch(workgroups, 1, 1);

// Read the output data.
  final outputData = Float32List(bufferSize);
  await outputBuffer.read(outputData, bufferSize);

// Update the UI 
  setState(() {
    final result = outputData.sublist(0, 16).map((value) => value.toDouble()).toList();
    print('Result: $result');
  });
}
  ```

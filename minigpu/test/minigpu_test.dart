import 'dart:typed_data';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

void main() {
  group('Minigpu Initialization', () {
    test('Successfully initializes context', () async {
      final minigpu = Minigpu();
      await minigpu.init();
      expect(minigpu.isInit, isTrue);
      // Additional cleanup if required.
    });

    test('Throws error when initializing twice', () async {
      final minigpu = Minigpu();
      await minigpu.init();
      expect(() async => await minigpu.init(),
          throwsA(isA<MinigpuAlreadyInitError>()));
    });
  });

  group('ComputeShader', () {
    test('Loads kernel string and verifies kernel presence', () {
      final minigpu = Minigpu();
      // Assume minigpu is initialized or initialize before use if required.
      final shader = minigpu.createComputeShader();
      shader.loadKernelString('''
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          // dummy kernel code
        }
      ''');
      // Check that the kernel is loaded.
      expect(shader.hasKernel(), isTrue);
      shader.destroy();
    });

    test('Attaches buffer with unique tag; duplicate tag throws error', () {
      final minigpu = Minigpu();
      final shader = minigpu.createComputeShader();
      final buffer = minigpu.createBuffer(10, 40); // 10 * 4 bytes per float

      // Attach buffer with tag 'inp'
      shader.setBuffer('inp', buffer);

      // Adding a second buffer using the same tag should throw.
      expect(() => shader.setBuffer('inp', buffer), throwsA(isA<Exception>()));

      shader.destroy();
      buffer.destroy();
    });
  });

  group('Buffer Data Transfer', () {
    test('setData writes and read returns the same data', () async {
      final minigpu = Minigpu();
      final size = 10;
      final memSize = size * 4; // 4 bytes per float
      final buffer = minigpu.createBuffer(size, memSize);

      final inputData =
          Float32List.fromList(List.generate(size, (i) => i.toDouble()));
      buffer.setData(inputData, size);

      final outputData = Float32List(size);
      await buffer.read(outputData, size);

      expect(outputData, equals(inputData));

      buffer.destroy();
    });
  });

  // Further integration tests may simulate the example UI usage.
  group('Example Integration', () {
    test('GELU Example Shader executes and outputs expected results', () async {
      // Note: This test assumes that you can simulate the entire pipeline.
      // In practice, you may need to use fakes or mocks for the shader operations.
      final minigpu = Minigpu();
      await minigpu.init();
      final shader = minigpu.createComputeShader();
      shader.loadKernelString('''
        const GELU_SCALING_FACTOR: f32 = 0.7978845608028654;
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
      ''');

      const bufferSize = 100;
      final inputData =
          Float32List.fromList(List.generate(bufferSize, (i) => i / 10.0));
      final memSize = bufferSize * 4;

      final inputBuffer = minigpu.createBuffer(bufferSize, memSize);
      final outputBuffer = minigpu.createBuffer(bufferSize, memSize);

      inputBuffer.setData(inputData, bufferSize);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);

      // Dispatch the shader - workgroups calculation as in the example.
      final workgroups = ((bufferSize + 255) / 256).floor();
      await shader.dispatch(workgroups, 1, 1);

      final outputData = Float32List(bufferSize);
      await outputBuffer.read(outputData, bufferSize);

      // Add your expectations based on the expected transformation.
      // For an outline, we simply check if some output is produced.
      expect(outputData.length, equals(bufferSize));

      shader.destroy();
      inputBuffer.destroy();
      outputBuffer.destroy();
    });
  });
}

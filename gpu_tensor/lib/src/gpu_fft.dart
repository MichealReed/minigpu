import 'dart:math' as math;
import 'dart:typed_data';
import 'package:minigpu/minigpu.dart';
import '../gpu_tensor.dart';

extension GpuFft on Tensor {
  /// Computes a 1D FFT on a tensor representing complex numbers in interleaved format.
  /// The tensor must be 1D and have an even number of elements, with exactly 2*n floats,
  /// where n is a power of two representing the FFT size.
  Future<Tensor> fft1d() async {
    if (shape.length != 1) {
      throw Exception("FFT supports 1D tensors only.");
    }
    int totalFloats = shape[0];
    if (totalFloats % 2 != 0) {
      throw Exception("FFT tensor length must be even (for complex numbers).");
    }
    int n = totalFloats ~/ 2; // number of complex points
    if ((n & (n - 1)) != 0) {
      throw Exception("FFT size ($n) must be a power of 2.");
    }
    if (n == 1) return this; // FFT of a single point

    // Number of stages = log2(n)
    int stages = (math.log(n) / math.ln2).toInt();

    // Double buffering: ping holds the current data, pong is an auxiliary buffer.
    Tensor ping = this;
    Tensor pong = await Tensor.create(shape, gpu: gpu);

    // For each FFT stage, perform butterfly computations.
    for (int s = 0; s < stages; s++) {
      int m = 1 << (s + 1); // size of current FFT block
      int half = m >> 1; // half the block size
      // There are always n/2 butterflies per stage.
      int numOperations = n >> 1;

      final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t: u32 = gid.x;
  if (t >= ${numOperations}u) {
    return;
  }
  let half: u32 = ${half}u;
  let m: u32 = ${m}u;
  let group: u32 = t / half;
  let pos: u32 = t % half;
  let i0: u32 = group * m + pos;
  let i1: u32 = i0 + half;
  let idx0: u32 = i0 * 2u;
  let idx1: u32 = i1 * 2u;
  let a: vec2<f32> = vec2<f32>(input[idx0], input[idx0 + 1u]);
  let b: vec2<f32> = vec2<f32>(input[idx1], input[idx1 + 1u]);
  let angle: f32 = -6.28318530718 * f32(pos) / f32(m);
  let w: vec2<f32> = vec2<f32>(cos(angle), sin(angle));
  let b_twiddled: vec2<f32> = vec2<f32>(
    b.x * w.x - b.y * w.y,
    b.x * w.y + b.y * w.x
  );
  let temp1: vec2<f32> = a + b_twiddled;
  let temp2: vec2<f32> = a - b_twiddled;
  output[idx0] = temp1.x;
  output[idx0 + 1u] = temp1.y;
  output[idx1] = temp2.x;
  output[idx1 + 1u] = temp2.y;
}
''';

      final ComputeShader shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      int workgroups = (numOperations + 255) ~/ 256;
      await shader.dispatch(workgroups, 1, 1);
      shader.destroy();

      // Swap buffers for the next stage.
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }
    // After all stages, 'ping' holds the FFT result.
    return ping;
  }

  Future<Tensor> fft2d() async {
    // If a 2D real tensor is supplied, upgrade it.
    if (shape.length == 2) {
      int rows = shape[0];
      int cols = shape[1];
      // Check that rows and cols are powers of 2.
      bool isPow2(int x) => (x & (x - 1)) == 0;
      if (!isPow2(rows) || !isPow2(cols)) {
        throw Exception("Both rows and cols must be powers of 2.");
      }
      final Float32List realData = await getData();
      final Float32List complexData = Float32List(rows * cols * 2);
      for (int i = 0; i < rows * cols; i++) {
        complexData[i * 2] = realData[i];
        complexData[i * 2 + 1] = 0.0;
      }
      return (await Tensor.create([rows, cols, 2], data: complexData, gpu: gpu))
          .fft2d();
    }
    // Otherwise, expect a tensor with shape [rows, cols, 2].
    if (shape.length != 3 || shape[2] != 2) {
      throw Exception("fft2d requires a tensor of shape [rows, cols, 2].");
    }
    int rows = shape[0];
    int cols = shape[1];
    // Check that rows and cols are powers of 2.
    bool isPow2(int x) => (x & (x - 1)) == 0;
    if (!isPow2(rows) || !isPow2(cols)) {
      throw Exception("Both rows and cols must be powers of 2.");
    }

    // Perform FFT on rows.
    int stagesRow = (math.log(cols) / math.ln2).toInt();
    Tensor ping = this;
    Tensor pong = await Tensor.create(shape, gpu: gpu);
    for (int s = 0; s < stagesRow; s++) {
      int m = 1 << (s + 1); // current FFT block size on row
      int half = m >> 1; // half block size
      int numOperations = rows * (cols >> 1);
      final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t: u32 = gid.x;
  if (t >= ${numOperations}u) { return; }
  let row: u32 = t / ${(cols >> 1)}u;
  let t_row: u32 = t % ${(cols >> 1)}u;
  let half: u32 = ${half}u;
  let m: u32 = ${m}u;
  let group: u32 = t_row / half;
  let pos: u32 = t_row % half;
  let i0: u32 = group * m + pos;
  let i1: u32 = i0 + half;
  let base: u32 = row * (${cols}u * 2u);
  let idx0: u32 = base + i0 * 2u;
  let idx1: u32 = base + i1 * 2u;
  let a: vec2<f32> = vec2<f32>(input[idx0], input[idx0+1u]);
  let b: vec2<f32> = vec2<f32>(input[idx1], input[idx1+1u]);
  let angle: f32 = -6.28318530718 * f32(pos) / f32(m);
  let w: vec2<f32> = vec2<f32>(cos(angle), sin(angle));
  let b_twiddled: vec2<f32> = vec2<f32>(
    b.x * w.x - b.y * w.y,
    b.x * w.y + b.y * w.x
  );
  let temp1: vec2<f32> = a + b_twiddled;
  let temp2: vec2<f32> = a - b_twiddled;
  output[idx0] = temp1.x;
  output[idx0+1u] = temp1.y;
  output[idx1] = temp2.x;
  output[idx1+1u] = temp2.y;
}
''';
      final ComputeShader shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      int workgroups = (numOperations + 255) ~/ 256;
      await shader.dispatch(workgroups, 1, 1);
      shader.destroy();
      // Swap ping and pong.
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }

    // Perform FFT on columns.
    int stagesCol = (math.log(rows) / math.ln2).toInt();
    pong = await Tensor.create(shape, gpu: gpu);
    for (int s = 0; s < stagesCol; s++) {
      int m = 1 << (s + 1); // current FFT block size on column
      int half = m >> 1; // half block size
      int numOperations = cols * (rows >> 1);
      final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t: u32 = gid.x;
  if (t >= ${numOperations}u) { return; }
  let col: u32 = t / ${(rows >> 1)}u;
  let t_col: u32 = t % ${(rows >> 1)}u;
  let half: u32 = ${half}u;
  let m: u32 = ${m}u;
  let group: u32 = t_col / half;
  let pos: u32 = t_col % half;
  let i0: u32 = group * m + pos;
  let i1: u32 = i0 + half;
  // For column FFT, each column's elements are separated by row stride = (cols * 2)
  let stride: u32 = ${cols}u * 2u;
  let base0: u32 = i0 * stride + col * 2u;
  let base1: u32 = i1 * stride + col * 2u;
  let a: vec2<f32> = vec2<f32>(input[base0], input[base0+1u]);
  let b: vec2<f32> = vec2<f32>(input[base1], input[base1+1u]);
  let angle: f32 = -6.28318530718 * f32(pos) / f32(m);
  let w: vec2<f32> = vec2<f32>(cos(angle), sin(angle));
  let b_twiddled: vec2<f32> = vec2<f32>(
    b.x * w.x - b.y * w.y,
    b.x * w.y + b.y * w.x
  );
  let temp1: vec2<f32> = a + b_twiddled;
  let temp2: vec2<f32> = a - b_twiddled;
  output[base0] = temp1.x;
  output[base0+1u] = temp1.y;
  output[base1] = temp2.x;
  output[base1+1u] = temp2.y;
}
''';
      final ComputeShader shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      int workgroups = (numOperations + 255) ~/ 256;
      await shader.dispatch(workgroups, 1, 1);
      shader.destroy();
      // Swap buffers.
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }
    return ping;
  }
}

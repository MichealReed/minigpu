import 'package:minigpu/minigpu.dart';

import 'gpu_tensor_base.dart';

extension TensorLinearOperator on Tensor {
  /// helper function to check if two batch shapes are equal
  bool _batchShapesEqual(List<int> shapeA, List<int> shapeB) {
    if (shapeA.length != shapeB.length) return false;
    for (int i = 0; i < shapeA.length; i++) {
      if (shapeA[i] != shapeB[i]) return false;
    }
    return true;
  }

  /// Matrix multiplication (dot product)
  /// for 2D tensors or batched matrix multiplication for higher dimensions.
  Future<Tensor> matMul(Tensor other) async {
// Both tensors must have rank at least 2.
    if (rank < 2 || other.rank < 2) {
      throw Exception("matMul requires tensors with rank >= 2.");
    }

    // Handle rank-2 matrix multiplication.
    if (rank == 2 && other.rank == 2) {
      int m = shape[0];
      int n = shape[1];
      if (other.shape[0] != n) {
        throw Exception(
            "Inner dimensions do not match for matrix multiplication.");
      }
      int p = other.shape[1];
      // The result has shape [m, p].
      Tensor result = await Tensor.create([m, p]);

      final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row: u32 = gid.x;
  let col: u32 = gid.y;
  if (row < ${m}u && col < ${p}u) {
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < ${n}u; k = k + 1u) {
      let aIndex = row * ${n}u + k;
      let bIndex = k * ${p}u + col;
      sum = sum + A[aIndex] * B[bIndex];
    }
    let cIndex = row * ${p}u + col;
    C[cIndex] = sum;
  }
}
''';

      final ComputeShader shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('A', buffer);
      shader.setBuffer('B', other.buffer);
      shader.setBuffer('C', result.buffer);
      int wgX = (m + 7) ~/ 8;
      int wgY = (p + 7) ~/ 8;
      await shader.dispatch(wgX, wgY, 1);
      shader.destroy();
      return result;
    } else {
// Assume we are doing batched matrix multiplication:
// The left tensor has matrix dimensions [m, n] and the right [n, p]
// where the batch is given by all preceding dimensions (which must match).
      int m = shape[rank - 2];
      int n = shape.last; // also left inner dim
      int p = other.shape.last;

// Get batch dimensions:
      List<int> batchShapeA = shape.sublist(0, rank - 2);
      List<int> batchShapeB = other.shape.sublist(0, other.rank - 2);
// For simplicity, require exact equality of batch dims.
      if (!_batchShapesEqual(batchShapeA, batchShapeB)) {
        throw Exception("Batch dimensions must match for batched matMul.");
      }
      int batch = batchShapeA.isEmpty ? 1 : batchShapeA.reduce((a, b) => a * b);

// The result shape is [batchShape, m, p]
      List<int> resultShape = List.from(batchShapeA)..addAll([m, p]);
      Tensor result = await Tensor.create(resultShape);

// Use a shader aware of the batch dimension (using 3D dispatch):
      final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
let b: u32 = gid.z;
let row: u32 = gid.x;
let col: u32 = gid.y;
if (b < ${batch}u && row < ${m}u && col < ${p}u) {
var sum: f32 = 0.0;
let offsetA: u32 = b * ${m * n}u;
let offsetB: u32 = b * ${n * p}u;
for (var k: u32 = 0u; k < ${n}u; k = k + 1u) {
let indexA = offsetA + row * ${n}u + k;
let indexB = offsetB + k * ${p}u + col;
sum = sum + A[indexA] * B[indexB];
}
let indexC: u32 = b * ${m * p}u + row * ${p}u + col;
C[indexC] = sum;
}
}
''';
      final ComputeShader shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('A', buffer);
      shader.setBuffer('B', other.buffer);
      shader.setBuffer('C', result.buffer);

// Compute workgroup counts.
      int wgX = (m + 7) ~/ 8;
      int wgY = (p + 7) ~/ 8;
      int wgZ = batch; // Or choose to workgroup-split this if needed.
      await shader.dispatch(wgX, wgY, wgZ);
      shader.destroy();
      return result;
    }
  }

  /// Performs a convolution supporting dilation and multi-channel input.
  ///
  /// For a single-channel (2D) input with a 2D kernel, this falls back to the
  /// original conv2d implementation. For multi-channel inputs:
  /// - The input tensor shape must be [H, W, Cin].
  /// - The kernel tensor shape must be [kH, kW, Cin, Cout].
  /// - The output shape is computed as:
  ///   [floor((H + 2*padH - dilationH*(kH-1) - 1)/strideH)+1,
  ///    floor((W + 2*padW - dilationW*(kW-1) - 1)/strideW)+1, Cout].
  /// Note: This convolution implementation assumes that multi-channel
  /// input is stored in a planar (channel‑first) format, meaning that all
  /// values for channel 0 are stored contiguously, followed by all values
  /// for channel 1, etc.

  Future<Tensor> conv({
    required Tensor kernel,
    int strideH = 1,
    int strideW = 1,
    int padH = 0,
    int padW = 0,
    int dilationH = 1,
    int dilationW = 1,
  }) async {
    // Multi-channel convolution: input rank 3 and kernel rank 4.
    if (rank == 3 && kernel.rank == 4) {
      int H = shape[0];
      int W = shape[1];
      int Cin = shape[2];
      int kH = kernel.shape[0];
      int kW = kernel.shape[1];
      int kernelCin = kernel.shape[2];
      int Cout = kernel.shape[3];

      if (Cin != kernelCin) {
        throw Exception(
            "Input channels ($Cin) do not match kernel channels ($kernelCin).");
      }
      // Effective kernel size after dilation.
      int eff_kH = dilationH * (kH - 1) + 1;
      int eff_kW = dilationW * (kW - 1) + 1;
      // Compute output dimensions.
      int outH = ((H + 2 * padH - eff_kH) ~/ strideH) + 1;
      int outW = ((W + 2 * padW - eff_kW) ~/ strideW) + 1;

      // Create output tensor with shape [outH, outW, Cout].
      Tensor result = await Tensor.create([outH, outW, Cout], gpu: gpu);

      final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Input dimensions
const H: u32 = ${H}u;
const W: u32 = ${W}u;
const Cin: u32 = ${Cin}u;
// Kernel dimensions
const kH: u32 = ${kH}u;
const kW: u32 = ${kW}u;
const Cout: u32 = ${Cout}u;
// Output dimensions
const outH: u32 = ${outH}u;
const outW: u32 = ${outW}u;
// Convolution parameters
const sH: u32 = ${strideH}u;
const sW: u32 = ${strideW}u;
const pH: i32 = $padH;
const pW: i32 = $padW;
const dH: u32 = ${dilationH}u;
const dW: u32 = ${dilationW}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx >= outH * outW * Cout) {
    return;
  }
  // Decode output indices.
  let tmp: u32 = idx / Cout;  // flatten [outH, outW]
  let o_ch: u32 = idx % Cout;
  let o_row: u32 = tmp / outW;
  let o_col: u32 = tmp % outW;

  var sum: f32 = 0.0;
  // Compute starting coordinate in the input.
  let inRowStart: i32 = i32(o_row * sH) - pH;
  let inColStart: i32 = i32(o_col * sW) - pW;

  // Loop over the kernel spatial dimensions.
  for (var i: u32 = 0u; i < kH; i = i + 1u) {
    for (var j: u32 = 0u; j < kW; j = j + 1u) {
      // Calculate input coordinate considering dilation.
      let inRow: i32 = inRowStart + i32(i * dH);
      let inCol: i32 = inColStart + i32(j * dW);
      for (var c: u32 = 0u; c < Cin; c = c + 1u) {
        var inputVal: f32 = 0.0;
        if (inRow >= 0 && inRow < i32(H) && inCol >= 0 && inCol < i32(W)) {
          // Use channel-first (planar) ordering.
          let inputIndex: u32 = c * (H * W) + (u32(inRow) * W + u32(inCol));
          inputVal = input[inputIndex];
        }
        // Kernel index: kernel shape is [kH, kW, Cin, Cout].
        let kernelIndex: u32 = ((i * kW + j) * Cin + c) * Cout + o_ch;
        sum = sum + inputVal * kernel[kernelIndex];
      }
    }
  }
  // Write the result. Output shape is [outH, outW, Cout].
  let outIndex: u32 = (o_row * outW + o_col) * Cout + o_ch;
  output[outIndex] = sum;
}
''';
      final ComputeShader shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', buffer);
      shader.setBuffer('kernel', kernel.buffer);
      shader.setBuffer('output', result.buffer);
      int workgroups = ((outH * outW * Cout) + 255) ~/ 256;
      await shader.dispatch(workgroups, 1, 1);
      shader.destroy();
      return result;
    } else {
      // Fallback: for 2D input and 2D kernel use the existing conv2d.
      return conv2d(kernel);
    }
  }

  /// Performs a 2D convolution on a 2D input tensor using a 2D kernel.
  ///
  /// The input tensor must have shape [H, W] and the kernel tensor must have
  /// shape [kH, kW]. A valid convolution is performed (no padding, stride 1),
  /// so the output tensor will have shape [H - kH + 1, W - kW + 1].
  Future<Tensor> conv2d(Tensor kernel) async {
    // Validate that both input and kernel are rank 2.
    if (rank != 2 || kernel.rank != 2) {
      throw Exception(
          "conv2d requires both input and kernel to be 2D tensors.");
    }
    int H = shape[0];
    int W = shape[1];
    int kH = kernel.shape[0];
    int kW = kernel.shape[1];
    if (H < kH || W < kW) {
      throw Exception(
          "Kernel dimensions must be smaller than or equal to input dimensions.");
    }
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    int totalOut = outH * outW;
    // Create the output tensor.
    Tensor result = await Tensor.create([outH, outW], gpu: gpu);

    // WGSL shader code: each invocation computes one output pixel.
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const H: u32 = ${H}u;
const W: u32 = ${W}u;
const kH: u32 = ${kH}u;
const kW: u32 = ${kW}u;
const outH: u32 = ${outH}u;
const outW: u32 = ${outW}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx >= outH * outW) {
    return;
  }
  // Compute row, col in output.
  let row: u32 = idx / outW;
  let col: u32 = idx % outW;
  var sum: f32 = 0.0;
  // Loop over the kernel.
  for (var i: u32 = 0u; i < kH; i = i + 1u) {
    for (var j: u32 = 0u; j < kW; j = j + 1u) {
      // Input index: corresponding row and col are shifted by kernel offsets.
      let inRow: u32 = row + i;
      let inCol: u32 = col + j;
      let inputIndex: u32 = inRow * W + inCol;
      let kernelIndex: u32 = i * kW + j;
      sum = sum + input[inputIndex] * kernel[kernelIndex];
    }
  }
  output[idx] = sum;
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('kernel', kernel.buffer);
    shader.setBuffer('output', result.buffer);
    int workgroups = (totalOut + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }
}

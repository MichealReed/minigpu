import 'package:minigpu/minigpu.dart';
import '../gpu_tensor.dart';

extension GpuActivation on Tensor {
  /// Applies the ReLU activation function elementwise.
  Future<Tensor> relu() async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    let v: f32 = input[i];
    output[i] = select(v, 0.0, v < 0.0);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Applies the Sigmoid activation function elementwise.
  Future<Tensor> sigmoid() async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    output[i] = sigmoid(input[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Computes the sine of each element.
  Future<Tensor> sin() async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if(i < ${size}u) {
    B[i] = sin(A[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int wk = (size + 255) ~/ 256;
    await shader.dispatch(wk, 1, 1);
    shader.destroy();
    return result;
  }

  /// Computes the cosine of each element.
  Future<Tensor> cos() async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if(i < ${size}u) {
    B[i] = cos(A[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int wk = (size + 255) ~/ 256;
    await shader.dispatch(wk, 1, 1);
    shader.destroy();
    return result;
  }

  /// Applies the Tanh activation function elementwise.
  Future<Tensor> tanh() async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

fn tanh_func(x: f32) -> f32 {
  let expPos = exp(x);
  let expNeg = exp(-x);
  return (expPos - expNeg) / (expPos + expNeg);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    output[i] = tanh_func(input[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Applies the Softmax activation function along the last dimension.
  /// For higher dimensional tensors, softmax is applied to each slice
  /// of size (last dimension) independently.
  Future<Tensor> softmax({int axis = -1}) async {
    // For now, we only support softmax along the last dimension.
    int d = shape.last; // inner dimension size
    int total = size;
    // Optionally, throw if axis is not last dimension:
    if (axis != -1 && axis != shape.length - 1) {
      throw Exception(
          "Currently only softmax along the last dimension is supported.");
    }
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let global: u32 = gid.x;
  if (global < ${total}u) {
    let d: u32 = ${d}u;
    let batchIndex: u32 = global / d;
    let offset: u32 = batchIndex * d;
    
    // Compute maximum value within this softmax group.
    var max_val: f32 = input[offset];
    for (var j: u32 = 1u; j < d; j = j + 1u) {
      max_val = max(max_val, input[offset + j]);
    }
    
    let shifted: f32 = input[global] - max_val;
    let exp_val: f32 = exp(shifted);
    var sum_exp: f32 = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) {
      sum_exp = sum_exp + exp(input[offset + j] - max_val);
    }
    output[global] = exp_val / sum_exp;
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    int workgroups = (total + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }
}

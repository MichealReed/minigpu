import 'package:minigpu/minigpu.dart';
import '../gpu_tensor.dart';

extension GpuActivation on Tensor {
  /// Applies the ReLU activation function elementwise.
  Future<Tensor> relu() async {
    Tensor result = await Tensor.create(shape, gpu: gpu);
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
    Tensor result = await Tensor.create(shape, gpu: gpu);
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
}

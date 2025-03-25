import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';

/// A helper that creates (or reuses) a default GPU context.
class DefaultMinigpu {
  static final instance = Minigpu();
}

/// A generalized tensor supporting any rank. Data is stored in a GPU buffer.
class Tensor {
  /// The shape in terms of dimensions: e.g. [3, 4, 5] for a 3×4×5 tensor.
  final List<int> shape;

  /// Total number of elements (computed as shape[0]shape[1]...).
  final int size;

  /// The rank (number of dimensions)
  int get rank => shape.length;

  /// The GPU context used by this tensor.
  final Minigpu gpu;

  /// The GPU buffer storing the data.
  late Buffer buffer;

// Private constructor.
  Tensor._(this.shape, {required this.gpu, Float32List? data})
      : size = shape.reduce((a, b) => a * b) {
// Each float is 4 bytes.
    buffer = gpu.createBuffer(size * 4);
    if (data != null) {
      if (data.length != size) {
        throw Exception(
            "Provided data length (${data.length}) does not match tensor size ($size)");
      }
      buffer.setData(data, size);
    } else {
// Initialize with zero.
      buffer.setData(Float32List(size), size);
    }
  }

  /// Asynchronous factory that initializes the GPU before creating the tensor.
  static Future<Tensor> create(List<int> shape,
      {Minigpu? gpu, Float32List? data}) async {
    gpu = gpu ?? DefaultMinigpu.instance;
    if (!gpu.isInitialized) {
      await gpu.init();
    }
    return Tensor._(shape, gpu: gpu, data: data);
  }

  void destroy() {
    buffer.destroy();
  }

  /// Creates a tensor by reusing an already existing [buffer] and specifying a new [shape].
  /// (This is useful for operations like reshape that do not need to copy data.)
  Tensor.fromBuffer(this.buffer, this.shape, {Minigpu? gpu})
      : gpu = gpu ?? DefaultMinigpu.instance,
        size = shape.reduce((a, b) => a * b);

  /// Reads back the data from the GPU buffer.
  Future<Float32List> getData() async {
    final Float32List data = Float32List(size);
    await buffer.read(data, size);
    return data;
  }

  void setData(Float32List data) {
    buffer.setData(data, size);
  }

  /// Elementwise addition. Returns a new tensor with the result.
  /// (Assumes both tensors have the same shape.)
  Future<Tensor> add(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise addition");
    }
    Tensor result = await Tensor.create(shape, gpu: gpu);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
let i: u32 = gid.x;
if (i < ${size}u) {
C[i] = A[i] + B[i];
}
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Elementwise subtraction.
  Future<Tensor> subtract(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise subtraction");
    }
    Tensor result = await Tensor.create(shape, gpu: gpu);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
let i: u32 = gid.x;
if (i < ${size}u) {
C[i] = A[i] - B[i];
}
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Reshapes the tensor into a new shape without changing the underlying data.
  /// Throws an exception if the total number of elements would differ.
  Tensor reshape(List<int> newShape) {
    int newSize = newShape.reduce((a, b) => a * b);
    if (newSize != size) {
      throw Exception(
          "New shape $newShape does not match total number of elements $size");
    }
    return Tensor.fromBuffer(buffer, newShape, gpu: gpu);
  }

  /// Operator overloads for more natural syntax.
  Future<Tensor> operator +(Tensor other) => add(other);
  Future<Tensor> operator -(Tensor other) => subtract(other);
  Future<Tensor> operator *(dynamic other) async {
    if (other is num) {
      return multiplyScalar(other.toDouble());
    } else if (other is Tensor) {
      return multiply(other);
    } else {
      throw Exception("Unsupported operand type for *");
    }
  }

  /// Transposes a 2D tensor.
  /// Assumes the tensor’s shape is [m, n] with data stored in row-major order.
  Future<Tensor> transpose() async {
    if (shape.length != 2) {
      throw Exception("Transpose is only implemented for 2D tensors.");
    }
    int m = shape[0], n = shape[1];
    Tensor result = await Tensor.create([n, m], gpu: gpu);

    final String shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row: u32 = gid.x;
  let col: u32 = gid.y;
  if (row < ${m}u && col < ${n}u) {
    // Corrected transposition for complex numbers.
    output[(col * ${m}u + row) * 2u] = input[(row * ${n}u + col) * 2u];
    output[(col * ${m}u + row) * 2u + 1u] = input[(row * ${n}u + col) * 2u + 1u];
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer("input", buffer);
    shader.setBuffer("output", result.buffer);
    int wgX = (m + 15) ~/ 16;
    int wgY = (n + 15) ~/ 16;
    await shader.dispatch(wgX, wgY, 1);
    shader.destroy();

    return result;
  }

  /// Elementwise multiplication (Hadamard product).
  Future<Tensor> multiply(Tensor other) async {
    if (other.size != size) {
      throw Exception(
          "Tensor sizes do not match for elementwise multiplication");
    }
    Tensor result = await Tensor.create(shape, gpu: gpu);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
let i: u32 = gid.x;
if (i < ${size}u) {
C[i] = A[i] * B[i];
}
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Multiplies every element in the tensor by a scalar value.
  Future<Tensor> multiplyScalar(double scalar) async {
    Tensor result = await Tensor.create(shape, gpu: gpu);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
let i: u32 = gid.x;
if (i < ${size}u) {
B[i] = A[i] * $scalar;
}
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  // A helper (you might substitute a proper list comparison if not in Flutter)
  bool _batchShapesEqual(List<int> shapeA, List<int> shapeB) {
    if (shapeA.length != shapeB.length) return false;
    for (int i = 0; i < shapeA.length; i++) {
      if (shapeA[i] != shapeB[i]) return false;
    }
    return true;
  }

  /// Matrix multiplication (dot product) for rank-2 tensors.
  /// Assumes this tensor has shape [m, n] and [other] has shape [n, p].
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
      Tensor result = await Tensor.create([m, p], gpu: gpu);

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
      Tensor result = await Tensor.create(resultShape, gpu: gpu);

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
}

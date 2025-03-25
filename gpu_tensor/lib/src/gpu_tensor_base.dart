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
    Tensor result = await Tensor.create(shape);
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
    Tensor result = await Tensor.create(shape);
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

  /// Operator overloads for more natural syntax.
  Future<Tensor> operator +(dynamic other) async {
    if (other is num) {
      return addScalar(other.toDouble());
    } else if (other is Tensor) {
      return add(other);
    } else {
      throw Exception("Unsupported operand type for +");
    }
  }

  Future<Tensor> operator -(dynamic other) async {
    if (other is num) {
      return subtractScalar(other.toDouble());
    } else if (other is Tensor) {
      return subtract(other);
    } else {
      throw Exception("Unsupported operand type for -");
    }
  }

  Future<Tensor> operator *(dynamic other) async {
    if (other is num) {
      return multiplyScalar(other.toDouble());
    } else if (other is Tensor) {
      return multiply(other);
    } else {
      throw Exception("Unsupported operand type for *");
    }
  }

  Future<Tensor> operator /(dynamic other) async {
    if (other is num) {
      return divideScalar(other.toDouble());
    } else if (other is Tensor) {
      return divide(other);
    } else {
      throw Exception("Unsupported operand type for /");
    }
  }

  /// Elementwise multiplication (Hadamard product).
  Future<Tensor> multiply(Tensor other) async {
    if (other.size != size) {
      throw Exception(
          "Tensor sizes do not match for elementwise multiplication");
    }
    Tensor result = await Tensor.create(shape);
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

  /// Adds a scalar value to every element in the tensor.
  Future<Tensor> addScalar(double scalar) async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = A[i] + $scalar;
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

  /// Subtracts a scalar value from every element in the tensor.
  Future<Tensor> subtractScalar(double scalar) async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = A[i] - $scalar;
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

  /// Multiplies every element in the tensor by a scalar value.
  Future<Tensor> multiplyScalar(double scalar) async {
    Tensor result = await Tensor.create(shape);
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

  /// Matrix multiplication (dot product)
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

  /// Elementwise division (A / B).
  Future<Tensor> divide(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise division");
    }
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    C[i] = A[i] / B[i];
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

  /// Divides every element in the tensor by a scalar.
  Future<Tensor> divideScalar(double scalar) async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = A[i] / $scalar;
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

  /// Raises every element in the tensor to the power of [exponent].
  Future<Tensor> powScalar(double exponent) async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = pow(A[i], $exponent);
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

  /// Computes the natural logarithm (ln) of each element.
  Future<Tensor> log() async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = log(A[i]);
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

  /// Computes the exponential (e^x) of each element.
  Future<Tensor> exp() async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = exp(A[i]);
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

  /// Computes the square root of each element.
  Future<Tensor> sqrt() async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = sqrt(A[i]);
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

  /// Computes the modulus (remainder) of each element by [divisor].
  Future<Tensor> modScalar(double divisor) async {
    // Ensure the divisor is expressed as a float literal (e.g. "3.0")
    String divisorLiteral = divisor.toStringAsFixed(1);
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = A[i] % $divisorLiteral;
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

  /// Performs an elementwise "greater than" comparison.
  /// Returns a tensor where each element is 1.0 if A[i] > other[i], otherwise 0.0.
  Future<Tensor> greaterThan(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    if (A[i] > B[i]) {
      C[i] = 1.0;
    } else {
      C[i] = 0.0;
    }
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

  /// Performs an elementwise "less than" comparison.
  /// Returns a tensor where each element is 1.0 if A[i] < other[i], otherwise 0.0.
  Future<Tensor> lessThan(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    if (A[i] < B[i]) {
      C[i] = 1.0;
    } else {
      C[i] = 0.0;
    }
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

  /// Performs an elementwise equality comparison.
  /// Returns a tensor where each element is 1.0 if A[i] equals other[i], otherwise 0.0.
  Future<Tensor> equalTo(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    if (A[i] == B[i]) {
      C[i] = 1.0;
    } else {
      C[i] = 0.0;
    }
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

  /// Performs an elementwise "not equal" comparison.
  /// Returns a tensor where each element is 1.0 if A[i] != other[i], otherwise 0.0.
  Future<Tensor> notEqualTo(Tensor other) async {
    // Implement by reusing equalTo and then subtracting from a tensor of ones.
    Tensor eq = await equalTo(other);
    Tensor ones = await Tensor.create(shape);
    // Fill 'ones' with 1.0.
    ones = await ones.addScalar(1.0);
    // ones - eq gives 1.0 for not equal cases.
    Tensor result = await ones.subtract(eq);
    eq.destroy();
    ones.destroy();
    return result;
  }

  /// Performs an elementwise "greater than or equal to" comparison.
  /// Returns a tensor with 1.0 if A[i] >= other[i], otherwise 0.0.
  Future<Tensor> greaterThanOrEqual(Tensor other) async {
    // A >= B is equivalent to NOT (A < B)
    Tensor lt = await lessThan(other);
    Tensor ones = await Tensor.create(shape);
    ones = await ones.addScalar(1.0);
    Tensor result = await ones.subtract(lt);
    lt.destroy();
    ones.destroy();
    return result;
  }

  /// Performs an elementwise "less than or equal to" comparison.
  /// Returns a tensor with 1.0 if A[i] <= other[i], otherwise 0.0.
  Future<Tensor> lessThanOrEqual(Tensor other) async {
    // A <= B is equivalent to NOT (A > B)
    Tensor gt = await greaterThan(other);
    Tensor ones = await Tensor.create(shape);
    ones = await ones.addScalar(1.0);
    Tensor result = await ones.subtract(gt);
    gt.destroy();
    ones.destroy();
    return result;
  }

  /// Computes the absolute value of each element.
  Future<Tensor> abs() async {
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = abs(A[i]);
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
}

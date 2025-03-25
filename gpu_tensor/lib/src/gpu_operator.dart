import 'package:minigpu/minigpu.dart';

import 'gpu_tensor_base.dart';

extension TensorOperator on Tensor {
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

  /// Reduces the tensor by summing values along the last dimension.
  /// For a tensor of shape [..., d], returns a tensor of shape [...].
  Future<Tensor> sum({int axis = -1}) async {
    if (axis != -1 && axis != shape.length - 1) {
      throw Exception(
          "Currently only sum along the last dimension is supported.");
    }
    int d = shape.last;
    int batch = size ~/ d;
    List<int> outShape = shape.sublist(0, shape.length - 1);
    Tensor result = await Tensor.create(outShape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx < ${batch}u) {
    let d: u32 = ${d}u;
    let offset: u32 = idx * d;
    var sum: f32 = A[offset];
    for (var j: u32 = 1u; j < d; j = j + 1u) {
      sum = sum + A[offset + j];
    }
    B[idx] = sum;
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (batch + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Reduces the tensor by computing the mean along the last dimension.
  Future<Tensor> mean({int axis = -1}) async {
    // Compute sum then divide by d.
    Tensor sums = await sum(axis: axis);
    int d = shape.last;
    Tensor result = await sums.multiplyScalar(1.0 / d);
    sums.destroy();
    return result;
  }

  /// Reduces the tensor by taking the maximum value along the last dimension.
  Future<Tensor> maxReduction({int axis = -1}) async {
    if (axis != -1 && axis != shape.length - 1) {
      throw Exception(
          "Currently only max reduction along the last dimension is supported.");
    }
    int d = shape.last;
    int batch = size ~/ d;
    List<int> outShape = shape.sublist(0, shape.length - 1);
    Tensor result = await Tensor.create(outShape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx < ${batch}u) {
    let d: u32 = ${d}u;
    let offset: u32 = idx * d;
    var max_val: f32 = A[offset];
    for (var j: u32 = 1u; j < d; j = j + 1u) {
      max_val = max(max_val, A[offset + j]);
    }
    B[idx] = max_val;
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (batch + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Reduces the tensor by taking the minimum value along the last dimension.
  Future<Tensor> minReduction({int axis = -1}) async {
    if (axis != -1 && axis != shape.length - 1) {
      throw Exception(
          "Currently only min reduction along the last dimension is supported.");
    }
    int d = shape.last;
    int batch = size ~/ d;
    List<int> outShape = shape.sublist(0, shape.length - 1);
    Tensor result = await Tensor.create(outShape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx < ${batch}u) {
    let d: u32 = ${d}u;
    let offset: u32 = idx * d;
    var min_val: f32 = A[offset];
    for (var j: u32 = 1u; j < d; j = j + 1u) {
      min_val = min(min_val, A[offset + j]);
    }
    B[idx] = min_val;
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (batch + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Argmax: Finds the index of the maximum element along the last dimension.
  /// The resulting tensor has shape equal to the input shape minus the last dimension
  /// and always stores indices as f32 values.
  Future<Tensor> argmax({int axis = -1}) async {
    if (axis != -1 && axis != shape.length - 1) {
      throw Exception(
          "Currently only argmax along the last dimension is supported.");
    }
    int d = shape.last;
    int batch = size ~/ d;
    List<int> outShape = shape.sublist(0, shape.length - 1);
    Tensor result = await Tensor.create(outShape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx < ${batch}u) {
    let d: u32 = ${d}u;
    let offset: u32 = idx * d;
    var max_val: f32 = A[offset];
    var max_index: u32 = 0u;
    for (var j: u32 = 1u; j < d; j = j + 1u) {
      let val = A[offset + j];
      if (val > max_val) {
         max_val = val;
         max_index = j;
      }
    }
    B[idx] = f32(max_index);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (batch + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }
}

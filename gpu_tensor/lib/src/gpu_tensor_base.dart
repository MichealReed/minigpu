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
}

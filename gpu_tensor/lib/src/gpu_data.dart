import 'dart:typed_data';
import 'gpu_tensor_base.dart';

extension TensorSlicing on Tensor {
  /// Creates a new tensor by slicing the flattened tensor data.
  /// [start] is the starting flat index and [end] is the ending flat index (exclusive).
  Future<Tensor> sliceLinear({required int start, required int end}) async {
    if (start < 0 || end > size || start >= end) {
      throw Exception(
          "Invalid slice indices: start=$start, end=$end, size=$size.");
    }
    int newSize = end - start;
    // Create a temporary CPU buffer to hold the sliced data.
    final slicedData = Float32List(newSize);
    // Use the buffer's read method using the flat offset.
    await buffer.read(slicedData, newSize, readOffset: start);
    // Create and return a new tensor from the sliced data.
    // Here we assume a 1D shape for the sliced tensor.
    return await Tensor.create([newSize], data: slicedData, gpu: gpu);
  }

  /// Slices the tensor based on multi-dimensional indices.
  ///
  /// [startIndices] and [endIndices] specify the lower (inclusive) and upper (exclusive)
  /// bounds for each dimension. The function computes the flat offset and the total number
  /// of elements to transfer based on the tensor's shape.
  Future<Tensor> slice({
    required List<int> startIndices,
    required List<int> endIndices,
  }) async {
    if (startIndices.length != shape.length ||
        endIndices.length != shape.length) {
      throw Exception(
          "startIndices and endIndices must match tensor rank (${shape.length}).");
    }

    // Calculate strides for the tensor.
    List<int> strides = List.filled(shape.length, 1);
    for (int i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Compute the flat offset from multi-dimensional indices.
    int flatOffset = 0;
    for (int i = 0; i < shape.length; i++) {
      if (startIndices[i] < 0 ||
          endIndices[i] > shape[i] ||
          startIndices[i] >= endIndices[i]) {
        throw Exception(
            "Invalid slice indices for dimension $i: start=${startIndices[i]}, end=${endIndices[i]}, shape=${shape[i]}.");
      }
      flatOffset += startIndices[i] * strides[i];
    }

    // Compute the new shape and the total number of elements to read.
    List<int> newShape = [];
    int numElems = 1;
    for (int i = 0; i < shape.length; i++) {
      int dimSize = endIndices[i] - startIndices[i];
      newShape.add(dimSize);
      numElems *= dimSize;
    }

    // Read only the necessary portion from the GPU buffer.
    final slicedData = Float32List(numElems);
    await buffer.read(slicedData, numElems, readOffset: flatOffset);

    // Create and return the new tensor from the sliced data.
    return await Tensor.create(newShape, data: slicedData, gpu: gpu);
  }
}

import 'dart:typed_data';
import 'package:minigpu/minigpu.dart';

import 'gpu_tensor_base.dart';

extension TensorData on Tensor {
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

  /// Returns the value of the tensor element at the given [indices].
  /// Throws an exception if the [indices] length does not match the tensor rank
  /// or if any index is out of bounds.
  Future<double> getElement(List<int> indices) async {
    if (indices.length != shape.length) {
      throw Exception(
          "Indices length (${indices.length}) does not match tensor rank (${shape.length}).");
    }

    // Calculate strides for the tensor (assumes row-major order).
    List<int> strides = List.filled(shape.length, 1);
    for (int i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Compute the flat offset from multi-dimensional indices.
    int flatIndex = 0;
    for (int i = 0; i < shape.length; i++) {
      if (indices[i] < 0 || indices[i] >= shape[i]) {
        throw Exception(
            "Index out of bounds for dimension $i: ${indices[i]} not in [0, ${shape[i] - 1}].");
      }
      flatIndex += indices[i] * strides[i];
    }

    // Instead of pulling the entire tensor data, allocate a small
    // buffer to hold only the required single element.
    final elementData = Float32List(1);
    await buffer.read(elementData, 1, readOffset: flatIndex);
    return elementData[0];
  }

  /// Reshapes the tensor into a new shape without changing the underlying data.
  /// Throws an exception if the total number of elements would differ.
  Tensor reshape(List<int> newShape) {
    int newSize = newShape.reduce((a, b) => a * b);
    if (newSize != size) {
      throw Exception(
          "New shape $newShape does not match total number of elements $size");
    }
    return Tensor.fromBuffer(buffer, newShape);
  }

  /// Transposes a tensor according to the given [axes] permutation.
  /// If [axes] is omitted, the dimensions are reversed (default behavior).
  /// For example, for a tensor with shape [2,3,4]:
  ///   - transpose() produces a tensor with shape [4,3,2].
  ///   - transpose(axes: [1,0,2]) swaps the first two dimensions producing shape [3,2,4].
  Future<Tensor> transpose({List<int>? axes}) async {
    final int rank = shape.length;
    // Use reverse order if no permutation is provided.
    axes ??= List<int>.generate(rank, (i) => rank - i - 1);
    if (axes.length != rank) {
      throw Exception(
          "Axes length (${axes.length}) must equal tensor rank ($rank).");
    }
    // Validate that axes is a permutation of 0..rank-1.
    final sortedAxes = List<int>.from(axes)..sort();
    for (int i = 0; i < rank; i++) {
      if (sortedAxes[i] != i) {
        throw Exception("Invalid axes permutation: $axes.");
      }
    }

    // Compute the new shape.
    List<int> newShape = axes.map((i) => shape[i]).toList();
    // Compute total number of elements of the output.
    final int outSize = newShape.fold(1, (prod, e) => prod * e);

    // Compute input strides (row-major layout) for the original tensor.
    List<int> inputStrides = List.filled(rank, 1);
    for (int i = rank - 2; i >= 0; i--) {
      inputStrides[i] = inputStrides[i + 1] * shape[i + 1];
    }

    // Compute factors for unraveling an output flat index into indices in [newShape].
    // outFactors[i] = product(newShape[i+1:]) with outFactors[last] = 1.
    List<int> outFactors = List.filled(rank, 1);
    for (int i = 0; i < rank; i++) {
      int prod = 1;
      for (int j = i + 1; j < rank; j++) {
        prod *= newShape[j];
      }
      outFactors[i] = prod;
    }

    // Compute the inverse permutation.
    // For each original dimension d, find its new position: invPermutation[d] = j if axes[j] == d.
    List<int> invPermutation = List.filled(rank, 0);
    for (int j = 0; j < rank; j++) {
      invPermutation[axes[j]] = j;
    }

    // Helper to format integer lists as WGSL constant arrays.
    String formatArray(List<int> arr) => arr.map((x) => "${x}u").join(", ");

    final String shaderCode = '''
const outSize : u32 = ${outSize}u;
const rank : u32 = ${rank}u;
const outFactors : array<u32, $rank> = array<u32, $rank>(${formatArray(outFactors)});
const inputStrides : array<u32, $rank> = array<u32, $rank>(${formatArray(inputStrides)});
const invPermutation : array<u32, $rank> = array<u32, $rank>(${formatArray(invPermutation)});

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let i: u32 = global_id.x;
  if (i >= outSize) {
    return;
  }
  var remainder: u32 = i;
  var outIndices: array<u32, rank>;
  for (var j: u32 = 0u; j < rank; j = j + 1u) {
    outIndices[j] = remainder / outFactors[j];
    remainder = remainder % outFactors[j];
  }
  var inIndex: u32 = 0u;
  // Reconstruct the input index using the inverse permutation.
  for (var d: u32 = 0u; d < rank; d = d + 1u) {
    let pos = outIndices[invPermutation[d]];
    inIndex = inIndex + pos * inputStrides[d];
  }
  output[i] = input[inIndex];
}
''';

    Tensor result = await Tensor.create(newShape, gpu: gpu);
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer("input", buffer);
    shader.setBuffer("output", result.buffer);
    int wgCount = (outSize + 255) ~/ 256;
    await shader.dispatch(wgCount, 1, 1);
    shader.destroy();

    return result;
  }
}

extension TensorPrintHelpers on Tensor {
  /// Returns a string representation of the first [counts] elements along each dimension.
  /// For 2D tensors, head([r,c]) retrieves the first r rows and c columns using
  /// separate GPU reads for each row.
  Future<String> head(List<int> counts) async {
    if (counts.length != shape.length) {
      throw Exception(
          "Counts length (${counts.length}) does not match tensor rank (${shape.length}).");
    }
    if (shape.length == 2) {
      int numRows = counts[0];
      int numCols = counts[1];
      int totalCols = shape[1];
      List<List<double>> rows = [];
      for (int r = 0; r < numRows; r++) {
        // Each row is stored contiguously.
        final rowData = Float32List(numCols);
        // Compute the flat offset: row start = r*totalCols.
        await buffer.read(rowData, numCols, readOffset: r * totalCols);
        rows.add(rowData);
      }
      return _format2D(rows);
    } else {
      // Fallback for tensor rank != 2, use our existing slice.
      List<int> startIndices = List.filled(shape.length, 0);
      Tensor subTensor =
          await slice(startIndices: startIndices, endIndices: counts);
      List<double> subData = await subTensor.getData();
      return _formatTensor(subData, counts);
    }
  }

  /// Returns a string representation of the last [counts] elements along each dimension.
  /// For 2D tensors, tail([r,c]) retrieves the last r rows and last c columns using
  /// separate GPU reads for each row.
  Future<String> tail(List<int> counts) async {
    if (counts.length != shape.length) {
      throw Exception(
          "Counts length (${counts.length}) does not match tensor rank (${shape.length}).");
    }
    if (shape.length == 2) {
      int numRows = counts[0];
      int numCols = counts[1];
      int totalCols = shape[1];
      int startRow = shape[0] - numRows;
      int startCol = totalCols - numCols;
      List<List<double>> rows = [];
      for (int r = startRow; r < shape[0]; r++) {
        final rowData = Float32List(numCols);
        // Each row starts at r*totalCols and we read starting from the last numCols.
        await buffer.read(rowData, numCols,
            readOffset: r * totalCols + startCol);
        rows.add(rowData);
      }
      return _format2D(rows);
    } else {
      // Fallback for tensor rank != 2 using our slice() methods.
      List<int> startIndices = [];
      for (int i = 0; i < shape.length; i++) {
        if (counts[i] > shape[i]) {
          throw Exception(
              "Tail count (${counts[i]}) exceeds tensor dimension $i size (${shape[i]}).");
        }
        startIndices.add(shape[i] - counts[i]);
      }
      Tensor subTensor = await slice(
          startIndices: startIndices,
          endIndices: shape
              .asMap()
              .entries
              .map((e) => e.value)
              .toList()); // endIndices equals the full dimensions.
      subTensor = await subTensor.slice(
        startIndices: List.filled(counts.length, 0),
        endIndices: counts,
      );
      List<double> subData = await subTensor.getData();
      return _formatTensor(subData, counts);
    }
  }

  /// Helper for formatting a 2D array.
  String _format2D(List<List<double>> rows) {
    List<String> rowStrings = rows.map((row) => row.toString()).toList();
    return "[${rowStrings.join(", ")}]";
  }

  /// Helper that formats a flat [data] list into a nested array string given the [newShape].
  String _formatTensor(List<double> data, List<int> newShape) {
    if (newShape.length == 1) {
      return data.toString();
    } else {
      int subTensorSize = newShape.sublist(1).reduce((a, b) => a * b);
      List<String> parts = [];
      for (int i = 0; i < newShape[0]; i++) {
        int startIdx = i * subTensorSize;
        int endIdx = startIdx + subTensorSize;
        List<double> subData = data.sublist(startIdx, endIdx);
        String subStr = _formatTensor(subData, newShape.sublist(1));
        parts.add(subStr);
      }
      return "[${parts.join(", ")}]";
    }
  }
}

import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

void main() {
  group('Tensor slicing tests', () {
    test('sliceLinear slices a 1D tensor correctly', () async {
      // Create a 1D tensor with 10 elements: 0, 1, ..., 9.
      final originalData =
          Float32List.fromList(List<double>.generate(10, (i) => i.toDouble()));
      // Create the tensor (passing null for gpu for testing purposes).
      Tensor tensor = await Tensor.create([10], data: originalData);

      // Slice from flat index 2 (inclusive) to 7 (exclusive).
      Tensor sliced = await tensor.sliceLinear(start: 2, end: 7);

      final result = await sliced.getData();
      expect(result, equals([2.0, 3.0, 4.0, 5.0, 6.0]));
      expect(sliced.shape, equals([5]));
    });

    test('slice slices a 2D tensor correctly', () async {
      // Create a 2D tensor with shape [2, 3]:
      // [0, 1, 2]
      // [3, 4, 5]
      final originalData = Float32List.fromList([0, 1, 2, 3, 4, 5]);
      Tensor tensor = await Tensor.create([2, 3], data: originalData);

      // Slice: select the second row (index 1) for all columns.
      // startIndices corresponds to [row, col] = [1, 0]
      // endIndices corresponds to [row, col] = [2, 3]
      Tensor sliced = await tensor.slice(
        startIndices: [1, 0],
        endIndices: [2, 3],
      );

      final result = await sliced.getData();
      // Expected shape is [1, 3] with data from the second row.
      expect(sliced.shape, equals([1, 3]));
      expect(result, equals([3.0, 4.0, 5.0]));
    });

    test('slice throws error on invalid multidimensional indices', () async {
      // Create a 2D tensor with shape [2, 3].
      final originalData =
          Float32List.fromList(List<double>.generate(6, (i) => i.toDouble()));
      Tensor tensor = await Tensor.create([2, 3], data: originalData);

      // Provide invalid slice indices (e.g. end index equals start index for a dimension).
      expect(
        () async => await tensor.slice(
          startIndices: [0, 2],
          endIndices: [1, 2],
        ),
        throwsException,
      );
    });

    // Additional tests for higher dimensions.
    group('Multidimensional slicing tests', () {
      test('slice a 3D tensor correctly', () async {
        // Create a 3D tensor with shape [2, 2, 3] (total 12 elements).
        // Data arranged in row-major order:
        // For dimension 0 = 0, sub-tensor:
        //   [ [0, 1, 2],
        //     [3, 4, 5] ]
        // For dimension 0 = 1, sub-tensor:
        //   [ [6, 7, 8],
        //     [9, 10, 11] ]
        final originalData = Float32List.fromList(
            List<double>.generate(12, (i) => i.toDouble()));
        Tensor tensor = await Tensor.create([2, 2, 3], data: originalData);

        // Slice: select the first "plane" (dimension 0 = 0) completely.
        // New shape will be [1, 2, 3] and expected data [0, 1, 2, 3, 4, 5].
        Tensor sliced = await tensor.slice(
          startIndices: [0, 0, 0],
          endIndices: [1, 2, 3],
        );

        final result = await sliced.getData();
        expect(sliced.shape, equals([1, 2, 3]));
        expect(result, equals([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
      });

      test('slice a 4D tensor correctly', () async {
        // Create a 4D tensor with shape [2, 2, 2, 2] (16 elements).
        // Fill with values 0...15 in row-major order.
        final originalData = Float32List.fromList(
            List<double>.generate(16, (i) => i.toDouble()));
        Tensor tensor = await Tensor.create([2, 2, 2, 2], data: originalData);

        // select element at indices [0, 0, :, :].
        // for dimension 0 & 1 fixed at 0, and complete slice for dimensions 2 and 3.
        // New shape: [1, 1, 2, 2] (4 elements).
        // Expected data from the original tensor:
        // [ [ [ [0, 1], [2, 3] ] ] ]
        Tensor sliced = await tensor.slice(
          startIndices: [0, 0, 0, 0],
          endIndices: [1, 1, 2, 2],
        );

        final result = await sliced.getData();
        expect(sliced.shape, equals([1, 1, 2, 2]));
        expect(result, equals([0.0, 1.0, 2.0, 3.0]));
      });
    });
  });
}

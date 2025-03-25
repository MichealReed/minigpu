import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

void main() {
  group('Tensor slicing tests', () {
    group('Tensor element access and print helpers', () {
      test('head returns correct representation for a 2D tensor', () async {
        // Create a 3x3 tensor:
        // [0, 1, 2]
        // [3, 4, 5]
        // [6, 7, 8]
        final data =
            Float32List.fromList(List<double>.generate(9, (i) => i.toDouble()));
        Tensor tensor = await Tensor.create([3, 3], data: data);
        // Get the first 2 rows and 2 columns.
        String headStr = await tensor.head([2, 2]);
        // Expected formatted string: "[[0.0, 1.0], [3.0, 4.0]]"
        expect(headStr, equals("[[0.0, 1.0], [3.0, 4.0]]"));
      });

      test('tail returns correct representation for a 2D tensor', () async {
        // Create a 3x3 tensor:
        // [0, 1, 2]
        // [3, 4, 5]
        // [6, 7, 8]
        final data =
            Float32List.fromList(List<double>.generate(9, (i) => i.toDouble()));
        Tensor tensor = await Tensor.create([3, 3], data: data);
        // Get the last 2 rows and 2 columns.
        String tailStr = await tensor.tail([2, 2]);
        // Expected formatted string: "[[4.0, 5.0], [7.0, 8.0]]"
        expect(tailStr, equals("[[4.0, 5.0], [7.0, 8.0]]"));
      });

      test('head returns correct representation for a 4D tensor', () async {
        // 4D tensor with shape [2,2,2,2] with data 0..15.
        final data = Float32List.fromList(
            List<double>.generate(16, (i) => i.toDouble()));
        Tensor tensor = await Tensor.create([2, 2, 2, 2], data: data);
        // Use head() with counts [1,1,2,2]. This extracts the sub-tensor
        // at indices [0,0,:,:]:
        // Expected sub-tensor (shape [1,1,2,2]) from tensor[0,0,:,:]:
        // Element indices: [0,0,0,0]=0, [0,0,0,1]=1, [0,0,1,0]=2, [0,0,1,1]=3.
        // Formatted as: "[[[0.0, 1.0], [2.0, 3.0]]]"
        String headStr = await tensor.head([1, 1, 2, 2]);
        expect(headStr, equals("[[[[0.0, 1.0], [2.0, 3.0]]]]"));
      });

      test('tail returns correct representation for a 4D tensor', () async {
        // 4D tensor with shape [2,2,2,2] with data 0..15.
        final data = Float32List.fromList(
            List<double>.generate(16, (i) => i.toDouble()));
        Tensor tensor = await Tensor.create([2, 2, 2, 2], data: data);
        // Use tail() with counts [1,1,2,2]. Fallback tail computes
        // startIndices = [2-1, 2-1, 2-2, 2-2] = [1,1,0,0],
        // then slices to get sub-tensor of shape [1,1,2,2] from tensor[1,1,:,:]:
        // Elements at tensor[1,1,0,0]=12, [1,1,0,1]=13, [1,1,1,0]=14, [1,1,1,1]=15.
        // Expected formatted output: "[[[12.0, 13.0], [14.0, 15.0]]]"
        String tailStr = await tensor.tail([1, 1, 2, 2]);
        expect(tailStr, equals("[[[[12.0, 13.0], [14.0, 15.0]]]]"));
      });
    });
  });
}

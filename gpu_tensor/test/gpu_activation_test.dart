import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

Future<void> main() async {
  group('Activation Tests', () {
    test('ReLU activation', () async {
      // Create a tensor with negative, zero, and positive values.
      List<int> shape = [4];
      Float32List inputData = Float32List.fromList([-1.0, 0.0, 1.0, 2.0]);

      Tensor tensor = await Tensor.create(shape, data: inputData);

      // Apply ReLU activation.
      Tensor reluTensor = await tensor.relu();
      Float32List result = await reluTensor.getData();

      // Expected values: negative values become 0.
      List<double> expected = [0.0, 0.0, 1.0, 2.0];
      expect(result.length, equals(expected.length));
      for (int i = 0; i < result.length; i++) {
        expect(result[i], closeTo(expected[i], 1e-5));
      }
    });

    test('Sigmoid activation', () async {
      // Create a tensor with a mix of negative and positive values.
      List<int> shape = [4];
      Float32List inputData = Float32List.fromList([-1.0, 0.0, 1.0, 2.0]);

      Tensor tensor = await Tensor.create(shape, data: inputData);

      // Apply Sigmoid activation.
      Tensor sigmoidTensor = await tensor.sigmoid();
      Float32List result = await sigmoidTensor.getData();

      // Expected results computed as: 1/(1+exp(-x))
      // sigmoid(-1) ~ 0.26894, sigmoid(0) ~ 0.5, sigmoid(1) ~ 0.73106, sigmoid(2) ~ 0.88080
      List<double> expected = [0.26894, 0.5, 0.73106, 0.88080];
      expect(result.length, equals(expected.length));
      for (int i = 0; i < result.length; i++) {
        expect(result[i], closeTo(expected[i], 1e-4));
      }
    });
  });
}

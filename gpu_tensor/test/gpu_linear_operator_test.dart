import 'dart:typed_data';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

Future<void> main() async {
  group('Tensor MatMul tests', () {
    test('Matrix multiplication (matMul) for rank-2 tensors', () async {
      var aData = Float32List.fromList([1, 2, 3, 4]);
      var bData = Float32List.fromList([5, 6, 7, 8]);
      var tensorA = await Tensor.create([2, 2], data: aData);
      var tensorB = await Tensor.create([2, 2], data: bData);
      var result = await tensorA.matMul(tensorB);
      var resultData = await result.getData();
      expect(result.shape, equals([2, 2]));
      expect(resultData, equals(Float32List.fromList([19, 22, 43, 50])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('Batched matrix multiplication (matMul) for higher-rank tensors',
        () async {
      var aData = Float32List.fromList([
        1, 2, 3, 4, 5, 6, // Batch 0: 2x3
        2, 3, 4, 5, 6, 7, // Batch 1: 2x3
      ]);
      var bData = Float32List.fromList([
        7, 8, 9, 10, 11, 12, // Batch 0: 3x2
        8, 9, 10, 11, 12, 13, // Batch 1: 3x2
      ]);
      var tensorA = await Tensor.create([2, 2, 3], data: aData);
      var tensorB = await Tensor.create([2, 3, 2], data: bData);
      var result = await tensorA.matMul(tensorB);
      var resultData = await result.getData();
      expect(result.shape, equals([2, 2, 2]));
      expect(
          resultData,
          equals(Float32List.fromList([
            58, 64, 139, 154, // Batch 0
            94, 103, 184, 202 // Batch 1
          ])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('Matrix multiplication with incompatible shapes throws exception',
        () async {
      var tensorA = await Tensor.create([2, 2]);
      var tensorB = await Tensor.create([3, 3]);
      expect(() => tensorA.matMul(tensorB), throwsException);
      tensorA.destroy();
      tensorB.destroy();
    });
  });

  group('Tensor convolution tests', () {
    test('conv2d returns correct output for 3x3 image and 2x2 kernel',
        () async {
      // Create a 3x3 input image:
      // [1, 2, 3
      //  4, 5, 6
      //  7, 8, 9]
      var imageShape = [3, 3];
      var imageData = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8, 9]);
      var imageTensor = await Tensor.create(imageShape, data: imageData);

      // Create a 2x2 kernel:
      // [1, 0
      //  0,-1]
      var kernelShape = [2, 2];
      var kernelData = Float32List.fromList([1, 0, 0, -1]);
      var kernelTensor = await Tensor.create(kernelShape, data: kernelData);

      // Expected output shape: [3-2+1, 3-2+1] => [2, 2]
      // Calculations:
      // Top-left = 1*1 + 2*0 + 4*0 + 5*(-1) = 1 - 5 = -4
      // Top-right = 2*1 + 3*0 + 5*0 + 6*(-1) = 2 - 6 = -4
      // Bottom-left = 4*1 + 5*0 + 7*0 + 8*(-1) = 4 - 8 = -4
      // Bottom-right = 5*1 + 6*0 + 8*0 + 9*(-1) = 5 - 9 = -4
      var expectedOutput = Float32List.fromList([-4, -4, -4, -4]);

      // Perform convolution.
      var outputTensor = await imageTensor.conv2d(kernelTensor);
      var outputData = await outputTensor.getData();
      expect(outputTensor.shape, equals([2, 2]));
      expect(outputData, equals(expectedOutput));

      imageTensor.destroy();
      kernelTensor.destroy();
      outputTensor.destroy();
    });

    test('conv2d with non 2D input or kernel throws exception', () async {
      // Create a 1D tensor as input.
      var badInput =
          await Tensor.create([4], data: Float32List.fromList([1, 2, 3, 4]));
      // Create a valid kernel 2x2.
      var kernelTensor = await Tensor.create([2, 2],
          data: Float32List.fromList([1, 0, 0, -1]));
      expect(() => badInput.conv2d(kernelTensor), throwsException);

      badInput.destroy();
      kernelTensor.destroy();

      // Now, create a valid image but a bad kernel.
      var imageTensor = await Tensor.create([4, 4], data: Float32List(16));
      var badKernel =
          await Tensor.create([3], data: Float32List.fromList([1, 2, 3]));
      expect(() => imageTensor.conv2d(badKernel), throwsException);

      imageTensor.destroy();
      badKernel.destroy();
    });
  });
}

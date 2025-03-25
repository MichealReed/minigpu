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
      // Single-channel 2D convolution test.
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

      // Expected output shape: [2, 2]
      // Calculations:
      // Top-left: 1*1 + 2*0 + 4*0 + 5*(-1) = 1 - 5 = -4
      // Top-right: 2*1 + 3*0 + 5*0 + 6*(-1) = 2 - 6 = -4
      // Bottom-left: 4*1 + 5*0 + 7*0 + 8*(-1) = 4 - 8 = -4
      // Bottom-right: 5*1 + 6*0 + 8*0 + 9*(-1) = 5 - 9 = -4
      var expectedOutput = Float32List.fromList([-4, -4, -4, -4]);

      // Use the original conv2d (single-channel) path.
      var outputTensor = await imageTensor.conv2d(kernelTensor);
      var outputData = await outputTensor.getData();
      expect(outputTensor.shape, equals([2, 2]));
      expect(outputData, equals(expectedOutput));

      imageTensor.destroy();
      kernelTensor.destroy();
      outputTensor.destroy();
    });

    test(
        'conv with multi-channel input returns correct output with dilation, stride and padding',
        () async {
      // Multi-channel convolution test.
      //
      // Input tensor shape: [H, W, Cin] where H = 3, W = 3, Cin = 2.
      // We use two channels per pixel.
      //
      // Kernel shape: [2, 2, Cin, Cout] with Cin = 2 and Cout = 1.
      //
      // define two channels:
      //
      // Channel 0 (first 3x3 matrix):
      // [ 1,  2,  3,
      //   4,  5,  6,
      //   7,  8,  9]
      //
      // Channel 1 (second 3x3 matrix):
      // [ 9,  8,  7,
      //   6,  5,  4,
      //   3,  2,  1]
      //
      // Use kernel values of 1 for every element.
      //
      // With stride=1, no padding, dilation=1:
      // Output dimensions: [3-2+1, 3-2+1] = [2, 2, 1].
      //
      // Calculation for top-left output (position (0,0)):
      // Channel 0 window: [1, 2; 4, 5] => 1+2+4+5 = 12.
      // Channel 1 window: [9, 8; 6, 5] => 9+8+6+5 = 28.
      // Total = 12 + 28 = 40.

      var imageShape = [3, 3, 2];
      var imageData = Float32List.fromList([
        // Channel 0:
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        // Channel 1:
        9, 8, 7,
        6, 5, 4,
        3, 2, 1,
      ]);
      var imageTensor = await Tensor.create(imageShape, data: imageData);

      var kernelShape = [2, 2, 2, 1];
      // All ones kernel.
      var kernelData = Float32List(2 * 2 * 2 * 1);
      for (int i = 0; i < kernelData.length; i++) {
        kernelData[i] = 1;
      }
      var kernelTensor = await Tensor.create(kernelShape, data: kernelData);

      // Convolution parameters:
      // stride = 1, pad = 0, dilation = 1.
      var outputTensor = await imageTensor.conv(
        kernel: kernelTensor,
        strideH: 1,
        strideW: 1,
        padH: 0,
        padW: 0,
        dilationH: 1,
        dilationW: 1,
      );
      // Expected output shape: [2, 2, 1].
      // Each output pixel should be 40 as computed above.
      var expectedOutput = Float32List.fromList([40, 40, 40, 40]);

      var outputData = await outputTensor.getData();
      expect(outputTensor.shape, equals([2, 2, 1]));
      expect(outputData, equals(expectedOutput));

      imageTensor.destroy();
      kernelTensor.destroy();
      outputTensor.destroy();
    });

    test('conv with channel mismatch throws exception', () async {
      // Create a multi-channel input with 3 channels.
      var imageTensor =
          await Tensor.create([4, 4, 3], data: Float32List(4 * 4 * 3));
      // Create a kernel expecting 2 channels instead of 3.
      var kernelTensor =
          await Tensor.create([3, 3, 2, 1], data: Float32List(3 * 3 * 2 * 1));
      expect(() => imageTensor.conv(kernel: kernelTensor), throwsException);

      imageTensor.destroy();
      kernelTensor.destroy();
    });
  });
}

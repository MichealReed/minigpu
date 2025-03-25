import 'dart:typed_data';
import 'package:gpu_tensor/gpu_tensor.dart';

Future<void> main() async {
  // ---------------------------
  // Basic tensor arithmetic demo:
  // ---------------------------
  // Create a 3x3 tensor (Tensor A) with values 1..9.
  final a = await Tensor.create(
    [3, 3],
    data: Float32List.fromList([
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
    ]),
  );

  // Create another 3x3 tensor (Tensor B) with values 9..1.
  final b = await Tensor.create(
    [3, 3],
    data: Float32List.fromList([
      9,
      8,
      7,
      6,
      5,
      4,
      3,
      2,
      1,
    ]),
  );

  // Elementwise addition.
  final added = await (a + b);
  final addedData = await added.getData();
  print("Elementwise addition result: $addedData");

  // Elementwise subtraction.
  final subtracted = await (a - b);
  final subtractedData = await subtracted.getData();
  print("Elementwise subtraction result: $subtractedData");

  // Scalar multiplication.
  final scalarMult = await a.multiplyScalar(2.0);
  final scalarMultData = await scalarMult.getData();
  print("Scalar multiplication result: $scalarMultData");

  // Matrix multiplication (dot product) of a and b.
  final matMulResult = await a.matMul(b);
  final matMulData = await matMulResult.getData();
  print("Matrix multiplication result: $matMulData");

  // Reshape the matrix multiplication result into a 1-D tensor.
  final reshaped = matMulResult.reshape([9]);
  final reshapedData = await reshaped.getData();
  print("Reshaped matrix multiplication result: $reshapedData");

  // FFT demo (1D FFT)
  // Create a real 1D tensor with 8 points.
  // The fft1d() function detects that the tensor is real (shape [8])
  // before performing the FFT.
  const int points1D = 8;
  final Float32List realData1D = Float32List(points1D);
  for (int i = 0; i < points1D; i++) {
    realData1D[i] = i.toDouble();
  }
  final realTensor1D = await Tensor.create([points1D], data: realData1D);
  final fft1dResult = await realTensor1D.fft1d();
  final fft1dResultData = await fft1dResult.getData();
  print("FFT 1D result: $fft1dResultData");

  // FFT demo (2D FFT)
  // Create a real 2D tensor with 4 rows and 4 columns.
  // The fft2d() method detects its shape [4, 4] and internally converts it
  // to a complex tensor of shape [4, 4, 2] with imaginary parts set to zero.
  const int rows = 4;
  const int cols = 4;
  final Float32List realData2D = Float32List(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    realData2D[i] = i.toDouble();
  }
  final realTensor2D = await Tensor.create([rows, cols], data: realData2D);
  final fft2dResult = await realTensor2D.fft2d();
  final fft2dResultData = await fft2dResult.getData();
  print("FFT 2D result: $fft2dResultData");

  // Cleanup: explicitly destroy all tensors to free GPU resources.
  a.destroy();
  b.destroy();
  added.destroy();
  subtracted.destroy();
  scalarMult.destroy();
  matMulResult.destroy();
  reshaped.destroy();
  fft1dResult.destroy();
  fft2dResult.destroy();
}

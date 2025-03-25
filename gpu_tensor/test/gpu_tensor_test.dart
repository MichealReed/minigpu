import 'dart:typed_data';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

Future<void> main() async {
  group('Tensor tests', () {
    test('Tensor creation with zeros', () async {
      var shape = [2, 3];
      var tensor = await Tensor.create(shape);
      var data = await tensor.getData();
      expect(data.length, equals(6));
      expect(data.every((v) => v == 0), isTrue);
      tensor.destroy();
    });

    test('Tensor creation with initial data', () async {
      var shape = [3];
      var initialData = Float32List.fromList([1, 2, 3]);
      var tensor = await Tensor.create(shape, data: initialData);
      var data = await tensor.getData();
      expect(data, equals(initialData));
      tensor.destroy();
    });

    test('Tensor elementwise addition', () async {
      var shape = [4];
      var aData = Float32List.fromList([1, 2, 3, 4]);
      var bData = Float32List.fromList([4, 3, 2, 1]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await tensorA.add(tensorB);
      var resultData = await result.getData();
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
      expect(resultData, equals(Float32List.fromList([5, 5, 5, 5])));
    });

    test('Tensor multiply scalar', () async {
      var shape = [4];
      var dataIn = Float32List.fromList([1, 2, 3, 4]);
      var tensor = await Tensor.create(shape, data: dataIn);
      var result = await tensor.multiplyScalar(2);
      var dataOut = await result.getData();
      expect(dataOut, equals(Float32List.fromList([2, 4, 6, 8])));
      tensor.destroy();
    });

    test('Tensor reshape carrying same data buffer', () async {
      var shape = [2, 2];
      var initialData = Float32List.fromList([1, 2, 3, 4]);
      var tensor = await Tensor.create(shape, data: initialData);
      var reshaped = tensor.reshape([4]);
      expect(reshaped.shape, equals([4]));
      // Ensure data remains the same after reshape.
      var data = await reshaped.getData();
      expect(data, equals(initialData));
      tensor.destroy();
    });

    test('Tensor reshape with mismatched size throws exception', () async {
      var shape = [2, 2];
      var tensor = await Tensor.create(shape);
      expect(() => tensor.reshape([3]), throwsException);
      tensor.destroy();
    });

    test('Tensor elementwise subtraction', () async {
      var shape = [4];
      var aData = Float32List.fromList([4, 4, 4, 4]);
      var bData = Float32List.fromList([1, 1, 1, 1]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await tensorA.subtract(tensorB);
      var resultData = await result.getData();
      expect(resultData, equals(Float32List.fromList([3, 3, 3, 3])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('Tensor elementwise multiplication', () async {
      var shape = [4];
      var aData = Float32List.fromList([1, 2, 3, 4]);
      var bData = Float32List.fromList([2, 3, 4, 5]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await tensorA.multiply(tensorB);
      var resultData = await result.getData();
      expect(resultData, equals(Float32List.fromList([2, 6, 12, 20])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('Operator overloads for +, -, and *', () async {
      var shape = [3];
      var aData = Float32List.fromList([1, 2, 3]);
      var bData = Float32List.fromList([4, 5, 6]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);

      // Test operator +
      var addResult = await (tensorA + tensorB);
      var addData = await addResult.getData();
      expect(addData, equals(Float32List.fromList([5, 7, 9])));

      // Test operator -
      var subResult = await (tensorB - tensorA);
      var subData = await subResult.getData();
      expect(subData, equals(Float32List.fromList([3, 3, 3])));

      // Test operator * with scalar
      var scalarResult = await (tensorA * 3);
      var scalarData = await scalarResult.getData();
      expect(scalarData, equals(Float32List.fromList([3, 6, 9])));

      // Test operator * with tensor (elementwise multiplication)
      var multResult = await (tensorA * tensorB);
      var multData = await multResult.getData();
      expect(multData, equals(Float32List.fromList([4, 10, 18])));
      tensorA.destroy();
      tensorB.destroy();
      addResult.destroy();
      subResult.destroy();
      scalarResult.destroy();
      multResult.destroy();
    });

    test('Matrix multiplication (matMul) for rank-2 tensors', () async {
      // Matrix A: 2x2 => [1, 2, 3, 4]
      // Matrix B: 2x2 => [5, 6, 7, 8]
      // Expected result: [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] => [19, 22, 43, 50]
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

    test(
      'Batched matrix multiplication (matMul) for higher-rank tensors',
      () async {
        // In this test we use batched matrices.
        // Tensor A with shape [2, 2, 3] represents 2 batches of 2x3 matrices.
        // Tensor B with shape [2, 3, 2] represents 2 batches of 3x2 matrices.
        // Expected result shape: [2, 2, 2]
        //
        // Batch 0:
        // A0 = [ [1, 2, 3],
        //        [4, 5, 6] ]
        // B0 = [ [7,  8],
        //        [9, 10],
        //        [11,12] ]
        // C0 = A0*B0 = [ [1*7+2*9+3*11, 1*8+2*10+3*12]  => [58, 64],
        //                [4*7+5*9+6*11, 4*8+5*10+6*12]  => [139,154] ]
        //
        // Batch 1:
        // A1 = [ [2, 3, 4],
        //        [5, 6, 7] ]
        // B1 = [ [8,  9],
        //        [10,11],
        //        [12,13] ]
        // C1 = A1*B1 = [ [2*8+3*10+4*12, 2*9+3*11+4*13] => [94,103],
        //                [5*8+6*10+7*12, 5*9+6*11+7*13] => [184,202] ]

        var aData = Float32List.fromList([
          // Batch 0: 2x3
          1, 2, 3, 4, 5, 6,
          // Batch 1: 2x3
          2, 3, 4, 5, 6, 7,
        ]);

        var bData = Float32List.fromList([
          // Batch 0: 3x2
          7, 8, 9, 10, 11, 12,
          // Batch 1: 3x2
          8, 9, 10, 11, 12, 13,
        ]);

        var tensorA = await Tensor.create([2, 2, 3], data: aData);
        var tensorB = await Tensor.create([2, 3, 2], data: bData);
        var result = await tensorA.matMul(tensorB);
        var resultData = await result.getData();

        expect(result.shape, equals([2, 2, 2]));
        expect(
          resultData,
          equals(
            Float32List.fromList([
              58, 64, 139, 154, // Batch 0 (flattened row-major)
              94, 103, 184, 202, // Batch 1 (flattened row-major)
            ]),
          ),
        );
        tensorA.destroy();
        tensorB.destroy();
        result.destroy();
      },
    );

    test('Batched matrix multiplication (matMul) for rank-4 tensors', () async {
      // We create two 4D tensors:
      // Tensor A with shape [2, 3, 2, 3] represents 2x3 batches of 2x3 matrices.
      // Tensor B with shape [2, 3, 3, 4] represents 2x3 batches of 3x4 matrices.
      //
      // For each batch, matrix multiplication:
      // A (2x3): [1,2,3,
      //           4,5,6]
      // B (3x4): [7,8,9,10,
      //           11,12,13,14,
      //           15,16,17,18]
      //
      // Expected result for each batch (2x4):
      // [74, 80, 86, 92,
      //  173,188,203,218]
      //
      // Since we have 2 * 3 = 6 batches, the final output shape should be [2, 3, 2, 4]
      // and the flattened result will be 6 copies of the expected batch.

      final aMatrix = <double>[
        1,
        2,
        3,
        4,
        5,
        6,
      ]; // aMatrix for each batch (2x3)
      final bMatrix = <double>[
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
      ]; // bMatrix for each batch (3x4)

      // Tensor A: 6 batches of aMatrix (each 6 elements) => total 36 elements.
      final List<double> aDataList = [];
      for (int i = 0; i < 6; i++) {
        aDataList.addAll(aMatrix);
      }
      final aData = Float32List.fromList(aDataList);

      // Tensor B: 6 batches of bMatrix (each 12 elements) => total 72 elements.
      final List<double> bDataList = [];
      for (int i = 0; i < 6; i++) {
        bDataList.addAll(bMatrix);
      }
      final bData = Float32List.fromList(bDataList);

      var tensorA = await Tensor.create([2, 3, 2, 3], data: aData);
      var tensorB = await Tensor.create([2, 3, 3, 4], data: bData);
      var result = await tensorA.matMul(tensorB);
      var resultData = await result.getData();

      final expectedBatch = <double>[74, 80, 86, 92, 173, 188, 203, 218];
      final List<double> expectedDataList = [];
      for (int i = 0; i < 6; i++) {
        expectedDataList.addAll(expectedBatch);
      }
      final expectedData = Float32List.fromList(expectedDataList);

      expect(result.shape, equals([2, 3, 2, 4]));
      expect(resultData, equals(expectedData));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test(
      'Matrix multiplication with incompatible shapes throws exception',
      () async {
        var tensorA = await Tensor.create([2, 2]);
        var tensorB = await Tensor.create([3, 3]);
        expect(() => tensorA.matMul(tensorB), throwsException);
        tensorA.destroy();
        tensorB.destroy();
      },
    );
  });
}

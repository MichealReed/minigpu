import 'dart:typed_data';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor maxPool tests', () {
    test('maxPool returns correct output for 2D tensor', () async {
      // Create a 4x4 tensor with values arranged row-major:
      // [ 1,  2,  3,  4,
      //   5,  6,  7,  8,
      //   9, 10, 11, 12,
      //  13, 14, 15, 16 ]
      var data = Float32List.fromList([
        1,
        2,
        3,
        4,
        5,
        6,
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
      ]);
      var tensor = await Tensor.create([4, 4], data: data);
      // For a rank-2 tensor, pooling is applied on both dimensions.
      // PoolSizes: [2,2] with stride [2,2] and no padding.
      // Expected output:
      // Block (0,0): [1,2,5,6] -> max is 6.
      // Block (0,1): [3,4,7,8] -> max is 8.
      // Block (1,0): [9,10,13,14] -> max is 14.
      // Block (1,1): [11,12,15,16] -> max is 16.
      var pooled = await tensor
          .maxPool(poolSizes: [2, 2], strides: [2, 2], pads: [0, 0]);
      var resultData = await pooled.getData();
      expect(pooled.shape, equals([2, 2]));
      expect(resultData, equals(Float32List.fromList([6, 8, 14, 16])));
      tensor.destroy();
      pooled.destroy();
    });

    test('maxPool returns correct output for 3D tensor with preserved channels',
        () async {
      // Create a 4x4 tensor with 2 channels.
      // For channel 0 the values are 1..16;
      // For channel 1 the values are 101..116.
      var channel0 = <double>[
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
      ];
      var channel1 = channel0.map((v) => v + 100).toList();
      var interleaved = <double>[];
      for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
          interleaved.add(channel0[r * 4 + c]);
          interleaved.add(channel1[r * 4 + c]);
        }
      }
      var data = Float32List.fromList(interleaved);
      // The tensor is assumed to be in channel‑last ordering.
      // In this interpretation, for a [4,4,2] tensor, index 0 (height) and index 1 (width)
      // are spatial dimensions and pooling should be applied over them.
      // The expected output shape is then:
      // [outHeight, outWidth, channels] where:
      //     outHeight = ((4 - 2)/2)+1 = 2,
      //     outWidth  = ((4 - 2)/2)+1 = 2,
      // and channels remain 2.
      // For channel 0, the 2x2 pooled blocks are:
      //  Block (0,0): [1,2,5,6] -> max is 6.
      //  Block (0,1): [3,4,7,8] -> max is 8.
      //  Block (1,0): [9,10,13,14] -> max is 14.
      //  Block (1,1): [11,12,15,16] -> max is 16.
      // Similarly for channel 1: [106,108,114,116]
      var tensor = await Tensor.create([4, 4, 2], data: data);
      var pooled = await tensor
          .maxPool(poolSizes: [2, 2], strides: [2, 2], pads: [0, 0]);
      var resultData = await pooled.getData();
      expect(pooled.shape, equals([2, 2, 2]));
      expect(
          resultData,
          equals(Float32List.fromList([
            6,
            106,
            8,
            108,
            14,
            114,
            16,
            116,
          ])));
      tensor.destroy();
      pooled.destroy();
    });

    test(
        'maxPool with pool dimensions greater than tensor rank throws exception',
        () async {
      // Create a rank-2 tensor.
      var tensor = await Tensor.create([4, 4], data: Float32List(16));
      // Request poolSizes for 3 dimensions on a rank-2 tensor should throw.
      expect(() => tensor.maxPool(poolSizes: [2, 2, 2], strides: [1, 1, 1]),
          throwsException);
      tensor.destroy();
    });
  });

  group('Tensor minPool tests', () {
    test('minPool returns correct output for 2D tensor', () async {
      var data = Float32List.fromList([
        16,
        12,
        8,
        4,
        14,
        10,
        6,
        2,
        12,
        8,
        4,
        0,
        10,
        6,
        2,
        -2,
      ]);
      var tensor = await Tensor.create([4, 4], data: data);

      // For a 2x2 pooling window with stride 2:
      // Block (0,0): [16,12,14,10] -> min is 10
      // Block (0,1): [8,4,6,2] -> min is 2
      // Block (1,0): [12,8,10,6] -> min is 6
      // Block (1,1): [4,0,2,-2] -> min is -2
      var pooled = await tensor.minPool(poolSizes: [2, 2], strides: [2, 2]);
      var resultData = await pooled.getData();
      expect(pooled.shape, equals([2, 2]));
      expect(resultData, equals(Float32List.fromList([10, 2, 6, -2])));
      tensor.destroy();
      pooled.destroy();
    });
  });
}

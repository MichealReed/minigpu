import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

/// A buffer.
final class Buffer {
  Buffer(PlatformBuffer buffer) : platformBuffer = buffer;

  final PlatformBuffer platformBuffer;

  /// Reads data from the buffer synchronously.
  Future<void> read(
    Float32List outputData,
    int size, {
    int readOffset = 0,
  }) async =>
      platformBuffer.read(outputData, size, elementOffset: readOffset);

  /// Writes data to the buffer.
  void setData(Float32List inputData, int size) =>
      platformBuffer.setData(inputData, size);

  /// Destroys the buffer.
  void destroy() => platformBuffer.destroy();
}

class MinigpuAlreadyInitError extends Error {
  MinigpuAlreadyInitError([this.message]);

  final String? message;

  @override
  String toString() => message == null
      ? 'Minigpu already initialized'
      : 'Minigpu already initialized: $message';
}

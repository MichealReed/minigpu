import 'dart:io';
import 'dart:typed_data';

abstract class MinigpuPlatform {
  MinigpuPlatform(); // No automatic assignment

  static MinigpuPlatform? _instance;

  /// Returns the current instance; throws if not yet initialized.
  static MinigpuPlatform get instance {
    if (_instance == null) {
      throw Exception(
          "MinigpuPlatform is not initialized. Did you call MinigpuPlatform.initialize()?");
    }
    return _instance!;
  }

  /// Explicitly initializes the platform instance.
  static void initialize(MinigpuPlatform instance) {
    _instance = instance;
  }

  Future<void> initializeContext();
  void destroyContext();
  PlatformComputeShader createComputeShader();
  PlatformBuffer createBuffer(int bufferSize);
}

abstract class PlatformComputeShader {
  void loadKernelString(String kernelString);
  bool hasKernel();
  void setBuffer(int tag, PlatformBuffer buffer);
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ);
  void destroy();
}

abstract class PlatformBuffer {
  Future<void> read(
    Float32List outputData,
    int readElements, {
    int elementOffset = 0,
    int readBytes = 0,
    int byteOffset = 0,
  });
  void setData(Float32List inputData, int size);
  void destroy();
}

final class MinigpuPlatformOutOfMemoryException implements Exception {
  @override
  String toString() => 'Out of memory';
}

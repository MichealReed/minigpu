import 'dart:typed_data';

import 'platform_stub/minigpu_platform_stub.dart'
    if (dart.library.ffi) 'package:minigpu_ffi/minigpu_ffi.dart'
    if (dart.library.js) 'package:minigpu_web/minigpu_web.dart';

abstract class MinigpuPlatform {
  MinigpuPlatform(); // No automatic assignment

  static MinigpuPlatform? _instance;

  /// Returns the current instance; throws if not yet initialized.
  static MinigpuPlatform get instance {
    _instance ??= registeredInstance();
    return _instance!;
  }

  MinigpuPlatform registerInstance() =>
      throw UnimplementedError('No platform implementation available.');

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

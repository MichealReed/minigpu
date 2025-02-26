import 'dart:typed_data';

import 'package:plugin_platform_interface/plugin_platform_interface.dart';

abstract class MinigpuPlatform extends PlatformInterface {
  MinigpuPlatform() : super(token: _token);

  static final Object _token = Object();

  static late MinigpuPlatform _instance;

  static MinigpuPlatform get instance => _instance;

  static set instance(MinigpuPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<void> initializeContext();
  void destroyContext();
  PlatformComputeShader createComputeShader();
  PlatformBuffer createBuffer(int size, int memSize);
}

abstract class PlatformComputeShader {
  void loadKernelString(String kernelString);
  bool hasKernel();
  void setBuffer(int tag, PlatformBuffer buffer);
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ);
  void destroy();
}

abstract class PlatformBuffer {
  Future<void> read(Float32List outputData, int readElements,
      {int elementOffset = 0, int readBytes = 0, int byteOffset = 0});
  void setData(Float32List inputData, int size);
  void destroy();
}

final class MinigpuPlatformOutOfMemoryException implements Exception {
  @override
  String toString() => 'Out of memory';
}

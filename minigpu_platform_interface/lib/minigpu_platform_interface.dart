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

  void initializeContext();
  void destroyContext();
  PlatformComputeShader createComputeShader();
  PlatformBuffer createBuffer(int size, int memSize);
}

abstract class PlatformComputeShader {
  void loadKernelString(String kernelString);
  bool hasKernel();
  void setBuffer(String kernel, String tag, PlatformBuffer buffer);
  void dispatch(String kernel, int groupsX, int groupsY, int groupsZ);
  void destroy();
}

abstract class PlatformBuffer {
  void readSync(ByteBuffer outputData, int size);
  void readAsync(ByteBuffer outputData, int size,
      void Function(Float32List) callback, dynamic userData);
  void setData(ByteBuffer inputData, int size);
  void destroy();
}

final class MinigpuPlatformOutOfMemoryException implements Exception {
  String toString() => 'Out of memory';
}

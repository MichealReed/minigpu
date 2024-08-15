import 'dart:typed_data';

import 'package:plugin_platform_interface/plugin_platform_interface.dart';

abstract class MinigpuPlatform extends PlatformInterface {
  MinigpuPlatform() : super(token: _token);

  static final Object _token = Object();

  static MinigpuPlatform _instance = MinigpuPlatform();

  static MinigpuPlatform get instance => _instance;

  static set instance(MinigpuPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  void initializeContext();
  void destroyContext();
  PlatformComputeShader createComputeShader();
  PlatformBuffer createBuffer(
      PlatformComputeShader shader, int size, int memSize);
}

abstract class PlatformComputeShader {
  void loadKernelString(String kernelString);
  void loadKernelFile(String path);
  bool hasKernel();
  void setBuffer(String kernel, String tag, PlatformBuffer buffer);
  void dispatch(String kernel, int groupsX, int groupsY, int groupsZ);
  void destroy();
}

abstract class PlatformBuffer {
  void readSync(PlatformBuffer otherBuffer);
  void readAsync(
      PlatformBuffer otherBuffer, void Function() callback, dynamic userData);
  void destroy();
}

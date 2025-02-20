import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:minigpu_web/bindings/minigpu_bindings.dart' as wasm;

class MinigpuWeb extends MinigpuPlatform {
  MinigpuWeb._();
  static void registerWith(dynamic _) =>
      MinigpuPlatform.instance = MinigpuWeb._();

  @override
  Future<void> initializeContext() async {
    await wasm.mgpuInitializeContext();
  }

  @override
  void destroyContext() {
    wasm.mgpuDestroyContext();
  }

  @override
  PlatformComputeShader createComputeShader() {
    final shader = wasm.mgpuCreateComputeShader();
    return WebComputeShader(shader);
  }

  @override
  PlatformBuffer createBuffer(int size, int memSize) {
    final buff = wasm.mgpuCreateBuffer(size, memSize);
    return WebBuffer(buff);
  }
}

class WebComputeShader implements PlatformComputeShader {
  final wasm.MGPUComputeShader _shader;

  WebComputeShader(this._shader);

  @override
  void loadKernelString(String kernelString) {
    wasm.mgpuLoadKernel(_shader, kernelString);
  }

  @override
  bool hasKernel() {
    return wasm.mgpuHasKernel(_shader);
  }

  @override
  void setBuffer(String tag, PlatformBuffer buffer) {
    // Updated: Pass the shader pointer as first argument
    wasm.mgpuSetBuffer(_shader, tag, (buffer as WebBuffer)._buffer);
  }

  @override
  Future<void> dispatch(
      String kernel, int groupsX, int groupsY, int groupsZ) async {
    await wasm.mgpuDispatch(_shader, kernel, groupsX, groupsY, groupsZ);
  }

  @override
  void destroy() {
    wasm.mgpuDestroyComputeShader(_shader);
  }
}

class WebBuffer implements PlatformBuffer {
  final wasm.MGPUBuffer _buffer;

  WebBuffer(this._buffer);

  @override
  Future<void> readSync(Float32List outputData, int size) async {
    await wasm.mgpuReadBufferSync(_buffer, outputData, size);
  }

  @override
  void readAsync(Float32List outputData, int size,
      void Function(Float32List) callback, dynamic userData) {
    // Updated to remove the unused userData parameter and use proper types.
    wasm.mgpuReadBufferAsync(_buffer, outputData, size, callback);
  }

  @override
  void setData(Float32List inputData, int size) {
    wasm.mgpuSetBufferData(_buffer, inputData, size);
  }

  @override
  void destroy() {
    wasm.mgpuDestroyBuffer(_buffer);
  }
}

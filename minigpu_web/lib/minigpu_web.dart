import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:minigpu_web/bindings/minigpu_bindings.dart' as wasm;

class MinigpuWeb extends MinigpuPlatform {
  MinigpuWeb._();
  static void registerWith(dynamic _) =>
      MinigpuPlatform.instance = MinigpuWeb._();

  @override
  void initializeContext() {
    wasm.mgpuInitializeContext();
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
  PlatformBuffer createBuffer(dynamic size, int memSize) {
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
  void setBuffer(String kernel, String tag, PlatformBuffer buffer) {
    wasm.mgpuSetBuffer(_shader, kernel, tag, (buffer as WebBuffer)._buffer);
  }

  @override
  void dispatch(String kernel, int groupsX, int groupsY, int groupsZ) {
    wasm.mgpuDispatch(_shader, kernel, groupsX, groupsY, groupsZ);
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
  void readSync(ByteBuffer outputData, int size) {
    wasm.mgpuReadBufferSync(_buffer, outputData, size);
  }

  void readAsync(dynamic outputData, int size,
      void Function(Float32List) callback, dynamic userData) {
    wasm.mgpuReadBufferAsync(_buffer, outputData, size, callback);
  }

  @override
  void setData(ByteBuffer inputData, int size) {
    wasm.mgpuSetBufferData(_buffer, inputData, size);
  }

  @override
  void destroy() {
    wasm.mgpuDestroyBuffer(_buffer);
  }
}

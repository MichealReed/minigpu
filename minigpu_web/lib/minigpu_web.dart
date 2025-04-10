import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:minigpu_web/bindings/minigpu_bindings.dart' as wasm;

MinigpuPlatform registeredInstance() => MinigpuWeb._();

class MinigpuWeb extends MinigpuPlatform {
  MinigpuWeb._();

  static void registerWith(dynamic _) => MinigpuWeb._();

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
  PlatformBuffer createBuffer(int bufferSize) {
    final buff = wasm.mgpuCreateBuffer(bufferSize);
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
  void setBuffer(int tag, PlatformBuffer buffer) {
    // Updated: Pass the shader pointer as first argument
    wasm.mgpuSetBuffer(_shader, tag, (buffer as WebBuffer)._buffer);
  }

  @override
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ) async {
    await wasm.mgpuDispatch(_shader, groupsX, groupsY, groupsZ);
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
  Future<void> read(
    Float32List outputData,
    int readElements, {
    int elementOffset = 0,
    int readBytes = 0,
    int byteOffset = 0,
  }) async {
    await wasm.mgpuReadBufferSync(
      _buffer,
      outputData,
      readElements: readElements,
      elementOffset: elementOffset,
      readBytes: readBytes,
      byteOffset: byteOffset,
    );
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

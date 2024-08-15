import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:minigpu_web/bindings/minigpu_bindings.dart' as wasm;

class MinigpuWeb extends MinigpuPlatform {
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
  PlatformBuffer createBuffer(
      PlatformComputeShader shader, int size, int memSize) {
    final buffer =
        wasm.mgpuCreateBuffer(shader as WebComputeShader, size, memSize);
    return WebBuffer(buffer);
  }
}

class WebComputeShader implements PlatformComputeShader {
  final wasm.MGPUComputeShader _shader;

  WebComputeShader(this._shader);

  @override
  void loadKernelString(String kernelString) {
    wasm.mgpuLoadKernelString(_shader, kernelString);
  }

  @override
  void loadKernelFile(String path) {
    wasm.mgpuLoadKernelFile(_shader, path);
  }

  @override
  bool hasKernel() {
    return wasm.mgpuHasKernel(_shader);
  }

  @override
  void setBuffer(String kernel, String tag, PlatformBuffer buffer) {
    wasm.mgpuSetBuffer(_shader, kernel, tag, buffer as WebBuffer);
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
  void readSync(PlatformBuffer otherBuffer) {
    wasm.mgpuReadBufferSync(_buffer, otherBuffer as WebBuffer);
  }

  @override
  void readAsync(
      PlatformBuffer otherBuffer, void Function() callback, dynamic userData) {
    wasm.mgpuReadBufferAsync(
        _buffer, otherBuffer as WebBuffer, allowInterop(callback), userData);
  }

  @override
  void destroy() {
    wasm.mgpuDestroyBuffer(_buffer);
  }
}

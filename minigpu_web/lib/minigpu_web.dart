import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:minigpu_web/bindings/minigpu_bindings.dart' as wasm;
import 'package:minigpu_web/bindings/wasm/wasm.dart';

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
    if (shader == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return WebComputeShader(shader);
  }

  @override
  PlatformBuffer createBuffer(
      PlatformComputeShader shader, int size, int memSize) {
    final buffer = wasm.mgpuCreateBuffer(
        (shader as WebComputeShader)._shader, size, memSize);
    if (buffer == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return WebBuffer(buffer);
  }
}

class WebComputeShader implements PlatformComputeShader {
  final Pointer<wasm.MGPUComputeShader> _shader;

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
  final Pointer<wasm.MGPUBuffer> _buffer;

  WebBuffer(this._buffer);

  @override
  void readSync(PlatformBuffer otherBuffer) {
    wasm.mgpuReadBufferSync(_buffer, (otherBuffer as WebBuffer)._buffer);
  }

  @override
  void readAsync(
      PlatformBuffer otherBuffer, void Function() callback, dynamic userData) {
    wasm.mgpuReadBufferAsync(
        _buffer, (otherBuffer as WebBuffer)._buffer, callback, userData);
  }

  @override
  void writeFloat32List(Float32List data) {
    final list = wasm.mgpuCreateFloat32List(data.length);
    heap.copyFloat32List(list, data);
    wasm.mgpuCopyFloat32ListToBuffer(_buffer, list, data.length);
    wasm.mgpuDestroyFloat32List(list);
  }

  @override
  Float32List readFloat32List(int size) {
    return wasm.mgpuCopyBufferToFloat32List(_buffer, size);
  }

  @override
  void destroy() {
    wasm.mgpuDestroyBuffer(_buffer);
  }
}

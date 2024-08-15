@JS('Module')
library minigpu_bindings;

import 'dart:typed_data';

import 'package:js/js.dart';
import 'package:js/js_util.dart';
import 'package:minigpu_web/bindings/wasm/wasm.dart';

// Opaque types
final class MGPUComputeShader extends Opaque {}

final class MGPUBuffer extends Opaque {}

// JS interop
@JS('ccall')
external dynamic _ccall(
    String name, String returnType, List<String> argTypes, dynamic args);

// Context functions
void mgpuInitializeContext() {
  _ccall('mgpuInitializeContext', 'void', [], []);
}

void mgpuDestroyContext() {
  _ccall('mgpuDestroyContext', 'void', [], []);
}

// Compute shader functions
Pointer<MGPUComputeShader> mgpuCreateComputeShader() {
  final result = _ccall('mgpuCreateComputeShader', 'number', [], []);
  return Pointer(result, 1, safe: true);
}

void mgpuDestroyComputeShader(Pointer<MGPUComputeShader> shader) {
  _ccall('mgpuDestroyComputeShader', 'void', ['number'], [shader.addr]);
  malloc.free(shader);
}

void mgpuLoadKernelString(
    Pointer<MGPUComputeShader> shader, String kernelString) {
  _ccall('mgpuLoadKernelString', 'void', ['number', 'string'],
      [shader.addr, kernelString]);
}

void mgpuLoadKernelFile(Pointer<MGPUComputeShader> shader, String path) {
  _ccall(
      'mgpuLoadKernelFile', 'void', ['number', 'string'], [shader.addr, path]);
}

bool mgpuHasKernel(Pointer<MGPUComputeShader> shader) {
  return _ccall('mgpuHasKernel', 'boolean', ['number'], [shader.addr]);
}

// Buffer functions
Pointer<MGPUBuffer> mgpuCreateBuffer(
    Pointer<MGPUComputeShader> shader, int size, int memSize) {
  final result = _ccall('mgpuCreateBuffer', 'number',
      ['number', 'number', 'number'], [shader.addr, size, memSize]);
  return Pointer(result, size, safe: true);
}

void mgpuDestroyBuffer(Pointer<MGPUBuffer> buffer) {
  _ccall('mgpuDestroyBuffer', 'void', ['number'], [buffer.addr]);
  malloc.free(buffer);
}

void mgpuSetBuffer(Pointer<MGPUComputeShader> shader, String kernel, String tag,
    Pointer<MGPUBuffer> buffer) {
  _ccall('mgpuSetBuffer', 'void', ['number', 'string', 'string', 'number'],
      [shader.addr, kernel, tag, buffer.addr]);
}

// Dispatch functions
void mgpuDispatch(Pointer<MGPUComputeShader> shader, String kernel, int groupsX,
    int groupsY, int groupsZ) {
  _ccall(
      'mgpuDispatch',
      'void',
      ['number', 'string', 'number', 'number', 'number'],
      [shader.addr, kernel, groupsX, groupsY, groupsZ]);
}

// Buffer read functions
void mgpuReadBufferSync(
    Pointer<MGPUBuffer> buffer, Pointer<MGPUBuffer> otherBuffer) {
  _ccall('mgpuReadBufferSync', 'void', ['number', 'number'],
      [buffer.addr, otherBuffer.addr]);
}

void mgpuReadBufferAsync(
    Pointer<MGPUBuffer> buffer,
    Pointer<MGPUBuffer> otherBuffer,
    void Function() callback,
    dynamic userData) {
  final callbackPtr = allowInterop(callback);
  _ccall(
      'mgpuReadBufferAsync',
      'void',
      ['number', 'number', 'number', 'number'],
      [buffer.addr, otherBuffer.addr, callbackPtr, userData]);
}

// Float32List functions
Pointer<Float> mgpuCreateFloat32List(int size) {
  return malloc.allocate<Float>(size);
}

void mgpuDestroyFloat32List(Pointer<Float> list) {
  malloc.free(list);
}

void mgpuCopyFloat32ListToBuffer(
    Pointer<MGPUBuffer> buffer, Pointer<Float> list, int size) {
  heap.copyFloat32List(buffer, list.asTypedList(size) as Float32List);
}

Float32List mgpuCopyBufferToFloat32List(Pointer<MGPUBuffer> buffer, int size) {
  final list = mgpuCreateFloat32List(size);
  heap.copyFloat32List(list, buffer.asTypedList(size) as Float32List);
  final result = list.asTypedList(size) as Float32List;
  mgpuDestroyFloat32List(list);
  return result;
}

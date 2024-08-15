// minigpu_bindings.dart

@JS('Module')
library minigpu_bindings;

import 'package:js/js.dart';
import 'package:js/js_util.dart';

// Opaque types
@JS()
class MGPUComputeShader extends Opaque {}

@JS()
class MGPUBuffer extends Opaque {}

// JS interop
@JS('ccall')
external dynamic _ccall(
    String name, String returnType, List<String> argTypes, List args);

// Context functions
void mgpuInitializeContext() {
  _ccall('mgpuInitializeContext', 'void', [], []);
}

void mgpuDestroyContext() {
  _ccall('mgpuDestroyContext', 'void', [], []);
}

// Compute shader functions
MGPUComputeShader mgpuCreateComputeShader() {
  return MGPUComputeShader(_ccall('mgpuCreateComputeShader', 'number', [], []));
}

void mgpuDestroyComputeShader(MGPUComputeShader shader) {
  _ccall('mgpuDestroyComputeShader', 'void', ['number'], [shader]);
}

void mgpuLoadKernelString(MGPUComputeShader shader, String kernelString) {
  _ccall('mgpuLoadKernelString', 'void', ['number', 'string'],
      [shader, kernelString]);
}

void mgpuLoadKernelFile(MGPUComputeShader shader, String path) {
  _ccall('mgpuLoadKernelFile', 'void', ['number', 'string'], [shader, path]);
}

bool mgpuHasKernel(MGPUComputeShader shader) {
  return _ccall('mgpuHasKernel', 'boolean', ['number'], [shader]);
}

// Buffer functions
MGPUBuffer mgpuCreateBuffer(MGPUComputeShader shader, int size, int memSize) {
  return MGPUBuffer(_ccall('mgpuCreateBuffer', 'number',
      ['number', 'number', 'number'], [shader, size, memSize]));
}

void mgpuDestroyBuffer(MGPUBuffer buffer) {
  _ccall('mgpuDestroyBuffer', 'void', ['number'], [buffer]);
}

void mgpuSetBuffer(
    MGPUComputeShader shader, String kernel, String tag, MGPUBuffer buffer) {
  _ccall('mgpuSetBuffer', 'void', ['number', 'string', 'string', 'number'],
      [shader, kernel, tag, buffer]);
}

// Dispatch functions
void mgpuDispatch(MGPUComputeShader shader, String kernel, int groupsX,
    int groupsY, int groupsZ) {
  _ccall(
      'mgpuDispatch',
      'void',
      ['number', 'string', 'number', 'number', 'number'],
      [shader, kernel, groupsX, groupsY, groupsZ]);
}

// Buffer read functions
void mgpuReadBufferSync(MGPUBuffer buffer, MGPUBuffer otherBuffer) {
  _ccall('mgpuReadBufferSync', 'void', ['number', 'number'],
      [buffer, otherBuffer]);
}

void mgpuReadBufferAsync(MGPUBuffer buffer, MGPUBuffer otherBuffer,
    Function callback, dynamic userData) {
  final callbackPtr = allowInterop(callback);
  _ccall(
      'mgpuReadBufferAsync',
      'void',
      ['number', 'number', 'number', 'number'],
      [buffer, otherBuffer, callbackPtr, userData]);
}

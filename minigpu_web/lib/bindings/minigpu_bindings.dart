// minigpu_bindings.dart
@JS('Module')
library minigpu_bindings;

import 'dart:js_interop';
import 'dart:typed_data';
import 'package:js/js_util.dart';

// Interop types
@JS()
@staticInterop
class MGPUComputeShader {}

extension MGPUComputeShaderExtension on MGPUComputeShader {
  external void loadKernel(String kernelString);
  external bool hasKernel();
}

@JS()
@staticInterop
class MGPUBuffer {}

extension MGPUBufferExtension on MGPUBuffer {
  external void setData(JSAny data, int size);
  external void readSync(JSAny outputData, int size);
  external void readAsync(JSAny outputData, int size, JSFunction callback);
}

// Context functions
@JS('_mgpuInitializeContext')
external void _mgpuInitializeContext();

void mgpuInitializeContext() {
  _mgpuInitializeContext();
}

@JS('_mgpuDestroyContext')
external void _mgpuDestroyContext();

void mgpuDestroyContext() {
  _mgpuDestroyContext();
}

// Compute shader functions
@JS('_mgpuCreateComputeShader')
external MGPUComputeShader _mgpuCreateComputeShader();

MGPUComputeShader mgpuCreateComputeShader() {
  return _mgpuCreateComputeShader();
}

@JS('_mgpuDestroyComputeShader')
external void _mgpuDestroyComputeShader(MGPUComputeShader shader);

void mgpuDestroyComputeShader(MGPUComputeShader shader) {
  _mgpuDestroyComputeShader(shader);
}

void mgpuLoadKernel(MGPUComputeShader shader, String kernelString) {
  shader.loadKernel(kernelString);
}

bool mgpuHasKernel(MGPUComputeShader shader) {
  return shader.hasKernel();
}

// Buffer functions
@JS('_mgpuCreateBuffer')
external MGPUBuffer _mgpuCreateBuffer(
    MGPUComputeShader shader, int size, int memSize);

MGPUBuffer mgpuCreateBuffer(MGPUComputeShader shader, int size, int memSize) {
  return _mgpuCreateBuffer(shader, size, memSize);
}

@JS('_mgpuDestroyBuffer')
external void _mgpuDestroyBuffer(MGPUBuffer buffer);

void mgpuDestroyBuffer(MGPUBuffer buffer) {
  _mgpuDestroyBuffer(buffer);
}

@JS('_mgpuSetBuffer')
external void _mgpuSetBuffer(
    MGPUComputeShader shader, String kernel, String tag, MGPUBuffer buffer);

void mgpuSetBuffer(
    MGPUComputeShader shader, String kernel, String tag, MGPUBuffer buffer) {
  _mgpuSetBuffer(shader, kernel, tag, buffer);
}

// Dispatch functions
@JS('_mgpuDispatch')
external void _mgpuDispatch(MGPUComputeShader shader, String kernel,
    int groupsX, int groupsY, int groupsZ);

void mgpuDispatch(MGPUComputeShader shader, String kernel, int groupsX,
    int groupsY, int groupsZ) {
  _mgpuDispatch(shader, kernel, groupsX, groupsY, groupsZ);
}

// Buffer read functions
void mgpuReadBufferSync(MGPUBuffer buffer, JSAny outputData, int size) {
  buffer.readSync(outputData, size);
}

typedef ReadBufferAsyncCallbackFunc = void Function(JSAny);

void mgpuReadBufferAsync(MGPUBuffer buffer, JSAny outputData, int size,
    ReadBufferAsyncCallbackFunc callback) {
  // buffer.readAsync(outputData, size, allowInterop(callback));
}

void mgpuSetBufferData(MGPUBuffer buffer, JSAny inputData, int size) {
  buffer.setData(inputData, size);
}

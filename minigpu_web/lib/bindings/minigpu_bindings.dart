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
  external void loadKernel(JSString kernelString);
  external JSBoolean hasKernel();
}

@JS()
@staticInterop
class MGPUBuffer {}

extension MGPUBufferExtension on MGPUBuffer {
  external void setData(JSAny data, JSNumber size);
  external void readSync(JSAny outputData, JSNumber size);
  external void readAsync(JSAny outputData, JSNumber size, JSFunction callback);
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
  shader.loadKernel(kernelString.toJS);
}

bool mgpuHasKernel(MGPUComputeShader shader) {
  return shader.hasKernel().toDart;
}

// Buffer functions
@JS('_mgpuCreateBuffer')
external MGPUBuffer _mgpuCreateBuffer(
    MGPUComputeShader shader, JSNumber size, JSNumber memSize);

MGPUBuffer mgpuCreateBuffer(MGPUComputeShader shader, int size, int memSize) {
  return _mgpuCreateBuffer(shader, size.toJS, memSize.toJS);
}

@JS('_mgpuDestroyBuffer')
external void _mgpuDestroyBuffer(MGPUBuffer buffer);

void mgpuDestroyBuffer(MGPUBuffer buffer) {
  _mgpuDestroyBuffer(buffer);
}

@JS('_mgpuSetBuffer')
external void _mgpuSetBuffer(
    MGPUComputeShader shader, JSString kernel, JSString tag, MGPUBuffer buffer);

void mgpuSetBuffer(
    MGPUComputeShader shader, String kernel, String tag, MGPUBuffer buffer) {
  _mgpuSetBuffer(shader, kernel.toJS, tag.toJS, buffer);
}

// Dispatch functions
@JS('_mgpuDispatch')
external void _mgpuDispatch(MGPUComputeShader shader, JSString kernel,
    JSNumber groupsX, JSNumber groupsY, JSNumber groupsZ);

void mgpuDispatch(MGPUComputeShader shader, String kernel, int groupsX,
    int groupsY, int groupsZ) {
  _mgpuDispatch(shader, kernel.toJS, groupsX.toJS, groupsY.toJS, groupsZ.toJS);
}

// Buffer read functions
void mgpuReadBufferSync(MGPUBuffer buffer, ByteBuffer outputData, int size) {
  buffer.readSync(outputData.toJS, size.toJS);
}

typedef ReadBufferAsyncCallbackFunc = void Function(Float32List);

void mgpuReadBufferAsync(MGPUBuffer buffer, Float32List outputData, int size,
    ReadBufferAsyncCallbackFunc callback) {
  buffer.readAsync(
      outputData.toJS,
      size.toJS,
      allowInterop((JSAny result) {
        ByteBuffer byteBuffer = result as ByteBuffer;
        Float32List float32list = Float32List.view(byteBuffer, 0, size);
        callback(float32list);
      }).toJS);
}

void mgpuSetBufferData(MGPUBuffer buffer, ByteBuffer inputData, int size) {
  buffer.setData(inputData.toJS, size.toJS);
}

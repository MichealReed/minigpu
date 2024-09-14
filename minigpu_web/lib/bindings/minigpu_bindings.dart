@JS('Module')
library minigpu_bindings;

import 'dart:js_interop';
import 'dart:typed_data';

typedef MGPUBuffer = JSNumber;
typedef MGPUComputeShader = JSNumber;

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

@JS('_mgpuLoadKernel')
external void _mgpuLoadKernel(MGPUComputeShader shader, JSString kernelString);

void mgpuLoadKernel(MGPUComputeShader shader, String kernelString) {
  _mgpuLoadKernel(shader, kernelString.toJS);
}

@JS('_mgpuHasKernel')
external JSBoolean _mgpuHasKernel(MGPUComputeShader shader);

bool mgpuHasKernel(MGPUComputeShader shader) {
  return _mgpuHasKernel(shader).toDart;
}

// Buffer functions
@JS('_mgpuCreateBuffer')
external MGPUBuffer _mgpuCreateBuffer(JSNumber size, JSNumber memSize);

MGPUBuffer mgpuCreateBuffer(int size, int memSize) {
  return _mgpuCreateBuffer(size.toJS, memSize.toJS);
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
@JS('_mgpuReadBufferSync')
external void _mgpuReadBufferSync(
    MGPUBuffer buffer, JSAny outputData, JSNumber size);

void mgpuReadBufferSync(MGPUBuffer buffer, ByteBuffer outputData, int size) {
  _mgpuReadBufferSync(buffer, outputData.toJS, size.toJS);
}

typedef ReadBufferAsyncCallbackFunc = void Function(Float32List);

@JS('_mgpuReadBufferAsync')
external void _mgpuReadBufferAsync(MGPUBuffer buffer, JSAny outputData,
    JSNumber size, JSFunction callback, JSAny userData);

void mgpuReadBufferAsync(MGPUBuffer buffer, Float32List outputData, int size,
    ReadBufferAsyncCallbackFunc callback) {
  _mgpuReadBufferAsync(
      buffer,
      outputData.toJS,
      size.toJS,
      ((JSAny result) {
        ByteBuffer byteBuffer = (result as JSObject).dartify() as ByteBuffer;
        Float32List float32list = Float32List.view(byteBuffer, 0, size);
        callback(float32list);
      }).toJS,
      null.jsify()!);
}

@JS('_mgpuSetBufferData')
external void _mgpuSetBufferData(
    MGPUBuffer buffer, JSAny inputData, JSNumber size);

void mgpuSetBufferData(MGPUBuffer buffer, ByteBuffer inputData, int size) {
  _mgpuSetBufferData(buffer, inputData.toJS, size.toJS);
}

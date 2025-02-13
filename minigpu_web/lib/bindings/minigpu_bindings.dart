@JS('Module')
library minigpu_bindings;

import 'dart:js_interop';
import 'dart:typed_data';

typedef MGPUBuffer = JSNumber;
typedef MGPUComputeShader = JSNumber;

@JS('_malloc')
external JSNumber _malloc(JSNumber size);

@JS('_free')
external void _free(JSNumber ptr);

@JS('_memcpy')
external void _memcpy(JSNumber dest, JSAny src, JSNumber size);

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
  return _mgpuHasKernel(shader).dartify() as bool;
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
external void _mgpuSetBuffer(JSString tag, MGPUBuffer buffer);

void mgpuSetBuffer(String tag, MGPUBuffer buffer) {
  _mgpuSetBuffer(tag.toJS, buffer);
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
    MGPUBuffer buffer, JSNumber outputDataPtr, JSNumber size);

void mgpuReadBufferSync(MGPUBuffer buffer, Float32List outputData, int size) {
  final ptr = _malloc((size * Float32List.bytesPerElement).toJS);
  try {
    _mgpuReadBufferSync(buffer, ptr, size.toJS);
    final jsArray = JSFloat32Array(outputData.buffer.toJS);
    _memcpy(outputData.offsetInBytes.toJS, ptr,
        (size * Float32List.bytesPerElement).toJS);
  } finally {
    _free(ptr);
  }
}

typedef ReadBufferAsyncCallbackFunc = void Function(Float32List);

@JS('_mgpuReadBufferAsync')
external void _mgpuReadBufferAsync(MGPUBuffer buffer, JSNumber outputDataPtr,
    JSNumber size, JSFunction callback, JSAny userData);

void mgpuReadBufferAsync(MGPUBuffer buffer, Float32List outputData, int size,
    ReadBufferAsyncCallbackFunc callback) {
  final ptr = _malloc((size * Float32List.bytesPerElement).toJS);
  _mgpuReadBufferAsync(
      buffer,
      ptr,
      size.toJS,
      ((JSAny _) {
        final jsArray = JSFloat32Array(outputData.buffer.toJS);
        _memcpy(outputData.offsetInBytes.toJS, ptr,
            (size * Float32List.bytesPerElement).toJS);
        callback(outputData);
        _free(ptr);
      }).toJS,
      null.jsify()!);
}

@JS('_mgpuSetBufferData')
external void _mgpuSetBufferData(
    MGPUBuffer buffer, JSNumber inputDataPtr, JSNumber size);

void mgpuSetBufferData(MGPUBuffer buffer, Float32List inputData, int size) {
  final byteSize = size * Float32List.bytesPerElement;
  final ptr = _malloc(byteSize.toJS);

  try {
    // Copy data to the allocated memory
    final jsArray = JSFloat32Array(inputData.buffer.toJS);
    _memcpy(ptr, inputData.offsetInBytes.toJS, byteSize.toJS);

    print(buffer);

    // Call the WASM function with the pointer
    _mgpuSetBufferData(buffer, ptr, size.toJS);
  } finally {
    // Free the allocated memory
    _free(ptr);
  }
}

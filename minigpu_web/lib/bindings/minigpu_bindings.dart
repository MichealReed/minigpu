@JS('Module')
library minigpu_bindings;

import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';

typedef MGPUBuffer = JSNumber;
typedef MGPUComputeShader = JSNumber;

// js interop
@JS("HEAPU8")
external JSUint8Array HEAPU8;

@JS("HEAPF32")
external JSFloat32Array HEAPF32;

Uint8List get _heapU8 => HEAPU8.toDart;

Float32List get _heapF32 => HEAPF32.toDart;

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
  MGPUComputeShader shader = _mgpuCreateComputeShader();
  print("SHADER PTR  $shader");
  return shader;
}

@JS('_mgpuDestroyComputeShader')
external void _mgpuDestroyComputeShader(MGPUComputeShader shader);

void mgpuDestroyComputeShader(MGPUComputeShader shader) {
  _mgpuDestroyComputeShader(shader);
}

@JS('allocateUTF8')
external JSString allocateUTF8(String str);

@JS('_mgpuLoadKernel')
external void _mgpuLoadKernel(MGPUComputeShader shader, JSNumber kernelString);

void mgpuLoadKernel(MGPUComputeShader shader, String kernelString) {
  final bytes = utf8.encode(kernelString);
  final kernelBytes = Uint8List(bytes.length + 1)
    ..setRange(0, bytes.length, bytes)
    ..[bytes.length] = 0; // null terminator

  // Allocate memory for the string.
  final allocSize = kernelBytes.length * kernelBytes.elementSizeInBytes;
  final ptr = _malloc(allocSize.toJS);
  try {
    _heapU8.setAll(ptr.toDartInt, kernelBytes);
    _mgpuLoadKernel(shader, ptr);
  } finally {
    _free(ptr);
  }
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
external void _mgpuSetBuffer(
    MGPUComputeShader shader, JSNumber tag, MGPUBuffer buffer);

void mgpuSetBuffer(MGPUComputeShader shader, String tag, MGPUBuffer buffer) {
  final bytes = utf8.encode(tag);
  final tagBytes = Uint8List(bytes.length + 1)
    ..setRange(0, bytes.length, bytes)
    ..[bytes.length] = 0; // null terminator

  final allocSize = tagBytes.length * tagBytes.elementSizeInBytes;
  final ptr = _malloc(allocSize.toJS);
  try {
    _heapU8.setAll(ptr.toDartInt, tagBytes);
    _mgpuSetBuffer(shader, ptr, buffer);
  } finally {
    _free(ptr);
  }
}

// Dispatch functions
@JS('_mgpuDispatch')
external void _mgpuDispatch(MGPUComputeShader shader, JSNumber kernel,
    JSNumber groupsX, JSNumber groupsY, JSNumber groupsZ);

void mgpuDispatch(MGPUComputeShader shader, String kernel, int groupsX,
    int groupsY, int groupsZ) {
  final bytes = utf8.encode(kernel);
  final kernelBytes = Uint8List(bytes.length + 1)
    ..setRange(0, bytes.length, bytes)
    ..[bytes.length] = 0; // null terminator

  final allocSize = kernelBytes.length * kernelBytes.elementSizeInBytes;
  final ptr = _malloc(allocSize.toJS);

  try {
    _heapU8.setAll(ptr.toDartInt, kernelBytes);
    _mgpuDispatch(shader, ptr, groupsX.toJS, groupsY.toJS, groupsZ.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_ccall')
external JSPromise _ccall(JSString name, JSString returnType,
    JSArray<JSString> argTypes, JSArray args, JSObject opts);

// Buffer read functions
@JS('_mgpuReadBufferSync')
external void _mgpuReadBufferSync(
    MGPUBuffer buffer, JSNumber outputDataPtr, JSNumber size);

void mgpuReadBufferSync(MGPUBuffer buffer, Float32List outputData, int size) {
  final byteSize = size * Float32List.bytesPerElement;
  final ptr = _malloc(byteSize.toJS);
  final startIndex = ptr.toDartInt ~/ 4;
  try {
    _mgpuReadBufferSync(buffer, ptr, size.toJS);
    final output = _heapF32.sublist(startIndex, startIndex + size);
    outputData.setAll(0, output);
    print(outputData);
  } finally {
    //_free(ptr);
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
  final byteSize = inputData.length * Float32List.bytesPerElement;
  final ptr = _malloc(byteSize.toJS);

  final startIndex = ptr.toDartInt ~/ 4;
  final endIndex = startIndex + inputData.length;

  try {
    // Copy inputData to HEAPF32 starting at the calculated index
    _heapF32.setRange(startIndex, endIndex, inputData);

    // Call the WASM function with the pointer
    _mgpuSetBufferData(buffer, ptr, size.toJS);
  } finally {
    // Free the allocated memory
    _free(ptr);
  }
}

@JS('Module')
library minigpu_bindings;

import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';
import 'package:js_interop_utils/js_interop_utils.dart';

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

Future<void> mgpuInitializeContext() async {
  await ccall(
    "mgpuInitializeContext".toJS,
    "void".toJS,
    <JSAny>[].toJSDeep,
    <JSAny>[].toJSDeep,
    {"async": true}.toJSDeep,
  ).toDart;
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
  final kernelBytes =
      Uint8List(bytes.length + 1)
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
  MGPUComputeShader shader,
  JSNumber tag,
  MGPUBuffer buffer,
);

void mgpuSetBuffer(MGPUComputeShader shader, int tag, MGPUBuffer buffer) {
  try {
    _mgpuSetBuffer(shader, tag.toJS, buffer);
  } finally {}
}

Future<void> mgpuDispatch(
  MGPUComputeShader shader,
  int groupsX,
  int groupsY,
  int groupsZ,
) async {
  try {
    await ccall(
      "mgpuDispatch".toJS,
      "void".toJS,
      ["number", "number", "number", "number", "number"].toJSDeep,
      [shader, groupsX.toJS, groupsY.toJS, groupsZ.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
  } finally {}
}

@JS('ccall')
external JSPromise ccall(
  JSString name,
  JSString returnType,
  JSArray argTypes,
  JSArray args,
  JSObject opts,
);

Future<void> mgpuReadBufferSync(
  MGPUBuffer buffer,
  Float32List outputData, {
  int readElements = 0,
  int elementOffset = 0,
  int readBytes = 0,
  int byteOffset = 0,
}) async {
  // Determine the number of elements to read.
  final int sizeToRead =
      (readElements > 0)
          ? readElements * Float32List.bytesPerElement
          : (readBytes > 0 ? readBytes : (outputData.length - elementOffset));

  // If readElements is provided, calculate the effective byte offset based on element count.
  final int effectiveByteOffset =
      (readElements > 0)
          ? elementOffset * Float32List.bytesPerElement
          : byteOffset;

  final JSNumber ptr = _malloc(sizeToRead.toJS);
  final int startIndex = ptr.toDartInt ~/ Float32List.bytesPerElement;

  try {
    await ccall(
      "mgpuReadBufferSync".toJS,
      "number".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, sizeToRead.toJS, effectiveByteOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;

    final int elementsToRead = sizeToRead ~/ Float32List.bytesPerElement;

    final output = _heapF32.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {}
}

@JS('_mgpuSetBufferData')
external void _mgpuSetBufferData(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuSetBufferData(MGPUBuffer buffer, Float32List inputData, int size) {
  final byteSize = inputData.length * Float32List.bytesPerElement;
  final ptr = _malloc(byteSize.toJS);

  final startIndex = ptr.toDartInt ~/ 4;
  final endIndex = startIndex + inputData.length;

  try {
    // Copy inputData to HEAPF32 starting at the calculated index
    _heapF32.setRange(startIndex, endIndex, inputData);

    // Call the WASM function with the pointer
    _mgpuSetBufferData(buffer, ptr, byteSize.toJS);
  } finally {}
}

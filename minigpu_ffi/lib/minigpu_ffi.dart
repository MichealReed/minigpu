// ignore_for_file: omit_local_variable_types

import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:minigpu_ffi/minigpu_ffi_bindings.dart' as ffi;
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

// Dynamic library
const String _libName = 'minigpu_ffi';

typedef ReadAsyncCallbackFunc = Void Function(Pointer<Void>);
typedef ReadAsyncCallback = Pointer<NativeFunction<ReadAsyncCallbackFunc>>;
final _bindings = ffi.minigpuFfiBindings(() {
  if (Platform.isMacOS || Platform.isIOS) {
    return DynamicLibrary.open('$_libName.framework/$_libName');
  } else if (Platform.isAndroid || Platform.isLinux) {
    return DynamicLibrary.open('lib$_libName.so');
  } else if (Platform.isWindows) {
    return DynamicLibrary.open('$_libName.dll');
  }
  throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
}());

// Minigpu FFI
class MinigpuFfi extends MinigpuPlatform {
  MinigpuFfi._();

  static void registerWith() => MinigpuPlatform.instance = MinigpuFfi._();

  @override
  Future<void> initializeContext() async {
    _bindings.mgpuInitializeContext();
  }

  @override
  void destroyContext() {
    _bindings.mgpuDestroyContext();
  }

  @override
  PlatformComputeShader createComputeShader() {
    final self = _bindings.mgpuCreateComputeShader();
    if (self == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return FfiComputeShader(self);
  }

  @override
  PlatformBuffer createBuffer(int size, int memSize) {
    final self = _bindings.mgpuCreateBuffer(size, memSize);
    if (self == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return FfiBuffer(self);
  }
}

// Compute shader FFI
final class FfiComputeShader implements PlatformComputeShader {
  FfiComputeShader(Pointer<ffi.MGPUComputeShader> self) : _self = self;

  final Pointer<ffi.MGPUComputeShader> _self;

  @override
  void loadKernelString(String kernelString) {
    final kernelStringPtr = kernelString.toNativeUtf8();
    try {
      _bindings.mgpuLoadKernel(_self, kernelStringPtr.cast());
    } finally {
      malloc.free(kernelStringPtr);
    }
  }

  @override
  bool hasKernel() {
    return _bindings.mgpuHasKernel(_self) != 0;
  }

  @override
  void setBuffer(int tag, PlatformBuffer buffer) {
    try {
      _bindings.mgpuSetBuffer(_self, tag, (buffer as FfiBuffer)._self);
    } finally {}
  }

  @override
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ) async {
    try {
      _bindings.mgpuDispatch(_self, groupsX, groupsY, groupsZ);
    } finally {}
  }

  @override
  void destroy() {
    _bindings.mgpuDestroyComputeShader(_self);
  }
}

// Buffer FFI
final class FfiBuffer implements PlatformBuffer {
  FfiBuffer(Pointer<ffi.MGPUBuffer> self) : _self = self;

  final Pointer<ffi.MGPUBuffer> _self;

  @override
  Future<void> read(Float32List outputData, int readElements,
      {int elementOffset = 0, int readBytes = 0, int byteOffset = 0}) async {
    // Determine the number of elements to read.
    final int totalElements = outputData.length;
    final int sizeToRead = readElements != 0
        ? readElements
        : (readBytes != 0
            ? readBytes ~/ sizeOf<Float>()
            : totalElements - elementOffset);

    // Calculate effective byte offset:
    // If readElements is provided, we use elementOffset * sizeOf<Float>().
    // Otherwise, we use the provided byteOffset.
    final int effectiveByteOffset =
        readElements != 0 ? elementOffset * sizeOf<Float>() : byteOffset;

    final int byteSize = sizeToRead * sizeOf<Float>();
    final Pointer<Float> outputPtr = malloc.allocate<Float>(byteSize);
    final List<double> outputTypedList = outputPtr.asTypedList(sizeToRead);

    // Pass byteSize and effective offset to the native binding.
    _bindings.mgpuReadBufferSync(
        _self, outputPtr, byteSize, effectiveByteOffset);

    // Copy the read data into outputData starting at elementOffset.
    outputData.setAll(elementOffset, outputTypedList);
    malloc.free(outputPtr);
    return Future.value();
  }

  @override
  void setData(Float32List inputData, int size) {
    int elementCount = inputData.length;
    final inputPtr = malloc.allocate<Float>(elementCount * sizeOf<Float>());
    final inputTypedList = inputPtr.asTypedList(elementCount);
    inputTypedList.setAll(0, inputData);
    _bindings.mgpuSetBufferData(
        _self, inputPtr, elementCount * sizeOf<Float>());
    malloc.free(inputPtr);
  }

  @override
  void destroy() {
    _bindings.mgpuDestroyBuffer(_self);
  }
}

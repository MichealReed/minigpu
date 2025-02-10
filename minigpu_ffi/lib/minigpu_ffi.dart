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
  void initializeContext() {
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
  void setBuffer(String tag, PlatformBuffer buffer) {
    final tagPtr = tag.toNativeUtf8();
    try {
      _bindings.mgpuSetBuffer(
          _self, tagPtr.cast(), (buffer as FfiBuffer)._self);
    } finally {
      malloc.free(tagPtr);
    }
  }

  @override
  void dispatch(String kernel, int groupsX, int groupsY, int groupsZ) {
    final kernelPtr = kernel.toNativeUtf8();
    try {
      _bindings.mgpuDispatch(
          _self, kernelPtr.cast(), groupsX, groupsY, groupsZ);
    } finally {
      malloc.free(kernelPtr);
    }
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

  static final Map<int, Completer<void>> _completers = {};

  static void _readAsyncCallback(Pointer<Void> userData) {
    final callbackId = userData.address;
    final completer = _completers.remove(callbackId);
    if (completer != null) {
      completer.complete();
    }
  }

  @override
  void readSync(Float32List outputData, int size) {
    int elementCount = outputData.length;
    final outputPtr = malloc.allocate<Float>(elementCount * sizeOf<Float>());
    final outputTypedList = outputPtr.asTypedList(elementCount);

    _bindings.mgpuReadBufferSync(
        _self, outputPtr, elementCount * sizeOf<Float>());

    outputData.setAll(0, outputTypedList);
    malloc.free(outputPtr);
  }

  @override
  Future<void> readAsync(dynamic outputData, int size,
      void Function(Float32List) callback, dynamic userData) {
    final completer = Completer<void>();
    final callbackId = identical(userData, null) ? 0 : userData.hashCode;
    _completers[callbackId] = completer;
    final callbackPtr =
        Pointer.fromFunction<ReadAsyncCallbackFunc>(_readAsyncCallback);
    final userDataPtr = Pointer<Void>.fromAddress(callbackId);
    _bindings.mgpuReadBufferAsync(
        _self, outputData, size, callbackPtr, userDataPtr);
    return completer.future.then((_) => callback(outputData));
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

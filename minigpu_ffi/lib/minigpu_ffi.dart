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
    final completer = Completer<void>();

    void nativeCallback() {
      completer.complete();
    }

    final nativeCallable =
        NativeCallable<Void Function()>.listener(nativeCallback);

    _bindings.mgpuInitializeContextAsync(nativeCallable.nativeFunction);

    await completer.future;
    nativeCallable.close();
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
      final completer = Completer<void>();

      void nativeCallback() {
        completer.complete();
      }

      final nativeCallable =
          NativeCallable<Void Function()>.listener(nativeCallback);
      _bindings.mgpuDispatchAsync(
          _self, groupsX, groupsY, groupsZ, nativeCallable.nativeFunction);
      await completer.future;
      nativeCallable.close();
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
  Future<void> read(
    Float32List outputData,
    int readElements, {
    int elementOffset = 0,
    int readBytes = 0,
    int byteOffset = 0,
  }) async {
    // Determine how many elements to read.
    final int totalElements = outputData.length;
    final int sizeToRead = readElements != 0
        ? readElements
        : (readBytes != 0
            ? readBytes ~/ sizeOf<Float>()
            : totalElements - elementOffset);

    // Calculate effective byte offset: if readElements is given we use elementOffset * sizeOf<ffi.Float>(),
    // otherwise we use the provided byteOffset.
    final int effectiveByteOffset =
        readElements != 0 ? elementOffset * sizeOf<Float>() : byteOffset;

    // byteSize in bytes to pass to the native function.
    final int byteSize = sizeToRead * sizeOf<Float>();

    // Allocate a temporary native Float array to receive the data.
    final Pointer<Float> outputPtr = malloc.allocate<Float>(byteSize);

    // Create a completer that will be completed when the native callback fires.
    final completer = Completer<void>();

    // This is the native callback, which must match the MGPUCallback signature.
    void nativeCallback() {
      // Signal that the asynchronous native operation completed.
      completer.complete();
    }

    // Wrap the Dart function as a native callable.
    final nativeCallable =
        NativeCallable<Void Function()>.listener(nativeCallback);

    // Call the asynchronous native function.
    // _self is your MGPUBuffer pointer; _bindings.mgpuReadBufferAsync was set up from the FFI lookup.
    _bindings.mgpuReadBufferAsync(
      _self,
      outputPtr,
      byteSize,
      effectiveByteOffset,
      nativeCallable.nativeFunction,
    );

    // Wait until the callback signals that the data is ready.
    await completer.future;

    // Convert the native memory to a Dart typed list.
    final List<double> readData = outputPtr.asTypedList(sizeToRead);
    // Copy the data into the provided outputData starting at elementOffset.
    outputData.setAll(elementOffset, readData);

    // Free the allocated native memory and close the native callback.
    malloc.free(outputPtr);
    nativeCallable.close();

    return;
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

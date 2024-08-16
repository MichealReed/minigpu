// ignore_for_file: omit_local_variable_types

import "dart:ffi";
import "dart:io";
import "dart:typed_data";

import "package:ffi/ffi.dart";
import "package:minigpu_ffi/minigpu_ffi_bindings.dart" as ffi;
import "package:minigpu_platform_interface/minigpu_platform_interface.dart";

// Dynamic library
const String _libName = "minigpu_ffi";
final _bindings = ffi.minigpuFfiBindings(() {
  if (Platform.isMacOS || Platform.isIOS) {
    return DynamicLibrary.open("$_libName.framework/$_libName");
  } else if (Platform.isAndroid || Platform.isLinux) {
    return DynamicLibrary.open("lib$_libName.so");
  } else if (Platform.isWindows) {
    return DynamicLibrary.open("$_libName.dll");
  }
  throw UnsupportedError("Unsupported platform: ${Platform.operatingSystem}");
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
  PlatformBuffer createBuffer(
      PlatformComputeShader shader, int size, int memSize) {
    final self = _bindings.mgpuCreateBuffer(
        (shader as FfiComputeShader)._self, size, memSize);
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
      _bindings.mgpuLoadKernelString(_self, kernelStringPtr.cast());
    } finally {
      malloc.free(kernelStringPtr);
    }
  }

  @override
  void loadKernelFile(String path) {
    final pathPtr = path.toNativeUtf8();
    try {
      _bindings.mgpuLoadKernelFile(_self, pathPtr.cast());
    } finally {
      malloc.free(pathPtr);
    }
  }

  @override
  bool hasKernel() {
    return _bindings.mgpuHasKernel(_self) != 0;
  }

  @override
  void setBuffer(String kernel, String tag, PlatformBuffer buffer) {
    final kernelPtr = kernel.toNativeUtf8();
    final tagPtr = tag.toNativeUtf8();
    try {
      _bindings.mgpuSetBuffer(
          _self, kernelPtr.cast(), tagPtr.cast(), (buffer as FfiBuffer)._self);
    } finally {
      malloc.free(kernelPtr);
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

  @override
  void readSync(PlatformBuffer otherBuffer) {
    _bindings.mgpuReadBufferSync(_self, (otherBuffer as FfiBuffer)._self);
  }

  @override
  void readAsync(
      PlatformBuffer otherBuffer, void Function() callback, dynamic userData) {
    final callbackPtr = Pointer.fromFunction<Void Function(Pointer<Void>)>(
        (Pointer<Void> userData) {
      callback();
    });
    _bindings.mgpuReadBufferAsync(
        _self, (otherBuffer as FfiBuffer)._self, callbackPtr, userData.cast());
  }

  @override
  void writeFloat32List(Float32List data) {
    final dataPtr = malloc.allocate<Float>(data.length);
    final dataList = dataPtr.asTypedList(data.length);
    dataList.setAll(0, data);
    _bindings.mgpuWriteFloat32List(_self, dataPtr, data.length);
    malloc.free(dataPtr);
  }

  @override
  Float32List readFloat32List(int size) {
    final dataPtr = malloc.allocate<Float>(size);
    _bindings.mgpuReadFloat32List(_self, dataPtr, size);
    final dataList = dataPtr.asTypedList(size);
    final result = Float32List.fromList(dataList);
    malloc.free(dataPtr);
    return result;
  }

  @override
  void destroy() {
    _bindings.mgpuDestroyBuffer(_self);
  }
}

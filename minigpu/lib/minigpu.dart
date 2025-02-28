import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

/// Controls the initialization and destruction of the minigpu context.
final class Minigpu {
  Minigpu() {
    _finalizer.attach(this, _platform);
  }

  static final _finalizer = Finalizer<MinigpuPlatform>(
    (platform) => platform.destroyContext(),
  );
  static final _shaderFinalizer = Finalizer<ComputeShader>(
    (shader) => shader.destroy(),
  );
  static final _bufferFinalizer = Finalizer<Buffer>(
    (buffer) => buffer.destroy(),
  );

  final _platform = MinigpuPlatform.instance;
  bool isInitialized = false;

  /// Initializes the minigpu context.
  Future<void> init() async {
    if (isInitialized) throw MinigpuAlreadyInitError();

    await _platform.initializeContext();
    isInitialized = true;
  }

  /// Creates a compute shader.
  ComputeShader createComputeShader() {
    final platformShader = _platform.createComputeShader();
    final shader = ComputeShader._(platformShader);
    _shaderFinalizer.attach(this, shader);
    return shader;
  }

  /// Creates a buffer.
  Buffer createBuffer(int size, int memSize) {
    final platformBuffer = _platform.createBuffer(size, memSize);
    final buff = Buffer._(platformBuffer);
    _bufferFinalizer.attach(this, buff);
    return buff;
  }
}

/// A compute shader.
final class ComputeShader {
  ComputeShader._(PlatformComputeShader shader) : _shader = shader;

  final PlatformComputeShader _shader;
  Map<String, int> _kernelTags = {};

  /// Loads a kernel string into the shader.
  void loadKernelString(String kernelString) =>
      _shader.loadKernelString(kernelString);

  /// Checks if the shader has a kernel loaded.
  bool hasKernel() => _shader.hasKernel();

  /// Sets a buffer for the specified kernel and tag.
  void setBuffer(String tag, Buffer buffer) {
    if (!_kernelTags.containsKey(tag)) {
      _kernelTags[tag] = _kernelTags.length;
    } else {
      _kernelTags[tag] = _kernelTags[tag]!;
    }
    _shader.setBuffer(_kernelTags[tag]!, buffer._buffer);
  }

  /// Dispatches the specified kernel with the given work group counts.
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ) async =>
      _shader.dispatch(groupsX, groupsY, groupsZ);

  /// Destroys the compute shader.
  void destroy() => _shader.destroy();
}

/// A buffer.
final class Buffer {
  Buffer._(PlatformBuffer buffer) : _buffer = buffer;

  final PlatformBuffer _buffer;

  /// Reads data from the buffer synchronously.
  Future<void> read(
    Float32List outputData,
    int size, {
    int readOffset = 0,
  }) async => _buffer.read(outputData, size, elementOffset: readOffset);

  /// Writes data to the buffer.
  void setData(Float32List inputData, int size) =>
      _buffer.setData(inputData, size);

  /// Destroys the buffer.
  void destroy() => _buffer.destroy();
}

class MinigpuAlreadyInitError extends Error {
  MinigpuAlreadyInitError([this.message]);

  final String? message;

  @override
  String toString() =>
      message == null
          ? 'Minigpu already initialized'
          : 'Minigpu already initialized: $message';
}

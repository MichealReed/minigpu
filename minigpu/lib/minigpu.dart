import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

/// Controls the initialization and destruction of the minigpu context.
final class Minigpu {
  Minigpu() {
    _finalizer.attach(this, _platform);
  }

  static final _finalizer =
      Finalizer<MinigpuPlatform>((platform) => platform.destroyContext());
  static final _shaderFinalizer =
      Finalizer<ComputeShader>((shader) => shader.destroy());
  static final _bufferFinalizer =
      Finalizer<Buffer>((buffer) => buffer.destroy());

  final _platform = MinigpuPlatform.instance;
  var isInit = false;

  /// Initializes the minigpu context.
  Future<void> init() async {
    if (isInit) throw MinigpuAlreadyInitError();

    _platform.initializeContext();
    isInit = true;
  }

  /// Creates a compute shader.
  ComputeShader createComputeShader() {
    final platformShader = _platform.createComputeShader();
    final shader = ComputeShader._(platformShader);
    _shaderFinalizer.attach(this, shader);
    return shader;
  }

  /// Creates a buffer.
  Buffer createBuffer(ComputeShader shader, int size, int memSize) {
    final platformBuffer =
        _platform.createBuffer(shader._shader, size, memSize);
    final buffer = Buffer._(platformBuffer);
    _bufferFinalizer.attach(this, buffer);
    return buffer;
  }
}

/// A compute shader.
final class ComputeShader {
  ComputeShader._(PlatformComputeShader shader) : _shader = shader;

  final PlatformComputeShader _shader;

  /// Loads a kernel string into the shader.
  void loadKernelString(String kernelString) =>
      _shader.loadKernelString(kernelString);

  /// Checks if the shader has a kernel loaded.
  bool hasKernel() => _shader.hasKernel();

  /// Sets a buffer for the specified kernel and tag.
  void setBuffer(String kernel, String tag, Buffer buffer) =>
      _shader.setBuffer(kernel, tag, buffer._buffer);

  /// Dispatches the specified kernel with the given work group counts.
  void dispatch(String kernel, int groupsX, int groupsY, int groupsZ) =>
      _shader.dispatch(kernel, groupsX, groupsY, groupsZ);

  /// Destroys the compute shader.
  void destroy() => _shader.destroy();
}

/// A buffer.
final class Buffer {
  Buffer._(PlatformBuffer buffer) : _buffer = buffer;

  final PlatformBuffer _buffer;

  /// Reads data from the buffer synchronously.
  void readSync(dynamic outputData, int size) =>
      _buffer.readSync(outputData, size);

  /// Reads data from the buffer asynchronously.
  void readAsync(dynamic outputData, int size, void Function() callback,
          dynamic userData) =>
      _buffer.readAsync(outputData, size, callback, userData);

  /// Writes data to the buffer.
  void setData(dynamic inputData, int size) => _buffer.setData(inputData, size);

  /// Destroys the buffer.
  void destroy() => _buffer.destroy();
}

class MinigpuAlreadyInitError extends Error {
  MinigpuAlreadyInitError([this.message]);

  final String? message;

  @override
  String toString() => message == null
      ? 'Minigpu already initialized'
      : 'Minigpu already initialized: $message';
}

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:minigpu_ffi/minigpu_ffi.dart' as ffi;

MinigpuPlatform registeredInstance() => ffi.registeredInstance();

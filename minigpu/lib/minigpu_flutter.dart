import "dart:io";
import "package:flutter/services.dart";
import "package:minigpu/compute_shader.dart";

extension LoadKernels on ComputeShader {
  /// Loads a kernel from an asset file.
  Future<void> loadKernelAsset(String assetPath) async {
    final data = await rootBundle.loadString(assetPath);
    loadKernelString(data);
  }

  /// Loads a kernel from a file.
  Future<void> loadKernelFile(String filePath) async {
    final file = File(filePath);
    final data = await file.readAsString();
    loadKernelString(data);
  }
}

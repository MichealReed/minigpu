import 'dart:convert';

import 'package:native_assets_cli/native_assets_cli.dart';
import 'package:logging/logging.dart';
import 'package:native_assets_cli/code_assets_builder.dart';
import 'package:native_toolchain_cmake/native_toolchain_cmake.dart';

// Needs web conditional import
import 'dart:io';

final sourceDir = Directory('./src');

void main(List<String> args) async {
  await build(args, (input, output) async {
    Logger logger = Logger('build');
    await runBuild(input, output, sourceDir.absolute.uri);
    final minigpuLib = await output.findAndAddCodeAssets(input, names: {
      'minigpu_ffi': 'minigpu_ffi_bindings.dart',
    });
    final webgpuLib = await output.findAndAddCodeAssets(input,
        names: {
          'webgpu_dawn': 'webgpu_dawn.dart',
        },
        outDir: sourceDir.absolute.uri.resolve('external'));
    logger.info('Added files: $minigpuLib');
    logger.info('Added files: $webgpuLib');
  });
}

const name = 'mingpu_ffi.dart';

Future<void> runBuild(
  BuildInput input,
  BuildOutputBuilder output,
  Uri sourceDir,
) async {
  final builder = CMakeBuilder.create(
    name: name,
    sourceDir: sourceDir,
    defines: {},
  );
  await builder.run(
    input: input,
    output: output,
    logger: Logger('')
      ..level = Level.ALL
      ..onRecord.listen((record) => stderr.writeln(record)),
  );
}

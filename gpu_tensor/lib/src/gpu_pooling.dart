import 'package:minigpu/minigpu.dart';
import 'gpu_tensor_base.dart';

extension TensorPoolingMax on Tensor {
  Future<Tensor> maxPool({
    required List<int> poolSizes,
    List<int>? strides,
    List<int>? pads,
    List<int>? poolAxes,
  }) async {
    int effectiveRank = shape.length;
    int numPool = poolSizes.length;
    strides ??= List.filled(numPool, 1);
    pads ??= List.filled(numPool, 0);
    poolAxes ??= (effectiveRank == 3)
        ? [0, 1]
        : List.generate(numPool, (i) => effectiveRank - numPool + i);
    if (poolAxes.length != numPool) {
      throw Exception("poolAxes length must equal poolSizes length");
    }
    for (int ax in poolAxes) {
      if (ax < 0 || ax >= effectiveRank) {
        throw Exception(
            "poolAxes values must be between 0 and effectiveRank-1");
      }
    }

    // Build output shape by replacing dimensions corresponding to pooling axes.
    List<int> outputShape = List.from(shape);
    for (int j = 0; j < numPool; j++) {
      int ax = poolAxes[j];
      int outDim = ((shape[ax] + pads[j] - poolSizes[j]) ~/ strides[j]) + 1;
      outputShape[ax] = outDim;
    }
    Tensor result = await Tensor.create(outputShape);

    // Compute input strides (row‑major).
    List<int> inStrides = List.filled(effectiveRank, 1);
    for (int i = effectiveRank - 2; i >= 0; i--) {
      inStrides[i] = inStrides[i + 1] * shape[i + 1];
    }
    // Compute output strides.
    List<int> outStrides = List.filled(effectiveRank, 1);
    for (int i = effectiveRank - 2; i >= 0; i--) {
      outStrides[i] = outStrides[i + 1] * outputShape[i + 1];
    }
    int totalOut = outputShape.reduce((a, b) => a * b);

    // Set up max pooling specific shader strings.
    String accumulatorInit = "var maxVal: f32 = -3.4e38;";
    String accumulatorUpdate =
        "if(localMask > 0.0 && val > maxVal) { maxVal = val; }";
    String outputTransform = "output[idx] = maxVal;";

    // Generate the shader code.
    String shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const totalOut: u32 = ${totalOut}u;
${() {
      String s = "";
      for (int i = 0; i < effectiveRank; i++) {
        s += "const in_$i: u32 = ${shape[i]}u;\n";
      }
      for (int j = 0; j < numPool; j++) {
        s += "const stride_$j: u32 = ${strides![j]}u;\n";
        s += "const pad_$j: u32 = ${pads![j]}u;\n";
        s += "const poolSize_$j: u32 = ${poolSizes[j]}u;\n";
      }
      return s;
    }()}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx >= totalOut) {
    return;
  }

  var idx_rem: u32 = idx;
${() {
      String s = "";
      for (int i = 0; i < effectiveRank; i++) {
        s += "  let coord_$i: u32 = idx_rem / ${outStrides[i]}u;\n";
        s += "  idx_rem = idx_rem % ${outStrides[i]}u;\n";
      }
      return s;
    }()}

  $accumulatorInit

${() {
      String s = "";
      // Begin nested loops over each pooling axis.
      for (int j = 0; j < numPool; j++) {
        s +=
            "  for (var p_$j: u32 = 0u; p_$j < poolSize_$j; p_$j = p_$j + 1u) {\n";
      }
      return s;
    }()}

    // Compute the linear input index for this pooling sub-sample.
    let inIndexTemp: i32 = ${() {
      List<String> terms = [];
      for (int i = 0; i < effectiveRank; i++) {
        int pAxis = poolAxes!.indexOf(i);
        if (pAxis == -1) {
          terms.add("(i32(coord_$i) * i32(${inStrides[i]}))");
        } else {
          terms.add(
              "((i32(coord_$i) * i32(${strides![pAxis]})) - i32(${pads![pAxis]}) + i32(p_$pAxis)) * i32(${inStrides[i]})");
        }
      }
      return terms.join(" + ");
    }()};
    let inIndex: u32 = u32(inIndexTemp);
    
    // For this pooling sub-sample, reinitialize a per-iteration mask.
    var localMask: f32 = 1.0;
${() {
      String s = "";
      for (int i = 0; i < effectiveRank; i++) {
        int pAxis = poolAxes!.indexOf(i);
        if (pAxis != -1) {
          s +=
              "    let in_${i}_coord: i32 = (i32(coord_$i) * i32(stride_$pAxis)) - i32(pad_$pAxis) + i32(p_$pAxis);\n";
          s +=
              "    if (in_${i}_coord < 0 || u32(in_${i}_coord) >= in_$i) { localMask = 0.0; }\n";
        }
      }
      return s;
    }()}
    
    let val: f32 = input[inIndex];
    $accumulatorUpdate

${() {
      String s = "";
      for (int j = 0; j < numPool; j++) {
        s += "  } // end pooling loop for pool index $j\n";
      }
      return s;
    }()}
  
  $outputTransform
}
''';

    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    int workgroups = (totalOut + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }
}

extension TensorPoolingMin on Tensor {
  Future<Tensor> minPool({
    required List<int> poolSizes,
    List<int>? strides,
    List<int>? pads,
    List<int>? poolAxes,
  }) async {
    int effectiveRank = shape.length;
    int numPool = poolSizes.length;
    strides ??= List.filled(numPool, 1);
    pads ??= List.filled(numPool, 0);
    poolAxes ??= (effectiveRank == 3)
        ? [0, 1]
        : List.generate(numPool, (i) => effectiveRank - numPool + i);
    if (poolAxes.length != numPool) {
      throw Exception("poolAxes length must equal poolSizes length");
    }
    for (int ax in poolAxes) {
      if (ax < 0 || ax >= effectiveRank) {
        throw Exception(
            "poolAxes values must be between 0 and effectiveRank-1");
      }
    }

    // Build output shape by replacing dimensions corresponding to pooling axes.
    List<int> outputShape = List.from(shape);
    for (int j = 0; j < numPool; j++) {
      int ax = poolAxes[j];
      int outDim = ((shape[ax] + pads[j] - poolSizes[j]) ~/ strides[j]) + 1;
      outputShape[ax] = outDim;
    }
    Tensor result = await Tensor.create(outputShape);

    // Compute input strides (row‑major).
    List<int> inStrides = List.filled(effectiveRank, 1);
    for (int i = effectiveRank - 2; i >= 0; i--) {
      inStrides[i] = inStrides[i + 1] * shape[i + 1];
    }
    // Compute output strides.
    List<int> outStrides = List.filled(effectiveRank, 1);
    for (int i = effectiveRank - 2; i >= 0; i--) {
      outStrides[i] = outStrides[i + 1] * outputShape[i + 1];
    }
    int totalOut = outputShape.reduce((a, b) => a * b);

    // Set up min pooling specific shader strings.
    String accumulatorInit = "var minVal: f32 = 3.4e38;";
    String accumulatorUpdate =
        "if(localMask > 0.0 && val < minVal) { minVal = val; }";
    String outputTransform = "output[idx] = minVal;";

    // Generate the shader code.
    String shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const totalOut: u32 = ${totalOut}u;
${() {
      String s = "";
      for (int i = 0; i < effectiveRank; i++) {
        s += "const in_$i: u32 = ${shape[i]}u;\n";
      }
      for (int j = 0; j < numPool; j++) {
        s += "const stride_$j: u32 = ${strides![j]}u;\n";
        s += "const pad_$j: u32 = ${pads![j]}u;\n";
        s += "const poolSize_$j: u32 = ${poolSizes[j]}u;\n";
      }
      return s;
    }()}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx >= totalOut) {
    return;
  }

  var idx_rem: u32 = idx;
${() {
      String s = "";
      for (int i = 0; i < effectiveRank; i++) {
        s += "  let coord_$i: u32 = idx_rem / ${outStrides[i]}u;\n";
        s += "  idx_rem = idx_rem % ${outStrides[i]}u;\n";
      }
      return s;
    }()}

  $accumulatorInit

${() {
      String s = "";
      // Begin nested loops over each pooling axis.
      for (int j = 0; j < numPool; j++) {
        s +=
            "  for (var p_$j: u32 = 0u; p_$j < poolSize_$j; p_$j = p_$j + 1u) {\n";
      }
      return s;
    }()}

    // Compute the linear input index for this pooling sub-sample.
    let inIndexTemp: i32 = ${() {
      List<String> terms = [];
      for (int i = 0; i < effectiveRank; i++) {
        int pAxis = poolAxes!.indexOf(i);
        if (pAxis == -1) {
          terms.add("(i32(coord_$i) * i32(${inStrides[i]}))");
        } else {
          terms.add(
              "((i32(coord_$i) * i32(${strides![pAxis]})) - i32(${pads![pAxis]}) + i32(p_$pAxis)) * i32(${inStrides[i]})");
        }
      }
      return terms.join(" + ");
    }()};
    let inIndex: u32 = u32(inIndexTemp);
    
    // For this pooling sub-sample, reinitialize a per-iteration mask.
    var localMask: f32 = 1.0;
${() {
      String s = "";
      for (int i = 0; i < effectiveRank; i++) {
        int pAxis = poolAxes!.indexOf(i);
        if (pAxis != -1) {
          s +=
              "    let in_${i}_coord: i32 = (i32(coord_$i) * i32(stride_$pAxis)) - i32(pad_$pAxis) + i32(p_$pAxis);\n";
          s +=
              "    if (in_${i}_coord < 0 || u32(in_${i}_coord) >= in_$i) { localMask = 0.0; }\n";
        }
      }
      return s;
    }()}
    
    let val: f32 = input[inIndex];
    $accumulatorUpdate

${() {
      String s = "";
      for (int j = 0; j < numPool; j++) {
        s += "  } // end pooling loop for pool index $j\n";
      }
      return s;
    }()}
  
  $outputTransform
}
''';

    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    int workgroups = (totalOut + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }
}

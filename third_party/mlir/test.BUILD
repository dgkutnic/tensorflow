load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl")

licenses(["notice"])

package(default_visibility = [":test_friends"])

# Please only depend on this from MLIR tests.
package_group(
    name = "test_friends",
    packages = ["//..."],
)

cc_library(
    name = "IRProducingAPITest",
    hdrs = ["APITest.h"],
    includes = ["."],
)

gentbl(
    name = "TestLinalgTransformPatternsIncGen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "lib/DeclarativeTransforms/TestLinalgTransformPatterns.h.inc",
        ),
    ],
    tblgen = "@llvm-project-master//mlir:mlir-tblgen",
    td_file = "lib/DeclarativeTransforms/TestLinalgTransformPatterns.td",
    td_srcs = [
        "@llvm-project-master//mlir:LinalgTransformPatternsTdFiles",
    ],
)

gentbl(
    name = "TestVectorTransformPatternsIncGen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "lib/DeclarativeTransforms/TestVectorTransformPatterns.h.inc",
        ),
    ],
    tblgen = "@llvm-project-master//mlir:mlir-tblgen",
    td_file = "lib/DeclarativeTransforms/TestVectorTransformPatterns.td",
    td_srcs = [
        "@llvm-project-master//mlir:VectorTransformPatternsTdFiles",
    ],
)

gentbl(
    name = "TestLinalgMatmulToVectorPatternsIncGen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "lib/DeclarativeTransforms/TestLinalgMatmulToVectorPatterns.h.inc",
        ),
    ],
    tblgen = "@llvm-project-master//mlir:mlir-tblgen",
    td_file = "lib/DeclarativeTransforms/TestLinalgMatmulToVectorPatterns.td",
    td_srcs = [
        "@llvm-project-master//mlir:VectorTransformPatternsTdFiles",
        "@llvm-project-master//mlir:LinalgTransformPatternsTdFiles",
    ],
)

gentbl(
    name = "TestOpsIncGen",
    strip_include_prefix = "lib/Dialect/Test",
    tbl_outs = [
        (
            "-gen-op-decls",
            "lib/Dialect/Test/TestOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "lib/Dialect/Test/TestOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls",
            "lib/Dialect/Test/TestOpsDialect.h.inc",
        ),
        (
            "-gen-enum-decls",
            "lib/Dialect/Test/TestOpEnums.h.inc",
        ),
        (
            "-gen-enum-defs",
            "lib/Dialect/Test/TestOpEnums.cpp.inc",
        ),
        (
            "-gen-rewriters",
            "lib/Dialect/Test/TestPatterns.inc",
        ),
    ],
    tblgen = "@llvm-project-master//mlir:mlir-tblgen",
    td_file = "lib/Dialect/Test/TestOps.td",
    td_srcs = [
        "@llvm-project-master//mlir:OpBaseTdFiles",
        "@llvm-project-master//mlir:include/mlir/IR/OpAsmInterface.td",
        "@llvm-project-master//mlir:include/mlir/Interfaces/CallInterfaces.td",
        "@llvm-project-master//mlir:include/mlir/Interfaces/ControlFlowInterfaces.td",
        "@llvm-project-master//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@llvm-project-master//mlir:include/mlir/Interfaces/SideEffects.td",
    ],
    test = True,
)

cc_library(
    name = "TestDialect",
    srcs = [
        "lib/Dialect/Test/TestDialect.cpp",
        "lib/Dialect/Test/TestPatterns.cpp",
    ],
    hdrs = [
        "lib/Dialect/Test/TestDialect.h",
    ],
    includes = [
        "lib/DeclarativeTransforms",
        "lib/Dialect/Test",
    ],
    deps = [
        ":TestOpsIncGen",
        "@llvm-project-master//llvm:support",
        "@llvm-project-master//mlir:ControlFlowInterfaces",
        "@llvm-project-master//mlir:Dialect",
        "@llvm-project-master//mlir:IR",
        "@llvm-project-master//mlir:InferTypeOpInterface",
        "@llvm-project-master//mlir:Pass",
        "@llvm-project-master//mlir:SideEffects",
        "@llvm-project-master//mlir:StandardOps",
        "@llvm-project-master//mlir:StandardToStandard",
        "@llvm-project-master//mlir:TransformUtils",
        "@llvm-project-master//mlir:Transforms",
    ],
)

cc_library(
    name = "TestIR",
    srcs = [
        "lib/IR/TestFunc.cpp",
        "lib/IR/TestMatchers.cpp",
        "lib/IR/TestSideEffects.cpp",
        "lib/IR/TestSymbolUses.cpp",
    ],
    deps = [
        ":TestDialect",
        "@llvm-project-master//llvm:support",
        "@llvm-project-master//mlir:IR",
        "@llvm-project-master//mlir:Pass",
        "@llvm-project-master//mlir:StandardOps",
        "@llvm-project-master//mlir:Support",
    ],
)

cc_library(
    name = "TestPass",
    srcs = [
        "lib/Pass/TestPassManager.cpp",
    ],
    deps = [
        "@llvm-project-master//llvm:support",
        "@llvm-project-master//mlir:IR",
        "@llvm-project-master//mlir:Pass",
        "@llvm-project-master//mlir:Support",
    ],
)

cc_library(
    name = "TestTransforms",
    srcs = glob([
        "lib/Transforms/*.cpp",
    ]),
    defines = ["MLIR_CUDA_CONVERSIONS_ENABLED"],
    includes = ["lib/Dialect/Test"],
    deps = [
        ":TestDialect",
        ":TestLinalgMatmulToVectorPatternsIncGen",
        ":TestLinalgTransformPatternsIncGen",
        ":TestVectorTransformPatternsIncGen",
        "@llvm-project-master//llvm:support",
        "@llvm-project-master//mlir:Affine",
        "@llvm-project-master//mlir:Analysis",
        "@llvm-project-master//mlir:EDSC",
        "@llvm-project-master//mlir:GPUDialect",
        "@llvm-project-master//mlir:GPUToCUDATransforms",
        "@llvm-project-master//mlir:GPUTransforms",
        "@llvm-project-master//mlir:IR",
        "@llvm-project-master//mlir:LinalgOps",
        "@llvm-project-master//mlir:LinalgTransforms",
        "@llvm-project-master//mlir:LoopOps",
        "@llvm-project-master//mlir:Pass",
        "@llvm-project-master//mlir:StandardOps",
        "@llvm-project-master//mlir:Support",
        "@llvm-project-master//mlir:TransformUtils",
        "@llvm-project-master//mlir:Transforms",
        "@llvm-project-master//mlir:VectorOps",
        "@llvm-project-master//mlir:VectorToLLVM",
        "@llvm-project-master//mlir:VectorToLoops",
    ],
)

cc_library(
    name = "TestAffine",
    srcs = glob([
        "lib/Dialect/Affine/*.cpp",
    ]),
    deps = [
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
    ],
)

cc_library(
    name = "TestSPIRV",
    srcs = glob([
        "lib/Dialect/SPIRV/*.cpp",
    ]),
    deps = [
        "@llvm-project-master//mlir:IR",
        "@llvm-project-master//mlir:Pass",
        "@llvm-project-master//mlir:SPIRVDialect",
        "@llvm-project-master//mlir:SPIRVLowering",
    ],
)

load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow:tensorflow.bzl", "tf_native_cc_binary")

licenses(["notice"])

glob_lit_tests(
    data = [
        ":extra_files",
        ":test_utilities",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = [
        "mlir",
        "cc",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir/lite/tests/flatbuffer2mlir:importer_test_legacy_reshape",
        "//tensorflow/compiler/mlir/lite/tests/flatbuffer2mlir:importer_test_min_max",
        "//tensorflow/compiler/mlir/lite:flatbuffer_to_string",
        "//tensorflow/compiler/mlir/lite:flatbuffer_translate",
        "@llvm-project//llvm:FileCheck",
    ],
)

# Bundle together all the files that are used by the non-mlir file-based tests.
filegroup(
    name = "extra_files",
    srcs = glob(
        [
            "**/importer_test_min_max.cc.mlir",
            "**/reshape.mlir",
            "**/*.tflite"
        ],
    ),
)

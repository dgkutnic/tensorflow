load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_copts")

cc_library(
    name = "plaidml_codegen_test",
    testonly = True,
    srcs = ["plaidml_codegen_test.cc"],
    hdrs = ["plaidml_codegen_test.h"],
    deps = [
        "//tensorflow/compiler/xla/service:plaidml_plugin",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

tf_cc_test(
    name = "plaidml_eltwise_op_test",
    srcs = ["plaidml_eltwise_op_test.cc"],
    extra_copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
#        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/plaidml:compiler",
        "//tensorflow/compiler/xla/service/plaidml/tests:plaidml_codegen_test",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "plaidml_conv_op_test",
    srcs = ["plaidml_conv_op_test.cc"],
    extra_copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/plaidml:compiler",
        "//tensorflow/compiler/xla/service/plaidml/tests:plaidml_codegen_test",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "plaidml_dot_op_test",
    srcs = ["plaidml_dot_op_test.cc"],
    extra_copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/plaidml:compiler",
        "//tensorflow/compiler/xla/service/plaidml/tests:plaidml_codegen_test",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "plaidml_mnist_cnn_test",
    srcs = ["plaidml_mnist_cnn_test.cc"],
    extra_copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/plaidml:compiler",
        "//tensorflow/compiler/xla/service/plaidml/tests:plaidml_codegen_test",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "plaidml_softmax_test",
    srcs = ["plaidml_softmax_test.cc"],
    extra_copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/plaidml:compiler",
        "//tensorflow/compiler/xla/service/plaidml/tests:plaidml_codegen_test",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "plaidml_trivial_model_test",
    srcs = ["plaidml_trivial_model_test.cc"],
    extra_copts = ["-fexceptions"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/plaidml:compiler",
        "//tensorflow/compiler/xla/service/plaidml/tests:plaidml_codegen_test",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

py_binary(
    name = "resnext",
    srcs = [
        "resnext.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

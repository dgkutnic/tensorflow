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
        "@com_intel_plaidml//plaidml:testenv",
    ],
)

tf_cc_test(
    name = "plaidml_compare_op_test",
    srcs = ["plaidml_compare_op_test.cc"],
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
    name = "plaidml_eltwise_op_test",
    srcs = ["plaidml_eltwise_op_test.cc"],
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
    name = "plaidml_logical_op_test",
    srcs = ["plaidml_logical_op_test.cc"],
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

genrule(
    name = "plaidml_conv_op_test_gen",
    srcs = [],
    outs = ["plaidml_conv_op_test.h.inc"],
    cmd = "XLA_FLAGS=\"--xla_dump_to=tensorflow/compiler/xla/service/plaidml/tests/conv_hlo_module\" TF_XLA_FLAGS=\"--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=0\" python tensorflow/compiler/xla/service/plaidml/tests/plaidml_conv_op_test_gen.py && mv tensorflow/compiler/xla/service/plaidml/tests/plaidml_conv_op_test.h.inc \"$@\"",
    tools = ["plaidml_conv_op_test_gen.py"]
)

tf_cc_test(
    name = "plaidml_conv_op_test",
    srcs = ["plaidml_conv_op_test.cc", "plaidml_conv_op_test.h.inc"],
    extra_copts = ["-fexceptions"],
    data = [":plaidml_conv_op_test_gen"],
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

genrule(
    name = "plaidml_reduce_op_test_gen",
    srcs = [],
    outs = ["plaidml_reduce_op_test.h.inc"],
    cmd = "XLA_FLAGS=\"--xla_dump_to=tensorflow/compiler/xla/service/plaidml/tests/reduce_hlo_module\" TF_XLA_FLAGS=\"--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=0\" python tensorflow/compiler/xla/service/plaidml/tests/plaidml_reduce_op_test_gen.py && mv tensorflow/compiler/xla/service/plaidml/tests/plaidml_reduce_op_test.h.inc \"$@\"",
    tools = ["plaidml_reduce_op_test_gen.py"]
)

genrule(
    name = "plaidml_redwin_op_test_gen",
    srcs = [],
    outs = ["plaidml_redwin_op_test.h.inc"],
    cmd = "XLA_FLAGS=\"--xla_dump_to=tensorflow/compiler/xla/service/plaidml/tests/redwin_hlo_module\" TF_XLA_FLAGS=\"--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=0\" python tensorflow/compiler/xla/service/plaidml/tests/plaidml_redwin_op_test_gen.py && mv tensorflow/compiler/xla/service/plaidml/tests/plaidml_redwin_op_test.h.inc \"$@\"",
    tools = ["plaidml_redwin_op_test_gen.py"]
)

tf_cc_test(
    name = "plaidml_reduce_op_test",
    srcs = ["plaidml_reduce_op_test.cc", "plaidml_reduce_op_test.h.inc", "plaidml_redwin_op_test.h.inc"],
    extra_copts = ["-fexceptions"],
    data = [":plaidml_reduce_op_test_gen", ":plaidml_redwin_op_test_gen"],
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
    name = "plaidml_shape_op_test_gen",
    srcs = [
        "plaidml_shape_op_test_gen.py"
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py"
    ],
)

genrule(
    name = "plaidml_shape_op_test_genrule",
    srcs = ["//:configure"],
    outs = ["plaidml_shape_op_test.h.inc"],
    cmd = "./configure && XLA_FLAGS=\"--xla_dump_to=tensorflow/compiler/xla/service/plaidml/tests/hlo_module\" TF_XLA_FLAGS=\"--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=0\" $(location :plaidml_shape_op_test_gen) && mv tensorflow/compiler/xla/service/plaidml/tests/plaidml_shape_op_test.h.inc $@",
    tools = [":plaidml_shape_op_test_gen"]
)

tf_cc_test(
    name = "plaidml_shape_op_test",
    srcs = ["plaidml_shape_op_test.cc", "plaidml_shape_op_test.h.inc"],
    extra_copts = ["-fexceptions"],
    data = [":plaidml_shape_op_test_genrule"],
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
    name = "plaidml_i3d_test",
    srcs = ["plaidml_i3d_test.cc", "i3d_pretrained_inputs_and_weights.h", "i3d_output.h"],
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
    name = "plaidml_i3d_3b_test",
    srcs = ["plaidml_i3d_3b_test.cc", "i3d_pretrained_inputs_and_weights.h", "i3d_3b_output.h"],
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
    name = "plaidml_resnext_test",
    srcs = ["plaidml_resnext_test.cc", "resnext50_pretrained_inputs_and_weights.h", "resnext50_output.h"],
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
        "//tensorflow/core/platform:resource_loader",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "plaidml_resnext_stage1_test",
    srcs = ["plaidml_resnext_stage1_test.cc", "resnext50_pretrained_inputs_and_weights.h", "resnext50_stage1_output.h"],
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
        "//tensorflow/core/platform:resource_loader",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "plaidml_resnext_stage2_test",
    srcs = ["plaidml_resnext_stage2_test.cc", "resnext50_pretrained_inputs_and_weights.h", "resnext50_stage2_output.h"],
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
        "//tensorflow/core/platform:resource_loader",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "plaidml_resnext_stage3_test",
    srcs = ["plaidml_resnext_stage3_test.cc", "resnext50_pretrained_inputs_and_weights.h", "resnext50_stage3_output.h"],
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
        "//tensorflow/core/platform:resource_loader",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "plaidml_resnext_stage4_test",
    srcs = ["plaidml_resnext_stage4_test.cc", "resnext50_pretrained_inputs_and_weights.h", "resnext50_stage4_output.h"],
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
        "//tensorflow/core/platform:resource_loader",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "plaidml_resnext_test_semantics",
    srcs = ["plaidml_resnext_test_semantics.cc"],
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

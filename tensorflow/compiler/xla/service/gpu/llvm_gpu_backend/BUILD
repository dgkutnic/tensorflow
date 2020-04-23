load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

cc_library(
    name = "llvm_gpu_backend",
    srcs = [
        "dump_ir_pass.cc",
        "gpu_backend_lib.cc",
        "utils.cc",
    ],
    hdrs = [
        "dump_ir_pass.h",
        "gpu_backend_lib.h",
        "utils.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/gpu:gpu_types",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project-master//llvm:amdgpu_code_gen",
        "@llvm-project-master//llvm:analysis",
        "@llvm-project-master//llvm:bit_reader",
        "@llvm-project-master//llvm:bit_writer",
        "@llvm-project-master//llvm:code_gen",
        "@llvm-project-master//llvm:core",
        "@llvm-project-master//llvm:ipo",
        "@llvm-project-master//llvm:ir_reader",
        "@llvm-project-master//llvm:linker",
        "@llvm-project-master//llvm:nvptx_code_gen",  # buildcleaner: keep
        "@llvm-project-master//llvm:objc_arc",  # buildcleaner: keep
        "@llvm-project-master//llvm:scalar",
        "@llvm-project-master//llvm:support",
        "@llvm-project-master//llvm:target",
    ],
)

tf_cc_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.cc"],
    data = [
        "tests_data/saxpy.ll",
    ],
    deps = [
        ":llvm_gpu_backend",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:resource_loader",
        "@llvm-project-master//llvm:core",
        "@llvm-project-master//llvm:support",
    ],
)

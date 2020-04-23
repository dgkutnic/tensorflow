workspace(name = "org_tensorflow")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

# Load tf_repositories() before loading dependencies for other repository so
# that dependencies like com_google_protobuf won't be overridden.
load("//tensorflow:workspace.bzl", "tf_repositories")
# Please add all new TensorFlow dependencies in workspace.bzl.
tf_repositories()

register_toolchains("@local_config_python//:py_toolchain")

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

load("//third_party/toolchains/preconfig/generate:archives.bzl",
     "bazel_toolchains_archive")

bazel_toolchains_archive()

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)

bazel_toolchains_repositories()

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

load("//third_party/toolchains/preconfig/generate:workspace.bzl",
     "remote_config_workspace")

remote_config_workspace()

# Use `swift_rules_dependencies` to fetch the toolchains. With the
# `git_repository` rules above, the following call will skip redefining them.
load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")
swift_rules_dependencies()

# We must check the bazel version before trying to parse any other BUILD
# files, in case the parsing of those build files depends on the bazel
# version we require here.
load("//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least("1.0.0")

load("//third_party/android:android_configure.bzl", "android_configure")
android_configure(name="local_config_android")
load("@local_config_android//:android.bzl", "android_workspace")
android_workspace()

# If a target is bound twice, the later one wins, so we have to do tf bindings
# at the end of the WORKSPACE file.
load("//tensorflow:workspace.bzl", "tf_bind")
tf_bind()

http_archive(
    name = "inception_v1",
    build_file = "//:models.BUILD",
    sha256 = "7efe12a8363f09bc24d7b7a450304a15655a57a7751929b2c1593a71183bb105",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/inception_v1.zip",
    ],
)

http_archive(
    name = "mobile_ssd",
    build_file = "//:models.BUILD",
    sha256 = "bddd81ea5c80a97adfac1c9f770e6f55cbafd7cce4d3bbe15fbeb041e6b8f3e8",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_android_export.zip",
    ],
)

http_archive(
    name = "mobile_multibox",
    build_file = "//:models.BUILD",
    sha256 = "859edcddf84dddb974c36c36cfc1f74555148e9c9213dedacf1d6b613ad52b96",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1a.zip",
    ],
)

http_archive(
    name = "stylize",
    build_file = "//:models.BUILD",
    sha256 = "3d374a730aef330424a356a8d4f04d8a54277c425e274ecb7d9c83aa912c6bfa",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/stylize_v1.zip",
    ],
)

http_archive(
    name = "speech_commands",
    build_file = "//:models.BUILD",
    sha256 = "c3ec4fea3158eb111f1d932336351edfe8bd515bb6e87aad4f25dbad0a600d0c",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/speech_commands_v0.01.zip",
    ],
)

# Required for dependency @com_github_grpc_grpc

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load(
    "@build_bazel_rules_apple//apple:repositories.bzl",
    "apple_rules_dependencies",
)

apple_rules_dependencies()

load(
    "@build_bazel_apple_support//lib:repositories.bzl",
    "apple_support_dependencies",
)

apple_support_dependencies()

load("@upb//bazel:repository_defs.bzl", "bazel_version_repository")

bazel_version_repository(name = "bazel_version")

load("//third_party/googleapis:repository_rules.bzl", "config_googleapis")

config_googleapis()

# required for dependency @com_intel_plaidml
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

load("@com_intel_plaidml//bzl:workspace.bzl", "plaidml_workspace")

plaidml_workspace()

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

register_toolchains("@com_intel_plaidml//:py_toolchain")

load("@bazel_latex//:repositories.bzl", "latex_repositories")

latex_repositories()

http_archive(
    name = "com_grail_bazel_toolchain",
    sha256 = "17e12dfba5ae4fc735dded73500e24546ca74459bac68b8d6f68c7785e3e4500",
    strip_prefix = "bazel-toolchain-7abc442059ea5b08f9c9bc7f0c1450b914aca090",
    urls = ["https://github.com/plaidml/bazel-toolchain/archive/7abc442059ea5b08f9c9bc7f0c1450b914aca090.zip"],
)

load("@com_grail_bazel_toolchain//toolchain:deps.bzl", "bazel_toolchain_dependencies")

bazel_toolchain_dependencies()

load("@com_grail_bazel_toolchain//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    llvm_version = "8.0.0",
)

load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")

llvm_register_toolchains()

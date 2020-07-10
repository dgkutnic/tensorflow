load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def repo():
    http_archive(
        name = "com_intel_plaidml",
        url = "https://github.com/plaidml/plaidml/archive/39b55f858e4f0a2415e8c8f1a5f141021dc5cc93.zip",
        sha256 = "25424e779944bd67d748bd7f790a3ca3a364eebf23197f9f72d7338447e0098b",
        strip_prefix = "plaidml-39b55f858e4f0a2415e8c8f1a5f141021dc5cc93",
    )
    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

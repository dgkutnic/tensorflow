load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def repo():
    http_archive(
        name = "com_intel_plaidml",
        url = "https://github.com/plaidml/plaidml/archive/a55edee05b07967caaafb0333eb17f3ad1538ea6.zip",
        sha256 = "2553ecf22a88f030b0a8b1830a27dbb7a755bb48f025b8ac9ae5dc9a8669440a",
        strip_prefix = "plaidml-a55edee05b07967caaafb0333eb17f3ad1538ea6",
    )
    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

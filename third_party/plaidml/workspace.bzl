load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def repo():
    http_archive(
        name = "com_intel_plaidml",
        url = "https://github.com/plaidml/plaidml/archive/a234cfe4ae272b10aec89f2be0487c6fb7386b2a.zip",
        sha256 = "480729aebb7b89600974a00e27a565095e55a805c96ec18712a4f2d638ba2a7f",
        strip_prefix = "plaidml-a234cfe4ae272b10aec89f2be0487c6fb7386b2a",
    )
    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

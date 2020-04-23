load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def repo():
    http_archive(
        name = "com_intel_plaidml",
        build_file = clean_dep("//third_party/plaidml:BUILD.bazel"),
        sha256 = "fac131c6509128147cf7a51e0d09c4cef68bbc442dd720f52a830519d7d209d6",
        url = "https://github.com/plaidml/plaidml/archive/plaidml-v1.zip",
        strip_prefix = "plaidml-plaidml-v1",
    )
    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

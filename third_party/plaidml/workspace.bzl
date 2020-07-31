load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def repo():
    http_archive(
        name = "com_intel_plaidml",
        url = "https://github.com/plaidml/plaidml/archive/218a6093c86d7a09bf2c6fcabaaeb11c030f657d.zip",
        sha256 = "4ad757288893a5b4fa606150ee45018dc213e4297e6baf7e1b1d4645682da880",
        strip_prefix = "plaidml-218a6093c86d7a09bf2c6fcabaaeb11c030f657d",
    )
    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

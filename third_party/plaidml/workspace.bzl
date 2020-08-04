load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def repo():
    http_archive(
        name = "com_intel_plaidml",
        url = "https://github.com/plaidml/plaidml/archive/4e36154f112ee2db4326e67ede0366393653e706.zip",
        sha256 = "5a5310f79bd6be9a30a053a7cadce2f2e9532ccb3b5af1bd8bf78f3574ae4887",
        strip_prefix = "plaidml-4e36154f112ee2db4326e67ede0366393653e706",
    )
    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

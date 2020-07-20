load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def repo():
    http_archive(
        name = "com_intel_plaidml",
        url = "https://github.com/plaidml/plaidml/archive/c0c3e376d9e00eff751753ea2ba89ce413ce7cd7.zip",
        sha256 = "d51c4529e480c28c2b35a5c82a5c47e7c557a875720e5bf9a4a641d42bee8a91",
        strip_prefix = "plaidml-c0c3e376d9e00eff751753ea2ba89ce413ce7cd7",
    )
    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

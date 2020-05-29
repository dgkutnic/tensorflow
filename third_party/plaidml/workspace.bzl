load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def repo():
    http_archive(
        name = "com_intel_plaidml",
        sha256 = "cdd5140214b38d25ff5e883bae4926940801ce6185ec8742c60932d92cff9f84",
        url = "https://github.com/plaidml/plaidml/archive/71bf06bbdf8f5c6d02a93564f2dffa24e65393f6.zip",
        strip_prefix = "plaidml-71bf06bbdf8f5c6d02a93564f2dffa24e65393f6",
    )
    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def repo():
    http_archive(
        name = "com_intel_plaidml",
#        build_file = clean_dep("//third_party/plaidml:BUILD.bazel"),
#        sha256 = "d97ac6d59f3a335c91d224a6cccd326c1aa057d779d89ba17f7cc4e00769c39a",
#        sha256 = "899df7cccfcd26dc3a33dfd7eba97e34ba115fd65910d7631e0feede0c277e53",
        sha256 = "cdd5140214b38d25ff5e883bae4926940801ce6185ec8742c60932d92cff9f84",
        url = "https://github.com/plaidml/plaidml/archive/71bf06bbdf8f5c6d02a93564f2dffa24e65393f6.zip",
#         url = "https://github.com/plaidml/plaidml/archive/c5e1ef5f4c432c70df5b546e5c3915f5c4be0902.zip",
        strip_prefix = "plaidml-71bf06bbdf8f5c6d02a93564f2dffa24e65393f6",
#        strip_prefix = "plaidml-plaidml-v1",
    )
    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

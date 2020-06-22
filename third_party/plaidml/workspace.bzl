load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def repo():
    http_archive(
        name = "com_intel_plaidml",
        url = "https://github.com/plaidml/plaidml/archive/a26df83497ed99ed0fe2e3993450789e6901f74d.zip",
        sha256 = "28973c204455cd93085f4df6b888706057736be2ff68f6a888206559b093fd11",
        strip_prefix = "plaidml-a26df83497ed99ed0fe2e3993450789e6901f74d",
    )
    http_archive(
        name = "gflags",
        url = "https://github.com/gflags/gflags/archive/e292e0452fcfd5a8ae055b59052fc041cbab4abf.zip",
        sha256 = "7d17922978692175c67ef5786a014df44bfbfe3b48b30937cca1413d4ff65f75",
        strip_prefix = "gflags-e292e0452fcfd5a8ae055b59052fc041cbab4abf",
    )

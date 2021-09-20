load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "fb9f3c9a6011f966200027843d894923ebc9cd0b",
    remote = "https://github.com/nelhage/rules_boost",
    shallow_since = "1591047380 -0700",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

http_archive(
    name = "com_github_fmtlib_fmt",
    urls = ["https://github.com/fmtlib/fmt/archive/refs/tags/8.0.1.tar.gz"],
    strip_prefix = "fmt-8.0.1",
    build_file_content = 
"""
cc_library(
    name = 'fmt',
    hdrs = glob(['include/**']),
    srcs = [
        'src/format.cc',
    ],
    includes = ['include'],
    visibility = ['//visibility:public'],
)
"""
)

http_archive(
    name = "com_google_googletest",
    urls = ["https://github.com/google/googletest/archive/011959aafddcd30611003de96cfd8d7a7685c700.zip"],
    strip_prefix = "googletest-011959aafddcd30611003de96cfd8d7a7685c700",
    sha256 = "6a5d7d63cd6e0ad2a7130471105a3b83799a7a2b14ef7ec8d742b54f01a4833c",
)

http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-992381ced716ae12122360b0fbadbc3dda436dbf",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/992381ced716ae12122360b0fbadbc3dda436dbf.zip"],
)
# We still require the pybind library.
http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.7.1",
  urls = ["https://github.com/pybind/pybind11/archive/v2.7.1.tar.gz"],
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")
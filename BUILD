cc_library(
    name = "autograd",
    srcs = [
        "src/autograd.cpp",
        "src/operators.cpp",
        "src/optimizer.cpp",
        "src/variable.cpp",
    ],
    hdrs = [
        "include/autograd/autograd.h",
        "include/autograd/operators.h",
        "include/autograd/optimizer.h",
        "include/autograd/variable.h",
    ],
    copts = ["-std=c++17"],
    includes = ["include"],
    deps = [
        "@boost//:algorithm",
        "@boost//:log",
        "@com_github_fmtlib_fmt//:fmt",
    ],
)

cc_test(
    name = "autograd_test",
    srcs = ["tests/autograd_test.cpp"],
    copts = ["-std=c++17"],
    deps = [
        ":autograd",
        "@com_github_fmtlib_fmt//:fmt",
        "@com_google_googletest//:gtest_main",
    ],
)

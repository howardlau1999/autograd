cc_library(
    name = "autograd",
    hdrs = [
        "include/autograd/autograd.h",
        "include/autograd/variable.h",
        "include/autograd/operators.h",
        "include/autograd/optimizer.h",
    ],
    includes = ["include"],
    srcs = [
        "src/autograd.cpp",
        "src/variable.cpp",
        "src/operators.cpp",    
        "src/optimizer.cpp",
    ],
    deps = ["@com_gitlab_libeigen_eigen//:eigen", "@boost//:log"],
    copts = ["-std=c++17"],
)

cc_test(
    name = "autograd_test",
    srcs = ["tests/autograd_test.cpp"],
    deps = [
        ":autograd", 
        "@com_gitlab_libeigen_eigen//:eigen", 
        "@com_google_googletest//:gtest_main",
        "@com_github_fmtlib_fmt//:fmt",
    ],
    copts = ["-std=c++17"],
)
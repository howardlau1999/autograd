cc_library(
    name = "autograd",
    hdrs = [
        "include/autograd/autograd.h",
        "include/autograd/variable.h",
        "include/autograd/operators.h",
    ],
    includes = ["include"],
    srcs = [
        "src/autograd.cpp",
        "src/variable.cpp",
        "src/operators.cpp",    
    ],
    deps = ["@com_gitlab_libeigen_eigen//:eigen", "@boost//:log"],
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
)
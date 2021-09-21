#include <autograd/autograd.h>
#include <autograd/variable.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(autograd_py, m) {
  m.doc() = R"pbdoc(
        Autograd Library in Python
    )pbdoc";

  py::class_<autograd::Variable, std::shared_ptr<autograd::Variable>>(
      m, "Variable")
      .def("__add__", autograd::operator+, py::is_operator())
      .def("__radd__", autograd::operator+, py::is_operator())
      .def("__mul__", autograd::operator*, py::is_operator())
      .def("__rmul__", autograd::operator*, py::is_operator())
      .def("__truediv__", autograd::operator/, py::is_operator())
      .def("__sub__",
           static_cast<std::shared_ptr<autograd::Variable> (*)(
               std::shared_ptr<autograd::Variable>,
               std::shared_ptr<autograd::Variable>)>(&autograd::operator-),
           py::is_operator())
      .def("__rsub__",
           static_cast<std::shared_ptr<autograd::Variable> (*)(
               std::shared_ptr<autograd::Variable>,
               std::shared_ptr<autograd::Variable>)>(&autograd::operator-),
           py::is_operator())
      .def("__neg__",
           static_cast<std::shared_ptr<autograd::Variable> (*)(
               std::shared_ptr<autograd::Variable>)>(&autograd::operator-),
           py::is_operator())
      .def("__pow__", autograd::operator^, py::is_operator())
      .def("backward",
           [](std::shared_ptr<autograd::Variable> root) {
             autograd::run_backward(*root);
           })
      .def("detach", &autograd::Variable::detach)
      .def("log", &autograd::Variable::log)
      .def("relu", &autograd::Variable::relu)
      .def("sigmoid", &autograd::Variable::sigmoid)
      .def("zero_grad", &autograd::Variable::zero_grad)
      .def("grad",
           [](std::shared_ptr<autograd::Variable> var) {
             return var->grad().front();
           })
      .def("value",
           [](std::shared_ptr<autograd::Variable> var) {
             return var->value().front();
           })
      .def("__repr__", &autograd::Variable::to_string);

  m.def("variable", &autograd::variable, R"pbdoc(
        Create an autograd variable.
    )pbdoc");

  m.def("print_graph", [](std::shared_ptr<autograd::Variable> root) {
    autograd::print_graph(*root);
  });

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
#if !defined(__TENSOR_H__)
#define __TENSOR_H__

#include <boost/log/trivial.hpp>
#include <fmt/format.h>
#include <memory>

namespace autograd {

class Node;

class Edge {
  std::shared_ptr<Node> grad_fn_;
  int input_nr_ = 0;

public:
  Edge() = default;
  Edge(std::shared_ptr<Node> grad_fn, int input_nr)
      : grad_fn_(grad_fn), input_nr_(input_nr) {}

  std::shared_ptr<Node> grad_fn() { return grad_fn_; }

  int input_nr() { return input_nr_; }

  void set_grad_fn(std::shared_ptr<Node> grad_fn) { grad_fn_ = grad_fn; }
};

class Variable : public std::enable_shared_from_this<Variable> {
  using T = float;

  // Autograd Metadata
  bool requires_grad_ = true;

public:
  T value_ = T();
  T grad_ = T();
  Edge gradient_edge_;

  void set_gradient_edge(Edge &&gradient_edge);

  Edge gradient_edge();

  void add_grad(T grad_value) { grad_ += grad_value; }

  std::shared_ptr<Variable> shared_ptr() { return shared_from_this(); }

  Variable() = default;

  Variable(T value) : value_(value) {}

  Variable operator=(T value) { value_ = value; }

  T grad() { return grad_; }
  
  T value() { return value_; }

  void zero_grad() { grad_ = 0.0f; }

  bool requires_grad() { return requires_grad_; }

  void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

  std::string to_string() const {
    return fmt::format("Variable @ {} value = {} grad = {}", fmt::ptr(this),
                       value_, grad_);
  }

  std::shared_ptr<Variable> detach();
  std::shared_ptr<Variable> log();
  std::shared_ptr<Variable> relu();
  std::shared_ptr<Variable> sigmoid();
};

std::shared_ptr<Variable> variable(float v);

std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs);
std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> var);
std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs);
std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs);
std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs);
std::shared_ptr<Variable> operator^(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs);

} // namespace autograd

#endif // __TENSOR_H__

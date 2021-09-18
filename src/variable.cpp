#include "autograd/variable.h"
#include "autograd/operators.h"

#include <cmath>
#include <memory>

namespace autograd {

Edge Variable::gradient_edge() {
  if (!gradient_edge_.grad_fn() && requires_grad_) {
    std::shared_ptr<AccumulateGrad> grad_fn =
        std::make_shared<AccumulateGrad>();
    grad_fn->variable_ = shared_from_this();
    grad_fn->add_input_nr();
    gradient_edge_.set_grad_fn(grad_fn);
  }
  return gradient_edge_;
}

std::shared_ptr<Variable> Variable::detach() {
  std::shared_ptr<Variable> variable = std::make_shared<Variable>(value_);
  variable->requires_grad_ = false;
  return variable;
}

std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs) {
  std::shared_ptr<AddBackward> grad_fn = std::make_shared<AddBackward>();
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(lhs->value_ + rhs->value_);
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(lhs->gradient_edge());
  grad_fn->add_next_edge(rhs->gradient_edge());
  return result;
}

std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs) {
  std::shared_ptr<SubBackward> grad_fn = std::make_shared<SubBackward>();
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(lhs->value_ - rhs->value_);
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(lhs->gradient_edge());
  grad_fn->add_next_edge(rhs->gradient_edge());
  return result;
}

std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs) {
  std::shared_ptr<MulBackward> grad_fn = std::make_shared<MulBackward>();
  grad_fn->self_ = lhs;
  grad_fn->other_ = rhs;
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(lhs->value_ * rhs->value_);
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(lhs->gradient_edge());
  grad_fn->add_next_edge(rhs->gradient_edge());
  return result;
}

std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs) {
  std::shared_ptr<DivBackward> grad_fn = std::make_shared<DivBackward>();
  grad_fn->self_ = lhs;
  grad_fn->other_ = rhs;
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(lhs->value_ / rhs->value_);
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(lhs->gradient_edge());
  grad_fn->add_next_edge(rhs->gradient_edge());
  return result;
}

std::shared_ptr<Variable> operator^(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs) {
  std::shared_ptr<PowBackward> grad_fn = std::make_shared<PowBackward>();
  grad_fn->self_ = lhs;
  grad_fn->other_ = rhs;
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(std::pow(lhs->value_, rhs->value_));
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(lhs->gradient_edge());
  grad_fn->add_next_edge(rhs->gradient_edge());
  return result;
}


std::shared_ptr<Variable> Variable::log() {
  std::shared_ptr<LogBackward> grad_fn = std::make_shared<LogBackward>();
  grad_fn->self_ = shared_from_this();
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(std::log(value_));
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(gradient_edge());
  return result;
}


} // namespace autograd
#include "autograd/operators.h"
#include <cmath>

namespace autograd {

variable_list AccumulateGrad::apply(variable_list &&grads) {
  auto grad = grads[0].value_;
  if (auto ptr = variable_.lock()) {
    ptr->grad_ += grad;
  }
  return variable_list();
}

variable_list AddBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_;
  variable_list grads_input{grad, grad};
  return grads_input;
}

variable_list MulBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_.front();
  variable_list grads_input{other_->value_.front() * grad, self_->value_.front() * grad};
  return grads_input;
}

variable_list DivBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_.front();
  variable_list grads_input{1.0f / other_->value_.front() * grad,
                            -self_->value_.front() / (other_->value_.front() * other_->value_.front()) *
                                grad};
  return grads_input;
}

variable_list SubBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_.front();
  variable_list grads_input{grad, -grad};
  return grads_input;
}

variable_list PowBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_.front();
  auto xvalue = self_->value_.front();
  auto yvalue = other_->value_.front();
  variable_list grads_input{grad * yvalue * std::pow(xvalue, yvalue - 1),
                            grad * std::pow(xvalue, yvalue) * std::log(xvalue)};
  return grads_input;
}

variable_list LogBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_.front();
  auto value = self_->value_.front();
  variable_list grads_input{grad / value};
  return grads_input;
}

variable_list ReLUBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_.front();
  auto value = self_->value_.front();
  float grad_value = value >= 0 ? grad : 0.0f;
  variable_list grads_input{grad_value};
  return grads_input;
}

variable_list NegBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_.front();
  variable_list grads_input{-grad};
  return grads_input;
}

} // namespace autograd

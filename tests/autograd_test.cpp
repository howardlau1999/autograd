#include <autograd/autograd.h>
#include <autograd/variable.h>
#include <cmath>
#include <gtest/gtest.h>

using autograd::Variable;

std::shared_ptr<Variable> variable(float v) {
  return std::make_shared<Variable>(v);
}

TEST(VariableBackward, Add) {
  auto x = variable(5.0);
  auto y = variable(4.0);
  auto z = x + y;
  ASSERT_FLOAT_EQ(z->value_, 9.0);
  autograd::run_backward(*z);
  ASSERT_FLOAT_EQ(x->grad_, 1.0);
  ASSERT_FLOAT_EQ(y->grad_, 1.0);
}

TEST(VariableBackward, Sub) {
  auto x = variable(5.0);
  auto y = variable(4.0);
  auto z = x - y;
  ASSERT_FLOAT_EQ(z->value_, 1.0);
  autograd::run_backward(*z);
  ASSERT_FLOAT_EQ(x->grad_, 1.0);
  ASSERT_FLOAT_EQ(y->grad_, -1.0);
}

TEST(VariableBackward, Mul) {
  auto x = variable(5.0);
  auto y = variable(4.0);
  auto z = x * y;
  ASSERT_FLOAT_EQ(z->value_, 20.0);
  autograd::run_backward(*z);
  ASSERT_FLOAT_EQ(x->grad_, 4.0);
  ASSERT_FLOAT_EQ(y->grad_, 5.0);
}

TEST(VariableBackward, MulAdd) {
  auto x = variable(5.0);
  auto y = variable(3.0);
  auto u = variable(4.0);
  auto z = u * (x + y);
  ASSERT_FLOAT_EQ(z->value_, 32.0);
  autograd::run_backward(*z);
  ASSERT_FLOAT_EQ(u->grad_, 8.0);
  ASSERT_FLOAT_EQ(y->grad_, 4.0);
  ASSERT_FLOAT_EQ(x->grad_, 4.0);
}

TEST(VariableBackward, Pow) {
  auto e = variable(std::exp(1));
  auto x = variable(3.0);
  auto z = e ^ x;
  ASSERT_FLOAT_EQ(z->value_, std::exp(3.0));
  autograd::run_backward(*z);
  ASSERT_FLOAT_EQ(x->grad_, std::exp(3.0));
}

TEST(VariableBackward, Log) {
  auto x = variable(1.0);
  auto z = x->log();
  ASSERT_FLOAT_EQ(z->value_, 0);
  autograd::run_backward(*z);
  ASSERT_FLOAT_EQ(x->grad_, 1.0);
}

TEST(VariableBackward, Order2) {
  auto x = variable(5.0);
  auto y = variable(3.0);
  auto z = y * x * x + x * y * y;
  ASSERT_FLOAT_EQ(z->value_, 120.0);
  autograd::run_backward(*z);
  ASSERT_FLOAT_EQ(y->grad_, 25.0 + 30.0); // dz/dy = x * x + 2 * x * y
  ASSERT_FLOAT_EQ(x->grad_, 30.0 + 9.0);  // dz/dx = 2 * y * x + y * y;
}

TEST(VariableBackward, NoGrad) {
  auto x = variable(5.0);
  auto y = variable(3.0);
  x->set_requires_grad(false);
  auto z = y * x * x + x * y * y;
  ASSERT_FLOAT_EQ(z->value_, 120.0);
  autograd::run_backward(*z);
  ASSERT_FLOAT_EQ(y->grad_, 25.0 + 30.0); // dz/dy = x * x + 2 * x * y
  ASSERT_FLOAT_EQ(x->grad_, 0.0);         // dz/dx = 0;
}

TEST(VariableBackward, StopGradient) {
  auto x = variable(5.0);
  auto y = variable(4.0);
  auto w = variable(3.0);
  auto u = variable(10.0);
  auto v = variable(20.0);
  auto stop_grad = (x + y)->detach();
  auto z = u * v + w * stop_grad;
  ASSERT_FLOAT_EQ(z->value_, 227.0);
  autograd::run_backward(*z);
  ASSERT_FLOAT_EQ(w->grad_, 9.0);
  ASSERT_FLOAT_EQ(x->grad_, 0.0);
  ASSERT_FLOAT_EQ(y->grad_, 0.0);
}

TEST(Integration, LinearRegression) {
  auto w = variable(0.128911248);
  auto b = variable(-0.423790183);
  float learning_rate = 0.003;
  // y = x + 1
  for (int i = 0; i <= 5000; ++i) {
    w->grad_ = 0.0;
    b->grad_ = 0.0;
    float loss = 0.0;
    for (float xv = 0.0; xv < 32.0; xv += 1.0) {
      auto x = variable(xv);
      auto y = variable(xv + 1.0f);
      x->set_requires_grad(false);
      y->set_requires_grad(false);
      auto z = (w * x + b - y) * (w * x + b - y);
      loss += z->value_;
      autograd::run_backward(*z);
    }
    w->value_ -= w->grad_ / 32.0f * learning_rate;
    b->value_ -= b->grad_ / 32.0f * learning_rate;
    if (i % 1000 == 0) {
      BOOST_LOG_TRIVIAL(debug)
          << "Iter = " << i << ", Loss = " << (loss / 32.0f);
    }
  }
  BOOST_LOG_TRIVIAL(debug) << "w = " << w->value_ << ", b = " << b->value_;
  ASSERT_NEAR(w->value_, 1.0, 1e-3);
  ASSERT_NEAR(b->value_, 1.0, 1e-3);
}
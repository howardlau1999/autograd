#include <autograd/autograd.h>
#include <autograd/optimizer.h>
#include <autograd/variable.h>
#include <cmath>
#include <fmt/format.h>
#include <gtest/gtest.h>

using autograd::Variable;
using autograd::variable;

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

TEST(VariableBackward, Neg) {
  auto x = variable(5.0);
  auto negx = -x;
  ASSERT_FLOAT_EQ(negx->value_, -5.0);
  autograd::run_backward(*negx);
  ASSERT_FLOAT_EQ(x->grad_, -1.0);
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

TEST(VariableForward, sigmoid) {
  auto x = variable(0.0f);
  auto y = x->sigmoid();
  ASSERT_FLOAT_EQ(y->value_, 0.5f);
}

TEST(VariableForward, log) {
  auto x = variable(1.0f);
  auto y = x->log();
  ASSERT_FLOAT_EQ(y->value_, 0.0f);
}

std::shared_ptr<Variable> mse_loss(std::shared_ptr<Variable> predicted,
                                   std::shared_ptr<Variable> target) {
  return (predicted - target) * (predicted - target);
}

TEST(Integration, Order1LinearRegression) {
  auto w = variable(0.128911248);
  auto b = variable(-0.423790183);

  // y = x + 1
  SGD sgd;
  for (int i = 0; i <= 5000; ++i) {
    zero_grad(w, b);
    auto z = variable(0.0);
    for (float xv = 0.0; xv < 32.0; xv += 1.0) {
      auto x = variable(xv);
      auto y = variable(xv + 1.0f);
      x->set_requires_grad(false);
      y->set_requires_grad(false);
      auto predicted = w * x + b;
      z = z + mse_loss(predicted, y);
    }
    auto batch_size = variable(32.0);
    z = z / batch_size;

    autograd::run_backward(*z);
    sgd.step(w, b);

    if (i % 1000 == 0) {
      BOOST_LOG_TRIVIAL(debug)
          << fmt::format("Iter = {}, Loss = {}", i, z->value_);
    }
  }
  BOOST_LOG_TRIVIAL(debug) << fmt::format("w = {}, b = {}", w->value_,
                                          b->value_);
  ASSERT_NEAR(w->value_, 1.0, 1e-3);
  ASSERT_NEAR(b->value_, 1.0, 1e-3);
}

struct XORNet {
  void _zero_grad() {
    zero_grad(layer1);
    zero_grad(layer2);
  }
  std::shared_ptr<Variable> forward(std::shared_ptr<Variable> x1,
                                    std::shared_ptr<Variable> x2) {
    std::shared_ptr<Variable> layer1_output[3] = {
        (layer1[0] * x1 + layer1[1] * x2 + layer1[2])->sigmoid(),
        (layer1[3] * x1 + layer1[4] * x2 + layer1[5])->sigmoid(),
        (layer1[6] * x1 + layer1[7] * x2 + layer1[8])->sigmoid(),
    };

    std::shared_ptr<Variable> layer2_output =
        (layer2[0] * layer1_output[0] + layer2[1] * layer1_output[1] +
         layer2[2] * layer1_output[2] + layer2[3])
            ->sigmoid();

    return layer2_output;
  }
  std::shared_ptr<Variable> layer1[2 * 3 + 3] = {
      variable(0.71423874), variable(-0.2349723), variable(0.32478342),
      variable(-0.234782),  variable(0.21328192), variable(-0.2389934),
      variable(0.234782),   variable(0.51328292), variable(0.81328192)};
  std::shared_ptr<Variable> layer2[3 * 1 + 1] = {
      variable(0.41328192), variable(-0.2389934), variable(-0.5349832),
      variable(-0.2349832)};
};

std::shared_ptr<Variable> bce_loss(std::shared_ptr<Variable> predicted,
                                   std::shared_ptr<Variable> target) {
  auto one = variable(1.0f)->detach();
  auto eps = variable(1e-7f)->detach();
  return -(target * (predicted + eps)->log()) -
         ((one - target) * (one - predicted + eps)->log());
}

TEST(Integration, XORNet_BCELoss) {
  std::shared_ptr<Variable> x[4 * 2] = {
      variable(0.0f), variable(0.0f), variable(1.0f), variable(0.0f),
      variable(0.0f), variable(1.0f), variable(1.0f), variable(1.0f),
  };
  std::shared_ptr<Variable> y[4] = {
      variable(0.0f),
      variable(1.0f),
      variable(1.0f),
      variable(0.0f),
  };
  auto one = variable(1.0f);

  SGD sgd;
  XORNet model;
  sgd.learning_rate_ = 0.5;
  for (int i = 0; i < 5000; ++i) {
    model._zero_grad();
    auto loss = variable(0.0f);
    for (int b = 0; b < 4; ++b) {
      auto output = model.forward(x[b * 2], x[b * 2 + 1]);
      loss = loss + bce_loss(output, y[b]);
    }
    auto batch_size = variable(4.0f);

    loss = loss / batch_size;
    autograd::run_backward(*loss);

    sgd.step(model.layer1);
    sgd.step(model.layer2);

    if (i % 1000 == 0) {
      BOOST_LOG_TRIVIAL(debug)
          << fmt::format("Iter = {}, Loss = {}", i, loss->value_);
    }
  }
  
  for (int b = 0; b < 4; ++b) {
    auto output = model.forward(x[b * 2], x[b * 2 + 1]);
    ASSERT_NEAR(output->value_, y[b]->value_, 1e-2);
    BOOST_LOG_TRIVIAL(debug)
        << fmt::format("xor({}, {}) = {}", x[b * 2]->value_,
                       x[b * 2 + 1]->value_, output->value_);
  }
}
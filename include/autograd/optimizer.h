#if !defined(__OPTIMIZER_H__)
#define __OPTIMIZER_H__

template <class... Variables> void zero_grad() { return; }

template <class T, class... Variables>
void zero_grad(T variable, Variables... variables) {
  variable->grad_ = 0.0;
  zero_grad(variables...);
}

class SGD {
public:
  float learning_rate_ = 0.003;

  template <class... Variables> void step() { return; }

  template <class T, class... Variables>
  void step(T variable, Variables... variables) {
    variable->value_ -= learning_rate_ * variable->grad_;
    step(variables...);
  }
};

#endif // __OPTIMIZER_H__

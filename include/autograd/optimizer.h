#if !defined(__OPTIMIZER_H__)
#define __OPTIMIZER_H__

template <class... Variables> void zero_grad() { return; }

template <class T, class... Variables>
void zero_grad(T variable, Variables... variables) {
  variable->grad_ = 0.0f;
  zero_grad(variables...);
}

template <class T, size_t N>
void zero_grad(T (&variables)[N]) {
    for (int i = 0; i < N; ++i) {
        variables[i]->grad_ = 0.0f;
    }
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

  template <class T, size_t N>
  void step(T (&variables)[N]) {
      for (int i = 0; i < N; ++i) {
          variables[i]->value_ -= learning_rate_ * variables[i]->grad_;
      }
  }
};

#endif // __OPTIMIZER_H__

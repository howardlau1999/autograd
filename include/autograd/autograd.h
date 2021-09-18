#if !defined(__AUTOGRAD_H__)
#define __AUTOGRAD_H__

#include "autograd/variable.h"
#include <boost/log/trivial.hpp>
#include <memory>
#include <vector>

namespace autograd {

using variable_list = std::vector<Variable>;
using edge_list = std::vector<Edge>;

class Node : public std::enable_shared_from_this<Node> {
  Node(Node const &) = delete;
  Node(Node &&) = delete;
  Node &operator=(Node const &) = delete;
  edge_list next_edges_;
  int input_nr_ = 0;

public:
  Node() = default;
  virtual ~Node() = default;
  virtual const char *name() { return typeid(*this).name(); }
  void add_next_edge(Edge &&edge) { next_edges_.push_back(edge); }
  int next_edges() { return next_edges_.size(); }
  int input_nr() { return input_nr_; }
  int add_input_nr() { ++input_nr_; }
  Edge next_edge(int i) { return next_edges_[i]; }
  virtual variable_list apply(variable_list &&variables) {}
};

void run_backward(Variable &root);
void print_graph(Variable &root);

} // namespace autograd

#endif // __AUTOGRAD_H__

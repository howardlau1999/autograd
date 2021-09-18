#include <autograd/autograd.h>
#include <queue>
#include <unordered_map>
#include <boost/log/trivial.hpp>

namespace autograd {

struct NodeTask {
  std::shared_ptr<Node> fn;
  variable_list variables;
};

void compute_dependencies(
    Variable &root,
    std::unordered_map<std::shared_ptr<Node>, int> &dependencies) {
  std::queue<std::shared_ptr<Node>> queue;
  queue.push(root.gradient_edge().grad_fn());
  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop();
    if (!node) {
      continue;
    }
    for (unsigned int i = 0; i < node->next_edges(); ++i) {
      auto edge = node->next_edge(i);
      auto grad_fn = edge.grad_fn();
      if (grad_fn) {
        dependencies[grad_fn] += 1;
        queue.push(grad_fn);
      }
    }
  }
}

void run_backward(Variable &root) {
  std::unordered_map<std::shared_ptr<Node>, int> dependencies;
  compute_dependencies(root, dependencies);
  Variable one(1.0);
  std::queue<NodeTask> queue;
  std::unordered_map<std::shared_ptr<Node>, NodeTask> not_ready;
  queue.push({root.gradient_edge().grad_fn(), {one}});
  while (!queue.empty()) {
    auto task = queue.front();
    queue.pop();
    auto outputs = task.fn->apply(std::move(task.variables));
    for (int i = 0; i < outputs.size(); ++i) {
      auto edge = task.fn->next_edge(i);
      auto fn = edge.grad_fn();
      if (!fn) continue;
      auto it = dependencies.find(fn);

      bool is_ready = false;
      if (it == dependencies.end()) {
        throw std::runtime_error("Dependency not found");
      } else if (--it->second == 0) {
        is_ready = true;
        dependencies.erase(it);
      }

      auto not_ready_it = not_ready.find(fn);
      if (not_ready_it == not_ready.end()) {
        variable_list inputs(fn->input_nr());
        inputs[edge.input_nr()].value_ += outputs[i].value_;
        if (is_ready) {
          queue.push({fn, std::move(inputs)});
        } else {
          not_ready[fn] = NodeTask{fn, std::move(inputs)};
        }
      } else {
        (not_ready_it->second).variables[edge.input_nr()].value_ += outputs[i].value_;
        if (is_ready) {
          queue.push(std::move(not_ready_it->second));
          not_ready.erase(not_ready_it);
        }
      }
    }
  }
}

} // namespace autograd
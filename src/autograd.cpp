#include <autograd/autograd.h>
#include <boost/algorithm/string/join.hpp>
#include <boost/log/trivial.hpp>
#include <cxxabi.h>
#include <fmt/format.h>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>

namespace autograd {

struct NodeTask {
  std::shared_ptr<Node> fn;
  variable_list variables;
};

void print_graph(Variable &root) {
  std::unordered_map<std::shared_ptr<Node>, bool> visited;
  std::queue<std::shared_ptr<Node>> queue;
  std::unordered_map<std::shared_ptr<Node>, std::string> node_names;
  std::unordered_map<std::shared_ptr<Node>, std::vector<std::string>>
      neighbours;

  auto get_node_name = [&](std::shared_ptr<Node> n) {
    auto it = node_names.find(n);
    if (it != node_names.end()) {
      return it->second;
    }
    char *buf =
        __cxxabiv1::__cxa_demangle(n->name(), nullptr, nullptr, nullptr);
    std::string&& name = fmt::format("{}_{}", buf + 10, fmt::ptr(n));
    free(buf);
    return node_names[n] = std::move(name);
  };

  queue.push(root.gradient_edge().grad_fn());
  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop();
    if (!node) {
      continue;
    }

    (void)get_node_name(node);

    for (int i = 0; i < node->next_edges(); ++i) {
      auto edge = node->next_edge(i);
      auto grad_fn = edge.grad_fn();
      if (grad_fn) {
        neighbours[node].push_back(get_node_name(grad_fn));
        if (!visited[grad_fn]) {
          visited[grad_fn] = true;
          queue.push(grad_fn);
        }
      }
    }
  }
  std::cout << "digraph {" << std::endl << std::endl;
  for (auto &[p, s] : node_names) {
    std::cout << "  " << s << std::endl;
  }
  std::cout << std::endl;
  for (auto &[p, ns] : neighbours) {
    fmt::print("  {} -> {{{}}}\n", get_node_name(p),
               boost::algorithm::join(ns, " "));
  }
  std::cout << std::endl << "}" << std::endl;
}

void compute_dependencies(
    Variable &root,
    std::unordered_map<std::shared_ptr<Node>, int> &dependencies) {
  std::queue<std::shared_ptr<Node>> queue;
  std::unordered_map<std::shared_ptr<Node>, bool> visited;
  queue.push(root.gradient_edge().grad_fn());
  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop();
    if (!node) {
      continue;
    }

    for (int i = 0; i < node->next_edges(); ++i) {
      auto edge = node->next_edge(i);
      auto grad_fn = edge.grad_fn();
      if (grad_fn) {
        dependencies[grad_fn] += 1;
        if (!visited[grad_fn]) {
          visited[grad_fn] = true;
          queue.push(grad_fn);
        }
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
    for (unsigned int i = 0; i < outputs.size(); ++i) {
      auto edge = task.fn->next_edge(i);
      auto fn = edge.grad_fn();
      if (!fn)
        continue;
      auto it = dependencies.find(fn);

      bool is_ready = false;
      if (it == dependencies.end()) {
        throw std::runtime_error("Dependency not found");
      } else {
        if (--it->second == 0) {
          is_ready = true;
          dependencies.erase(it);
        }
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
        (not_ready_it->second).variables[edge.input_nr()].value_ +=
            outputs[i].value_;
        if (is_ready) {
          queue.push(std::move(not_ready_it->second));
          not_ready.erase(not_ready_it);
        }
      }
    }
  }
  if (!not_ready.empty()) {
    throw std::runtime_error("Some tasks are not finished");
  }
}

} // namespace autograd
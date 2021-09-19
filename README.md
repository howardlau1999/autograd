简单地演示了 PyTorch 中自动求导机制的原理。

## 编译运行

使用 [Bazel](https://docs.bazel.build/versions/4.2.1/install.html)

```bash
bazel run autograd_test
```

包含了一个使用 MSE 损失函数的一次直线的线性回归，以及一个使用 BCE 损失函数的带有 sigmoid 激活函数的两层非线性 XOR 网络。

## API

`autograd::run_backward(Variable& root)`: 以 root 为根节点，以拓扑排序进行一次反向传播。

`autograd::print_graph(Variable& root)`: 以 root 为根节点，打印出 `dot` 格式的计算图，可以使用 `graphviz` 进行可视化。

## 文件内容

`src/autograd.cpp`：反向传播 API，根据反向计算图进行拓扑排序并计算梯度。

`src/operators.cpp`：反向算子，例如 `AddBackward` 等。

`src/variable.cpp`：存储值和梯度的变量，是对 `float` 的包装。

## 数据结构

`sutrct Edge`: 计算图的边，保存了指向的终点 `grad_fn_` 以及在起点的出边中的顺序 `input_nr_`。

`struct Node`: 计算图的节点，算子的基类，其中 `variable_list apply(variable_list)` 函数接收变量列表，进行运算，并返回一个变量列表。`next_edge(int i)` 返回反向计算图中该节点的一条出边。

`class Variable`: 实际保存运算值的类。

## PyTorch 的实现细节

1. 构造反向计算图的时候，如果在前向运算中，所有的操作数都不需要梯度（requires_grad = false），则不构建反向计算节点。
2. 反向传播过程中，如果一条边不需要梯度，就直接返回空的变量。
3. 反向传播过程中，会先使用一个线程进行递归遍历，递归深度过大的时候会启动多个线程。
4. 反向传播的起点支持 Jacobian 矩阵输入（比较少用到）。
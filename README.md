## 编译运行

```bash
bazel run autograd_test
```

## 数据结构

`sutrct Edge`: 计算图的边，保存了指向的终点 `grad_fn_` 以及在起点的出边中的顺序 `input_nr_`。

`struct Node`: 计算图的节点，算子的基类，其中 `variable_list apply(variable_list)` 函数接收变量列表，进行运算，并返回一个变量列表。`next_edge(int i)` 返回反向计算图中该节点的一条边。

`class Variable`: 实际保存运算值的类。

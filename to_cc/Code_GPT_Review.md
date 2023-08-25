# Code

## Analysis_Cluster

- [ ] 1

这段代码定义了两个主要的类：bub_instant 和 brain，它们对应表示 "泡"（可能是指一种医学成像中的特定区域或结构）和大脑。

bub_instant 对象代表了一个特定的 "泡"。初始化函数__init__创建一个 "泡" 的实例，为其分配一系列的属性，包括它的空间位置(coordinate)，ID，嵌入向量(emb)，相邻矩阵(adj_mat)等。还包含了对3-hop特性的提取，每个 "泡" 的标签(roi_label)，每个 "泡" 的粗糙度(sulc)，曲率(curv)等。
__repr__方法定义了当你打印一个 "泡" 对象的时候会返回什么，这里它返回的是一个包含一些关键属性的JSON字符串。
find_neighber方法用于找出一个 "泡" 对象的邻居 "泡"。
get_fiber_vtk, get_bubble_vtk ,get_fpoints_vtk方法用来从给定的路径获取相关的vtk文件。

brain 对象代表了一个特定的大脑。它的初始化函数在创建一个大脑实例时为其分配一系列的属性，如大脑对应的学类别id，白质表面，纤维，等等。

__repr__方法定义了当你尝试打印一个大脑对象时会返回什么。这里它返回的是一个包含一些关键属性的JSON字符串。
get_lh_vtk, get_rh_vtk, get_allROIs_vtk, get_fiber_vtk方法, 用于从给定的路径获取相关的vtk文件。
get_ratio方法用于获取所有 "泡" 和良好 "泡" 的比例。
check_duplicon方法用于检查是否存在重复的 "泡" 。
总的来说，这段代码对 "泡" （bub_instant）和大脑（brain）进行了抽象，定义了他们的类别，并为每个类别定义了一些属性和方法。在神经影像学领域，这种设计模式可以帮助处理和分析复杂的大脑影像数据。
### Brain_and_Hings.py

### BrainLibrary.py

### Collect_ALL.py

### LabelROI_HCP_june2022.py

### statistic_common_3hinges_2023_ap.py

## Model_3H

## trace_map_script

# VTK_code

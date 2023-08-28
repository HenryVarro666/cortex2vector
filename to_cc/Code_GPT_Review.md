# Code

## Analysis_Cluster
### Brain_and_Hings.py
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
### BrainLibrary.py

### Collect_ALL.py

- [ ] 1
这段 Python 代码实现了一个脑成像数据处理流程。具体来说，它主要包含以下步骤：

代码首先定义了一些函数，用于读取 vtk 文件，计算坐标，绘制气泡（代表脑中的某个区域），并收集颜色信息。

_collect_init函数根据输入的头皮数据文件目录和邻接特征矩阵目录，读取这些目录下的文件，并根据所读取的数据在每个图像的指定区域绘制气泡。

_collect_brains函数遍历数据目录，为每一个子目录（即每一个病人的脑图像数据）创建一个 brain 对象，并将这些 brain 对象以字典的形式返回。

_collect_next函数定位到每个气泡的坐标，为这些气泡生成新的特征（如嵌入、邻接矩阵、连通特征等），并同样以字典的形式返回。

主程序部分则是按照一定的顺序执行以上步骤。其中，用户可以通过输入来决定执行哪些步骤。

总的来说，该程序的作用是对脑成像数据进行处理，以便于后续的数据分析或机器学习任务。

### LabelROI_HCP_june2022.py

### statistic_common_3hinges_2023_ap.py

- [ ] 1
这段代码通过聚类方法实现特征向量的共享属性分析。代码可以分为以下几部分：

定义一系列工具函数来读取输入数据、生成需要聚类的数据和准备其它的相关数据。

聚类分析部分，使用能够度量输入特征向量间相似度的层次聚类方法。agglomerative clustering和affinity propagation（注释掉了，AP的方法并未使用）。

主函数部分，根据指定的主题，从输入路径中读取与主题相关的数据，对每个输入的特征向量类型进行聚类分析，并保存聚类结果和模型信息。

命令行参数解析部分，定义了一系列的命令行参数，用于指定数据的路径、输出的路径、聚类参数等。这些参数可以在运行脚本时动态指定。

这个脚本运行时将根据聚类参数的设定来对图像嵌入进行聚类分析，并将结果保存以供后续处理。

## Model_3H

### Graph_embedding_calculate_results.py  
### Graph_embedding_get_loss.py             

### dataloader_embedding.py

### Graph_embedding_get_model_parameters.py 

### hop_2
The folder stores model weights
### Graph_embedding_loss.py                 

### sampler.py
### Graph_embedding_main_initial_eye.py

- [ ] 1

这个脚本是一个使用 PyTorch 深度学习框架的训练和评估模型的脚本。主要包括以下功能：

argparse 用于解析命令行参数。
torch 是 PyTorch 框架，提供了构建神经网络的所有必要组件。
dataloader_embedding、Graph_embedding_model_initial_eye、utils 是其他自定义脚本的引用。
SummaryWriter 用于将数据写入 TensorBoard，方便可视化。
pdb 是 Python 的调试库。
该脚本定义了七个函数：main, train, validate, test 和带参数的入口函数。

main 函数处理数据和目录，设置 GPU 为训练设备，并对数据进行多次 (10 次) 迭代训练、验证和测试。每次迭代 (称作一个实验) 会创建它自己的训练、验证和测试结果目录。如果预训练模型可用，它将被加载到模型中。每一个 epoch 都会训练并验证模型，优化器的学习率会根据验证损失相应地进行调整。如果验证损失小于当前最佳的，则模型会被保存。最后，会在每个实验结束时运行测试，并删除模型、优化器等内存占用的对象。

train 函数训练模型。输入的数据是一个批量的 multi-hot 特征，然后将其馈送到模型计算损失，执行反向传播和优化步骤，并记录平均损失和批处理时间，每步都会打印日志。

validate 函数用于评估模型在验证数据集上的性能。计算损失，并返回平均损失。

test 函数用于在测试集上检验模型性能，它从指定目录中加载模型并将其设置为评估模式，然后利用测试数据进行评估，保存嵌入、解码等结果。

带参数的入口函数处，解析了许多命令行参数，包括文件路径、训练参数、预训练模型、GPU ID 等。针对每一个预训练模型都会运行 main 函数进行训练。

- [ ] 2

这段Python代码是创建并训练一种图嵌入模型的脚本。主要包含以下部分：

预处理： 一些导入和初始设置，主要包含对运行环境的配置和需要使用的库的导入。包括torch，argparse和time等。

主函数（main）： 这个函数是整个脚本的主体，它接收由argparse库解析的命令行参数，然后进行模型创建、训练、验证和测试等步骤。

辅助函数（train, validate, test）： 这些函数被主函数调用以完成模型的训练、验证和测试步骤。

train 函数：接受数据加载器、模型、优化器、写入器、时期和设备等参数，然后训练模型，并将结果写入TensorBoard日志。

validate 函数：它接受和 train 相同的参数，并对模型进行验证，我们会得到一个平均的验证损失。

test 函数：加载已训练的模型并进行测试，它会測試模型对新数据的预测能力，测试结果将被保存，用于比较不同模型或参数设置的效果。

命令行参数解析： 使用argparse库解析命令行参数。
在主函数中，首先创建了必要的目录结构以存储模型、训练结果、验证结果和测试结果。然后，通过args快速加载命令行参数，包括学习率、批次大小、嵌入维度等。接下来构建和训练模型，并在每个epoch后验证模型，然后在所有epoch完成后测试模型，记录和打印测试结果。

在执行这份脚本时，你需要提供一些命令行参数，例如数据路径、模型存储路径、日志路径等。
### Graph_embedding_model_initial_eye.py

- [ ] 1

这段Python代码定义了两个神经网络模型，graph_embedding 和 graph_embedding2，它们都是用来对图结构数据进行嵌入（embedding）的。两个模型的目标都是将输入的多热标签 (multi-hot-code，通常用于表示对象存在于多个类别中) 进行嵌入和解码。

下面是两个模型的详细解释：

graph_embedding：

初始化函数__init__中，定义了一些网络层，包括一个线性层w_node_embedding，用于将输入转化成嵌入向量，以及embedding_combine_coef，decoder_divide_coef和w_node_decoder用于解码嵌入向量。

forward方法首先将输入转化为嵌入形式然后使用已训练参数将它解码，得到的x_decoder看做是对原始输入的预测，而x_embedding和x_de_embedding用于计算损失。

graph_embedding2：

同graph_embedding类似，不过在嵌入和解嵌入过程中引入了激活函数F.leaky_relu，并且在嵌入层w_node_embedding的初始化过程中，使用单位矩阵进行初始化。

在主函数main中，首先确保了可用的设备（如果有GPU则使用，否则使用CPU）。然后，生成了一些随机的多热特征用于训练。在每个epoch中，通过优化器（这里使用Adam优化器）计算模型的前向过程并得到损失，然后通过反向传播计算模型参数的梯度，并更新参数。

这种类型的模型通常在一些需要处理图结构数据的任务中使用，例如社交网络分析、蛋白质网络、交通网络分析等。

### utils.py
## trace_map_script

## VTK_code

### Combine_ROIs_all.py  

- [ ] 1 
这段Python代码主要用于处理列入名单和预设例外的vtk（VTK是一种实用的3D计算机图形学、图像处理和可视化的开源软件库）文件。主要包含以下几个函数：

read_vtk_file(vtk_file)：读取给定vtk文件，并返回包含其多维数据的对象。

combine_rois(root_dir, roi_list, except_list, output_dir, name)：此函数接收一个目录root_dir，用于读取roi_list中列出的vtk文件。对于那些不包含预设例外except_list的vtk文件，该函数会将它们的内容合并，其中主要包含一系列的点和多边形，并将这些信息写入一个新的vtk文件，这个文件将被保存在output_dir下并以给定的name命名。

combine_rois_colorful(root_dir, roi_list, except_list, output_dir, name)：这个函数实现的功能类似于combine_rois，但是该函数在合并vtk文件的过程中保留了每个文件的颜色信息。

delete_roi_from_whole(surf_root, surf, label_root, roi_list, except_list, output_dir, name)：此函数用于删除某些vtk数据。它首先读取了一组预设的vtk文件，然后从这些文件中删除了except_list列出的vtk文件中存在的多边形，最后将结果写入一个新的vtk文件。

在程序的主函数部分，它首先获取了名单中的所有用户（或试验定位），然后在每个用户中取得名单与例外清单，并调用上述函数进行处理，处理的结果写入一个新的vtk文件中。
### New_Color_fib_v3.py  

- [ ] 1

这段 Python 脚本的主要功能是从一种脑磁共振成像（Magnetic Resonance Imaging, MRI）数据格式（.trk）中读取轨迹，然后将这些轨迹转化成另一种格式（.vtk）。这个脚本使用了 vtk, nibabel 等包来帮助读取、处理和写入这种图片和几何数据。

主要函数：

tractography_from_trackvis_file(filename)：该函数从给定的脑磁共振成像数据文件 (.trk 格式) 中读取轨迹数据。

assign_color(fib)：这个函数是给给定的轨迹上的点分配颜色的函数。

calculate_points_color_for(fiber_coordinate_list, p_id_list)：计算一根轨迹上各个点在两个平行平面之间的颜色，返回一个字典，其中的键为点的 id，值为对应的颜色。

extend_fib(fib): 对来自轨迹文件的轨迹（fiber）进行扩展，防止喷射错误。

read_tracts(tracts): 从输入的轨迹中计算出各个点的坐标和颜色。

calculate_coordinate_in_dti_space_for_DStudio_fiber(coordinate_in_DStudio_list, t1_hdr): 从原始的轨迹空间重新映射到 DTI 空间。

主函数中，首先读取命令行输入的文件名，然后使用定义好的一些函数来处理轨迹数据，最后写入新的 vtk 文件。此脚本的主要目的是数据格式的转换和数据的重新映射，使数据符合 vtk 格式以及 DTI 空间。
### Plot_many_bubbles.py 

- [ ] 1
这段Python代码联合多个 vtk 文件并保存为新的 vtk 文件。代码主要分为以下部分：

导入所需库：包括 scipy（用于科学计算），numpy（用于数组操作），os（用于操作系统相关的任务），shutil（用于文件操作），json（用于处理 JSON 数据）， openpyxl 和 pandas（用于 Excel操作）等等。并且，从VTK相关模块中导入了大量的类和方法。

HTMLToFromRGBAColor类：一个转换颜色格式的静态类，包含将RGB颜色转换为HTML颜色、将HTML颜色转换为RGB颜色，并将RGB颜色转换为亮度等方法。

ReadPolyData函数：根据文件扩展名，使用相关的VTK reader来读取并更新polydata。这个函数返回一个对应文件扩展名的reader对象。

read_vtk_file函数：使用vtkPolyDataReader读取vtk文件并更新其中的元数据。这个函数返回一个polydata对象。

combine_multi_v函数：这是主要的处理函数，输入一系列的文件名，并且对每个文件进行预处理、合并操作，然后写入新的vtk文件。

主函数：在主函数中，首先打开并加载了两个pickle文件，包含了brains数据和hings数据。然后对每个brain数据进行处理，获取其所有的ROI相关的vtk文件，并且与对应的hing vtk文件进行合并，最终写入新的vtk文件。

这是很具体的代码，取决于你的目标和使用场景，可能要对其进行一些修改。例如，你可能需要更改文件路径，或者选择读取和写入不同格式的文件。使用VTK库进行科学计算和可视化是一个非常强大的工具，你可以查看VTK的官方文档，了解更多你可以使用的功能。


### Power_view_alpha.py

- [ ] 1

Power_view_alpha.py是一个用于并行处理、分析、可视化各种格式的神经成像数据的Python脚本。

`Power_view_alpha.py`是一个Python脚本，使用Python3执行。它似乎用于处理医学图像，特别是神经成像数据。它的一部分是对内外源代码文件的引入，像`numpy`、`nibabel`，它们在科学计算和图像处理中非常常见。

该脚本有许多辅助功能，许多功能都是针对特定的输入数据格式或类型所设计的。显然，此代码的主要目标是三维图像数据，特殊的数据类型包括`trackvis`数据（这是一种在神经科学研究中广泛使用的数据格式），还有`vtk`数据，这是一种通用的图像分割和渲染格式。其他辅助函数包括进行颜色分配和坐标变换的功能。

代码中还包含一些辅助功能，用于验证数据矩阵的属性，包括正定性和厄米半正定性。

此外，`Power_view_alpha.py`还包含对线程的使用，这意味着它可能会利用多核处理器执行并行运算以提高效率。其中一些功能也包含了对底层数据结构的深入理解，这使得这个脚本能更精确地针对特定的数据格式或类型进行操作。

但需要注意的是，这个脚本代码中有一些函数被注释掉了，可能是作者在开发和测试过程中暂时禁用的。
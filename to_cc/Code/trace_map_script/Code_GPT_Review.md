## create_trace_map.sh

这是一个Bash脚本用于在目录 $root_dir/$target_dir 中的每一个文件执行特定的操作。以下是对这段代码的详细解释：

首先，这个脚本定义了两个目录变量，root_dir 和 target_dir。将其组合后，您将得到全路径。

然后，脚本通过ls $root_dir/$target_dir获取目录下所有的文件名，并将其保存在filenames变量中。

随后，脚本开始一个循环，将对目录下的每个文件名（在此循环中记为filename）执行一系列操作：

echo $filename: 在控制台输出当前处理的文件名。

使用vtkBit2Acs命令行工具（位于/media/zhulab/disk_h/HCP/code/），将源文件 fiber.vtk 转换为 fiber_asc.vtk。源文件位于 original_data 的子目录下。

使用TraceMapGenerator.jar（位于/media/zhulab/disk_h/HCP/trace_map/），对转换后的fiber_asc.vtk文件及相关的lh.white_transform_change.vtk文件进行处理，默认输出结果到$root_dir/LinZhao_300_data/${filename}_lh的文件中。并制定另一组相关文件作为参数输入：${filename}_3hinge_ids_lh.txt。

再次使用TraceMapGenerator.jar，这次对fiber_asc.vtk和rh.white_transform_change.vtk文件进行处理，同样指定相关文件作为参数，并将输出结果默认保存到$root_dir/LinZhao_300_data/${filename}_rh中。

注意：TraceMapGenerator.jar后面的参数和操作可能需要参考相关文档以理解，以上解释只是基于参数的命名和常理进行的推断。例如，-s可能代表源文件，-f可能指的是待处理的文件，-pl可能指的是输入路径，-o可能指的是输出路径，-ov可能是某个特殊的操作标志。再次强调，具体含义需要参考到该.jar文件的使用说明。

## create_tace_map_local.sh
这是一个Bash脚本，用于在$target_dir目录下的每个文件或子目录进行特定的操作。以下是对这段代码的详细解释：

首先，这个脚本定义了目标目录变量target_dir, 并且获取该目录下的所有文件或子目录名称列表filenames。

然后，脚本开始一个循环，对列表中的每个文件名或子目录名进行以下操作：

echo $filename: 在控制台上打印当前处理的文件或子目录名。

#开始的一行是被注释掉的代码，脚本在执行时会忽略这一行。

rm -rf $target_dir/$filename/ft: 删除名为ft的目录及其内部所有文件，这个目录在$target_dir/$filename路径下。

mkdir $target_dir/$filename/ft: 在同样的路径下创建新的名为ft的目录。

在接下来的代码中，使用Java jar文件（位于/mnt/disk1/HCP_luzhang_do/Analysis/trace_map_script/路径下的TraceMapGenerator.jar）来完成对lh_fix.white.vtk和fiber_fix.vtk文件的处理，并将结果输出到${filename}_lh文件中。此时，还指定了相关文件lh_collect.txt作为输入，并在操作结束后删除原文件（由-ov标志指定）和输出文件（由-of标志指定）。

再次运行相同的Java jar文件，这次处理rh_fix.white.vtk和fiber_fix.vtk文件，并将结果输出到${filename}_rh文件中。同样，指定了rh_collect.txt作为输入，并在操作结束后删除原文件和输出文件。

同样地，对于Java jar命令行的参数详解可能需要参考相关文档以获得，以上解释主要是基于参数的名称和常规理解进行的猜测。例如，-s可能代表源文件，-f可能代表的是待处理的文件，-pl可能代表输入路径，-o可能代表输出路径，-ov，-of和-op可能是操作标志。强调一下，具体参数含义需要查阅相关文档或工具的使用说明才能准确知晓。
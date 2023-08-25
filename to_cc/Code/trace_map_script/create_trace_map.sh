#!/bin/bash

root_dir=/media/zhulab/disk_h/HCP
target_dir=HCP_sphere_GyralNet


filenames=$(ls $root_dir/$target_dir)

for filename in $filenames; 
do

echo $filename

/media/zhulab/disk_h/HCP/code/vtkBit2Acs $root_dir/$target_dir/$filename/original_data/fiber.vtk $root_dir/$target_dir/$filename/original_data/fiber_asc.vtk

java -jar /media/zhulab/disk_h/HCP/trace_map/TraceMapGenerator.jar -s $root_dir/$target_dir/$filename/original_data/lh.white_transform_change.vtk -f $root_dir/$target_dir/$filename/original_data/fiber_asc.vtk -pl $root_dir/LinZhao_300_data/${filename}_3hinge_ids_lh.txt -o $root_dir/LinZhao_300_data/${filename}_lh -ov

java -jar /media/zhulab/disk_h/HCP/trace_map/TraceMapGenerator.jar -s $root_dir/$target_dir/$filename/original_data/rh.white_transform_change.vtk -f $root_dir/$target_dir/$filename/original_data/fiber_asc.vtk -pl $root_dir/LinZhao_300_data/${filename}_3hinge_ids_rh.txt -o $root_dir/LinZhao_300_data/${filename}_rh -ov

done


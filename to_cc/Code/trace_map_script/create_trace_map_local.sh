#!/bin/bash

target_dir=./Test1

filenames=$(ls $target_dir)

for filename in $filenames; 
do

echo $filename

#/mnt/disk2/work_2022_3/Threehhh/trace_map_script/vtkBit2Acs $target_dir/$filename/output/track_ras_extend_color.vtk $target_dir/$filename/output/fiber_asc.vtk
rm -rf $target_dir/$filename/ft
mkdir $target_dir/$filename/ft
java -jar /mnt/disk1/HCP_luzhang_do/Analysis/trace_map_script/TraceMapGenerator.jar -r 2 -s $target_dir/$filename/Surf/lh_fix.white.vtk -f $target_dir/$filename/fib/fiber_fix.vtk -pl $target_dir/$filename/lh_collect.txt -o $target_dir/$filename/ft/${filename}_lh -ov -of -op
java -jar /mnt/disk1/HCP_luzhang_do/Analysis/trace_map_script/TraceMapGenerator.jar -r 2 -s $target_dir/$filename/Surf/rh_fix.white.vtk -f $target_dir/$filename/fib/fiber_fix.vtk -pl $target_dir/$filename/rh_collect.txt -o $target_dir/$filename/ft/${filename}_rh -ov -of -op
#java -jar /mnt/zhulab/disk_h/HCP/trace_map/TraceMapGenerator.jar -s $root_dir/$target_dir/$filename/original_data/rh.white_transform_change.vtk -f $root_dir/$target_dir/$filename/original_data/fiber_asc.vtk -pl $root_dir/LinZhao_300_data/${filename}_3hinge_ids_rh.txt -o $root_dir/LinZhao_300_data/${filename}_rh -ov

done


#Readme

# Files:
/mnt/disk1/HCP_luzhang_do/ADNI_run/results/local_work_adni
/exp_4 -- Embedding results

/mnt/disk1/HCP_luzhang_do/ADNI_run/make_data/Graph_embedding_data_adni
/Graph_embedding_data_adni/adj_feature_matrix -- 3hing_ids list; 3hing adj; 3 hing 0hop features
/Graph_embedding_data_adni/multi_hop_feature_matrix -- 3hings 1,2,3 hop features
/Graph_embedding_data_adni/node_input_data -- Input data for each nodes

/mnt/disk2/_Proposal_data_2020/ADNI_3H/ADNI_GyralNet_files
-- excle files stores 3hings features of ADNI

/mnt/disk2/_Proposal_data_2020/ADNI_3H/ADNI_Gyralnet_surf

#/mnt/disk1/HCP_luzhang_do/Analysis/HCP_Surf_Gyralnet_3hinge
#Subjects/ 


# Running Steps:

1, In make_data/
(need labels.txt)
local_create_3hinge_adj_feature_matrix.py
local_create_3hinge_node_input.py

from 3H excel files to generate model input

2, In model/
local_Graph_embedding_main_initial_eye_test_only.py

run model, test inference only
load models' weights from /hop2/

results in /mnt/disk1/HCP_luzhang_do/ADNI_run/results/local_work_adni/testing_results/exp_4

3, Do AG Clusters

/mnt/disk1/HCP_luzhang_do/ADNI_run/Calculate_2023_adni/statistic_common_3hinges.py

saved data in /mnt/disk1/HCP_luzhang_do/ADNI_run/results/local_work_adni

4, Collect results, ploting and saved to pickle file

/mnt/disk1/HCP_luzhang_do/ADNI_run/Calculate_2023_adni/Collect_ALL.py

Actions: 0 or 1:
0: _collect_init
05: manually do fiber trace map 
1: _collect_brains saved everthings into Dictionary




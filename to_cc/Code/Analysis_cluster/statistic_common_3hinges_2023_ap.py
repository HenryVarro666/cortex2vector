import argparse
import os
import pdb
from collections import defaultdict
import numpy as np
import time
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import shutil

'''
============================================================== Vector ==========================================================
'''

# def get_3hinge_name_data(path):
#     print('======================== get_3hinge_name_data \t' + time.asctime(time.localtime(time.time())) + '=============================')
#     name_data_dict = defaultdict(list)
#     file_list = [file for file in os.listdir(path)]
#     for file in file_list:
#         name = file.split('.')[0]
#         data = np.loadtxt(path + '/' + file)
#         # data = np.where(data > 0, 1.0, 0.0)
#         # pdb.set_trace()
#         data = data[2:]
#         if name not in name_data_dict:
#             name_data_dict[name] = [data]
#         else:
#             print('3hinge exist!')
#             exit()
#     return name_data_dict


def judge_3hinge_exist(hinge_type_name_dict, current_data):
    exist = 0
    position = -1
    for index in hinge_type_name_dict.keys():
        data = hinge_type_name_dict[index][0][0]
        if (data==current_data).all():
            exist = 1
            position = index
            break
    return exist, position


def get_all_kinds_of_3hinge(name_data_dict):
    print('======================== get_all_kinds_of_3hinge \t' + time.asctime(time.localtime(time.time())) + '=============================')
    hinge_type_name_dict = defaultdict(list)
    i = 0
    num = 0
    print('   :<---', end='\r')
    for name in name_data_dict.keys():
        print(str(i), end='\r')
        i += 1
        data = name_data_dict[name][0]
        exist, position = judge_3hinge_exist(hinge_type_name_dict, data)
        if exist:
            hinge_type_name_dict[position][1].append(name)
        else:
            hinge_type_name_dict[num] = [[data],[name]]
            num += 1
    return hinge_type_name_dict


def select_common_3hinge(hinge_type_name_dict):
    common_3hinge_list_250 = list()
    common_3hinge_list_300 = list()
    common_3hinge_list_350 = list()
    common_3hinge_list_400 = list()
    common_3hinge_list_200 = list()
    for index in hinge_type_name_dict.keys():
        length = len(hinge_type_name_dict[index][1])
        if length >= 400:
            common_3hinge_list_400.append(hinge_type_name_dict[index][1])
        elif length >=350:
            common_3hinge_list_350.append(hinge_type_name_dict[index][1])
        elif length >= 300:
            common_3hinge_list_300.append(hinge_type_name_dict[index][1])
        elif length >= 250:
            common_3hinge_list_250.append(hinge_type_name_dict[index][1])
        elif length >=200:
            common_3hinge_list_200.append(hinge_type_name_dict[index][1])
    return common_3hinge_list_400, common_3hinge_list_350, common_3hinge_list_300, common_3hinge_list_250, common_3hinge_list_200


def write(common_3hinge_list, outprefix):
    with open(outprefix, 'w') as f:
        for name_list in common_3hinge_list:
            for name in name_list:
                f.write("%s\t" % name)
            f.write("\n")
    f.close()


def find_common_subject(common_3hinge_file):
    common_subject_list = list()
    final_name_list = list()
    total_name_list = list()
    for line in open(common_3hinge_file):
        total_name_list = total_name_list + line.strip('\n').split('\t')
    total_name_list = list(set(total_name_list))
    total_subject_list = list(set(['_'.join(name.split('_')[0:2]) for name in total_name_list if '_' in name]))
    for line in open(common_3hinge_file):
        current_name_list = line.strip('\n').split('\t')
        common_subject_list = list(set(['_'.join(name.split('_')[0:2]) for name in current_name_list if '_'.join(name.split('_')[0:2]) in total_subject_list]))
        total_subject_list = common_subject_list
        print(len(total_subject_list))
        print(total_subject_list)
    pdb.set_trace()

    '100206_lh_5_embedding_combine'


'''
============================================================== similarity ==========================================================
'''

def get_3hinge_embedding(path):
    print('======================== get_3hinge_name_data \t' + time.asctime(time.localtime(time.time())) + '=============================')
    name_embedding_dict = defaultdict(list)
    file_list = [file for file in os.listdir(path) if file.endswith('embedding_combine.txt')]
    for file in file_list:
        name = '_'.join(file.split('_')[0:3])
        data = np.loadtxt(path + '/' + file)
        if name not in name_embedding_dict.keys():
            name_embedding_dict[name] = [data]
        else:
            print('3hinge exist!')
            exit()
        pdb.set_trace()
    return name_embedding_dict


def create_DataID_pointID_dict(subject_list, sphere_list, mapping_file_dir):
    print('======================== create_DataID_pointID_dict \t' + time.asctime(time.localtime(time.time())) + '=============================')
    DataID_pointID_dict = dict()
    for subject in subject_list:
        for sphere in sphere_list:
            mapping_file = mapping_file_dir + '/' + subject + '_3hinge_ids_' + sphere + '.txt'
            ## ..../Graph_embedding/Common_3hinge_results/Graph_embedding_data_500/adj_feature_matrix/100206_3hinge_ids_lh.txt
            f = open(mapping_file, "r")
            lines = f.readlines()
            ## 2533      10
            ## 509      6
            ## 755      7
            for line in lines:
                line = line.strip('\n').replace(' ', '')
                pointID = line.split('\t')[0]
                index = line.split('\t')[1]
                if subject + '_' + sphere + '_' + index in DataID_pointID_dict.keys():
                    print('existing key:', subject + '_' + sphere + '_' + index)
                else:
                    DataID_pointID_dict[subject + '_' + sphere + '_' + index] = subject + '_' + sphere + '_' + pointID
    return DataID_pointID_dict


def create_dataId_embedding_dict(DataID_list, inputdir):
    print('======================== create_dataId_embedding_dict \t' + time.asctime(time.localtime(time.time())) + '=============================')
    DataId_embedding_dict = defaultdict(list)
    for DataID in DataID_list:
        embedding = np.loadtxt(inputdir + '/' + DataID + '_embedding.txt')
        embedding_combine = np.loadtxt(inputdir + '/' + DataID + '_embedding_combine.txt')
        DataId_embedding_dict[DataID] = []
        DataId_embedding_dict[DataID].append(np.float32(embedding))
        DataId_embedding_dict[DataID].append(np.float32(embedding_combine))
    return DataId_embedding_dict


def create_cluster_data(participant_DataID_list, DataID_embedding_dict, hot_type=0):
    print('======================== create_cluster_data \t' + time.asctime(time.localtime(time.time())) + '=============================')
    clusterInputIndex_DataID_dict = dict()
    data = list()
    for i in range(len(participant_DataID_list)):
        participant_DataID = participant_DataID_list[i]
        clusterInputIndex_DataID_dict[i] = participant_DataID
        if hot_type == 0:
            embedding_vector = DataID_embedding_dict[participant_DataID][0][0, :]
        elif hot_type == 1:
            embedding_vector = DataID_embedding_dict[participant_DataID][0][1, :]
        elif hot_type == 2:
            embedding_vector = DataID_embedding_dict[participant_DataID][0][2, :]
        elif hot_type == 3:
            embedding_vector = DataID_embedding_dict[participant_DataID][0][3, :]
        elif hot_type == 4:
            embedding_vector = DataID_embedding_dict[participant_DataID][1]
        else:
            print('hot type error!')
            exit()
        data.append(embedding_vector)
    data = np.stack(data)
    return data, clusterInputIndex_DataID_dict


def write_clusterID_dataID_pointID_mapping(DataID_pointID_dict, clusterInputIndex_DataID_dict, clusterInputIndex_label_dict, out_prefix):
    f = open(out_prefix + '_clusterID_dataID_pointID_mapping.txt', 'w')
    f.write('ClusterInputID'  + '-------' +  'DataID' + '-------' + 'PointID' + '-------' + 'ClusterLabel'+ '\n')
    for index in clusterInputIndex_DataID_dict.keys():
        DataID = clusterInputIndex_DataID_dict[index]
        pointID = DataID_pointID_dict[DataID]
        label = clusterInputIndex_label_dict[index]
        f.write(str(index)  + '-------' +  str(DataID) + '-------' + pointID + '-------' + str(label)+ '\n')
    f.close()


def save_model_info(model_info, info_type, out_prefix):
    if info_type == 'children_clusterID':
        np.savetxt(out_prefix + '_' + info_type + '.txt', model_info, fmt='%d')
    elif info_type == 'distances':
        np.savetxt(out_prefix + '_' + info_type + '.txt', model_info, fmt='%.4f')
    elif info_type == 'n_clusters':
        f = open(out_prefix + '_' + info_type + '.txt', 'w')
        f.write(str(model_info) + '\n')
        f.close()
    else:
        print('wrong info type:', info_type)
        exit()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def AG(data, cosine_dis, out_prefix):
    print('======================== AG \t' + time.asctime(time.localtime(time.time())) + '=============================')
    clustering = AgglomerativeClustering(n_clusters=None, affinity='cosine',  linkage='complete', distance_threshold=cosine_dis).fit(data)

    save_model_info(clustering.n_clusters_,'n_clusters', out_prefix)
    save_model_info(clustering.distances_, 'distances', out_prefix)
    # clustering.distances_ (122521,) clustering.distances_[140:145] array([0., 0., 0., 0., 0.])
    save_model_info(clustering.children_, 'children_clusterID', out_prefix)
    # clustering.children_.shape (122521, 2)
    '''
    p clustering.children_[1:6]
    array([[    1,   214],
           [    2,  1110],
           [    4,   194],
           [    5, 27331],
           [    6, 15634]])
    '''
    plot_dendrogram(clustering, truncate_mode='lastp', p=clustering.n_clusters_)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig(out_prefix + "_tree.png")
    plt.clf()

    labels = clustering.labels_
    # clustering.labels_.shape (122522,)
    # clustering.labels_[0:6] array([ 64, 522, 687, 517,  97, 369])
    clusterInputIndex_center_dict = dict()
    ## data.shape (122522, 128)
    for i in range(data.shape[0]):
        clusterInputIndex_center_dict[i] = labels[i]
    return clusterInputIndex_center_dict

def AP(data, cosine_dis, out_prefix):
    print('======================== AP \t' + time.asctime(time.localtime(time.time())) + '=============================')
    clustering = AffinityPropagation(random_state=5).fit(data)

    save_model_info(len(clustering.cluster_centers_indices_),'n_clusters', out_prefix)
    save_model_info(clustering.cluster_centers_, 'centers', out_prefix)
    save_model_info(clustering.labels_, 'label_clusterID', out_prefix)

    plot_dendrogram(clustering, truncate_mode='lastp', p=clustering.n_clusters_)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig(out_prefix + "_tree.png")
    plt.clf()

    labels = clustering.labels_
    clusterInputIndex_center_dict = dict()
    for i in range(data.shape[0]):
        clusterInputIndex_center_dict[i] = labels[i]
    return clusterInputIndex_center_dict


def write_pointID_label_info(DataID_pointID_dict, clusterInputIndex_DataID_dict, clusterInputIndex_label_dict, out_prefix):
    f = open(out_prefix + '.txt', 'w')
    for index in clusterInputIndex_label_dict.keys():
        DataID = clusterInputIndex_DataID_dict[index]
        label = clusterInputIndex_label_dict[index]
        pointID = DataID_pointID_dict[DataID]
        f.write(pointID + '-------' + str(label) + '\n')
    f.close()



def main(args):
    # name_data_dict = get_3hinge_name_data(args.root_dir)
    # hinge_type_name_dict = get_all_kinds_of_3hinge(name_data_dict)
    # list_400, list_350, list_300, list_250, list_200 = select_common_3hinge(hinge_type_name_dict)
    # write(list_400, args.out_dir + 'list_400_23_value.txt')
    # write(list_350, args.out_dir + 'list_350_23_value.txt')
    # write(list_300, args.out_dir + 'list_300_23_value.txt')
    # write(list_250, args.out_dir + 'list_250_23_value.txt')
    # write(list_200, args.out_dir + 'list_200_23_value.txt')
    # find_common_subject(args.out_dir + 'list_400_23.txt')

    inputdir = args.root_dir + '/' + args.input
    AG_outdir = args.root_dir + '/' + args.out_dir + '/AG_results_' + str(args.sub_num) + '_' + str(args.cosine_dis)
    AG_model_info_out_dir = args.root_dir + '/' + args.out_dir + '/AG_results_model_' + str(args.sub_num) + '_' + str(args.cosine_dis)

    ground_truth_dir = args.root_dir + '/' + args.ground_truth_dir
    mapping_file_dir = args.root_dir + '/' + args.mapping_file_dir
    subject_list = [ID.split('_')[0] for ID in os.listdir(inputdir) if not ID.startswith('.')]
    subject_list = list(set(subject_list))
    subject_list.sort()
    subject_list = subject_list[0:args.sub_num]
    subject_list = subject_list
    DataID_list = ['_'.join(ID.split('_')[0:3]) for ID in os.listdir(inputdir) if ID.split('_')[0] in subject_list]
    DataID_list = list(set(DataID_list))
    DataID_list.sort()

    #pdb.set_trace()
    if os.path.exists(AG_outdir):
        shutil.rmtree(AG_outdir)
    os.makedirs(AG_outdir)

    if os.path.exists(AG_model_info_out_dir):
        shutil.rmtree(AG_model_info_out_dir)
    os.makedirs(AG_model_info_out_dir)

    #pdb.set_trace()
    '''================================ create subject sphere index pointID dict ============================='''
    sphere_list = ['lh', 'rh']
    DataID_pointID_dict = create_DataID_pointID_dict(subject_list, sphere_list, mapping_file_dir)
    ##
    DataID_embedding_dict = create_dataId_embedding_dict(DataID_list, inputdir)

    '''================================================ AG ============================================='''
    participant_subject = subject_list
    participant_sphere = ['lh', 'rh']
    participant_DataID_list = [dataID for dataID in DataID_list if dataID.split('_')[0] in participant_subject and
                               dataID.split('_')[1] in participant_sphere]
    hot_type_list = [1,2,4]
    for hot_type in hot_type_list:
        data, clusterInputIndex_DataID_dict = create_cluster_data(participant_DataID_list, DataID_embedding_dict,
                                                                  hot_type=hot_type)
        ## data.shape (122522, 128)
        ## clusterInputIndex_DataID_dict 122522 pairs (num: ID -- like, 122485: '174841_rh_66')
        out_prefix = AG_model_info_out_dir + '/hot_type_' + str(hot_type)
        clusterInputIndex_label_dict = AG(data, args.cosine_dis, out_prefix)
        write_clusterID_dataID_pointID_mapping(DataID_pointID_dict, clusterInputIndex_DataID_dict, clusterInputIndex_label_dict, out_prefix)
        out_prefix = AG_outdir + '/hot_type_' + str(hot_type)
        write_pointID_label_info(DataID_pointID_dict, clusterInputIndex_DataID_dict, clusterInputIndex_label_dict,
                                  out_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph_embedding common 3hinge')
    # parser.add_argument('-r', '--root_dir', type=str, default='../Graph_embedding_data_500/node_input_data', help='path for root')
    parser.add_argument('-r', '--root_dir', type=str, default='/mnt/disk1/HCP_luzhang_do/Analysis/Graph_embedding/Common_3hinge_results', help='path for root')
    parser.add_argument('-i', '--input', type=str, default='exp_6', help='path for root')
    parser.add_argument('-g', '--ground_truth_dir', type=str, default='Graph_embedding_data_500/node_input_data', help='test data set dir')
    parser.add_argument('-m', '--mapping_file_dir', type=str, default='Graph_embedding_data_500/adj_feature_matrix', help='mapping_file_dir')
    parser.add_argument('-o', '--out_dir', type=str, default='common_3hinge_cluster_results', help='output dir')
    parser.add_argument('-cosine', '--cosine_dis', type=float, default=0.1, help='cosine_dis')
    parser.add_argument('-num', '--sub_num', type=float, default=360, help='cosine_dis')

    args = parser.parse_args()

    cosine_list = [0.05, 0.08, 0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3]

    for cosine in cosine_list:
        args.cosine_dis = cosine
        print(args)
        main(args)
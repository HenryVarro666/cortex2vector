import os
import shutil
import xlsxwriter
import xlrd
import numpy as np
from collections import defaultdict
import nibabel.freesurfer.io as io
import pdb


def get_individual_connection_info(excel_file1, outdir, sphere, subject):
    hinge_feature_dict = dict()
    hinge_connection_dict = defaultdict(list)

    worksheet1 = xlrd.open_workbook(excel_file1)
    sheets1 = worksheet1.sheet_names()
    sheet1 = worksheet1.sheet_by_name(sheets1[0])
    rows1 = sheet1.nrows
    cols1 = sheet1.ncols

    for row in range(1, rows1):
        hinge_id = int(sheet1.cell_value(row, 0))
        neighbor_1 = int(sheet1.cell_value(row, 2))
        neighbor_2 = int(sheet1.cell_value(row, 3))
        neighbor_3 = int(sheet1.cell_value(row, 4))
        hinge_label = sheet1.cell_value(row, 1)

        if hinge_id not in hinge_feature_dict.keys():
            hinge_feature_dict[hinge_id] = hinge_label
        else:
            print('existing 3 hinge in feature dict')
        
        if hinge_id not in hinge_connection_dict.keys():
            hinge_connection_dict[hinge_id] = [neighbor_1, neighbor_2, neighbor_3]
        else:
            print('existing 3 hinge in connection dict')
    return hinge_connection_dict, hinge_feature_dict



def create_adj(hinge_combine_dict, outdir, sphere, subject):
    hinge_list = list(hinge_combine_dict.keys())
    f = open(outdir + '/' + str(subject) + '_3hinge_ids_' + sphere + '.txt', 'w')
    hinge_id_index_dict = dict()
    for i in range(len(hinge_list)):
        hinge_id_index_dict[hinge_list[i]] = i
        f.write('%s \t %s\n' % (str(hinge_list[i]), str(i)))
    f.close()
    adj = np.zeros(shape=(len(hinge_list), len(hinge_list)))
    for id in hinge_combine_dict.keys():
        for neighbor in hinge_combine_dict[id]:
            if neighbor in hinge_combine_dict.keys():
                adj[hinge_id_index_dict[id], hinge_id_index_dict[neighbor]] = 1
    adj_1_hop = adj + np.eye(len(hinge_list))
    adj_2_hop = np.dot(adj_1_hop, adj_1_hop)
    adj_3_hop = np.dot(adj_2_hop, adj_1_hop)
    np.savetxt(outdir + '/' + str(subject) + '_3hinge_adj_' + sphere + '_1_hop.txt', adj_1_hop, fmt='%d')
    np.savetxt(outdir + '/' + str(subject) + '_3hinge_adj_' + sphere + '_2_hop.txt', adj_2_hop, fmt='%d')
    np.savetxt(outdir + '/' + str(subject) + '_3hinge_adj_' + sphere + '_3_hop.txt', adj_3_hop, fmt='%d')


def read_labels(label_file):
    label_list = list()
    with open(label_file) as f:
        for line in f:
            label_list.append(line.strip('\n')[3:])
    label_list = list(set(label_list))
    return label_list


def create_label_index_dict(label_list, outdir):
    f = open(outdir + '/' + 'label_index_mapping.txt', 'w')

    index = 0
    label_index_dict = dict()
    for label in label_list:
        if label not in label_index_dict.keys():
            label_index_dict[label] = index
            f.write('%30s \t\t %s\n' % (label, str(index)))
            index += 1
    f.close()

    return label_index_dict



def create_one_hot_feature(label_index_dict, hinge_feature_dict, outdir, sphere, subject):
    one_hot_feature_list = list()
    feature_num = len(list(label_index_dict.keys()))
    for hinge in hinge_feature_dict.keys():
        label = hinge_feature_dict[hinge]
        index = label_index_dict[label[3:]]
        one_hot_vector = np.zeros(shape = (feature_num,))
        one_hot_vector[index] = 1
        one_hot_feature_list.append(one_hot_vector)
    one_hot_feature_array = np.stack(one_hot_feature_list)
    
    np.savetxt(outdir + '/' + str(subject) + '_3hinge_0_hop_feature_' + sphere + '.txt', one_hot_feature_array, fmt='%d')


if __name__ == '__main__':
    file_root = '../HCP_GyralNet_files'
    outdir_root = '../Graph_embedding_data_500'
    matrix_outdir = outdir_root + '/' + 'adj_feature_matrix_500'

    if not os.path.exists(outdir_root):
        os.makedirs(outdir_root)

    if os.path.exists(matrix_outdir):
        shutil.rmtree(matrix_outdir)
    os.makedirs(matrix_outdir)

    label_file = '../labels.txt'
    label_list = read_labels(label_file)
    label_index_dict = create_label_index_dict(label_list, matrix_outdir)
    # exit()

    subjects_list = [subject.split('_')[0] for subject in os.listdir(file_root) if not subject.startswith('.')]
    subjects_list = list(set(subjects_list))
    subjects_list.sort()
    sphere_list = ['lh', 'rh']

    for subject in subjects_list:
        for sphere in sphere_list:
            file1 = file_root + '/' + subject + '_info_' + sphere + '.xlsx'
            hinge_connection_dict, hinge_feature_dict = get_individual_connection_info(file1, matrix_outdir, sphere, subject)
            create_adj(hinge_connection_dict, matrix_outdir, sphere, subject)
            create_one_hot_feature(label_index_dict, hinge_feature_dict, matrix_outdir, sphere, subject)



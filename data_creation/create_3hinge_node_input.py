import os
import shutil
import xlsxwriter
import xlrd
import numpy as np
from collections import defaultdict
import nibabel.freesurfer.io as io
import pdb

def generate_node_feature(one_hot_feature, hop_1_feature, hop_2_feature, hop_3_feature, outdir, prefix):
    for node in range(one_hot_feature.shape[0]):
        node_feature = np.zeros(shape = (4, one_hot_feature.shape[1]))
        node_feature[0,:] = one_hot_feature[node, :]
        node_feature[1,:] = hop_1_feature[node, :]
        node_feature[2,:] = hop_2_feature[node, :]
        node_feature[3,:] = hop_3_feature[node, :]
        np.savetxt(outdir + '/' + prefix + '_' + str(node) + '.txt', node_feature, fmt='%.1f')



if __name__ == '__main__':
    input_root = '../Graph_embedding_data_500/adj_feature_matrix_500'
    out_root = '../Graph_embedding_data_500'
    multi_hop_feature_outdir = out_root + '/' + 'multi_hop_feature_matrix_500'
    node_outdir = out_root + '/' + 'node_input_data_500'

    if os.path.exists(multi_hop_feature_outdir):
        shutil.rmtree(multi_hop_feature_outdir)
    os.makedirs(multi_hop_feature_outdir)

    if os.path.exists(node_outdir):
        shutil.rmtree(node_outdir)
    os.makedirs(node_outdir)

    # exit()

    subjects_list = [subject.split('_')[0] for subject in os.listdir(input_root) if not subject.startswith('.') and not subject.startswith('label')]
    subjects_list = list(set(subjects_list))
    subjects_list.sort()
    sphere_list = ['lh', 'rh']

    for subject in subjects_list:
        for sphere in sphere_list:
            one_hot_feature = np.loadtxt(input_root + '/' + str(subject) + '_3hinge_0_hop_feature_' + sphere + '.txt', dtype='int')
            adj_1_hop = np.loadtxt(input_root + '/' + str(subject) + '_3hinge_adj_' + sphere + '_1_hop.txt', dtype='int')
            adj_2_hop = np.loadtxt(input_root + '/' + str(subject) + '_3hinge_adj_' + sphere + '_2_hop.txt', dtype='int')
            adj_3_hop = np.loadtxt(input_root + '/' + str(subject) + '_3hinge_adj_' + sphere + '_3_hop.txt', dtype='int')
            hop_1_feature = np.dot(adj_1_hop, one_hot_feature)
            hop_2_feature = np.dot(adj_2_hop, one_hot_feature)
            hop_3_feature = np.dot(adj_3_hop, one_hot_feature)
            np.savetxt(multi_hop_feature_outdir + '/' + str(subject) + '_3hinge_0_hop_feature_' + sphere + '.txt', one_hot_feature, fmt='%d')
            np.savetxt(multi_hop_feature_outdir + '/' + str(subject) + '_3hinge_1_hop_feature_' + sphere + '.txt', hop_1_feature, fmt='%d')
            np.savetxt(multi_hop_feature_outdir + '/' + str(subject) + '_3hinge_2_hop_feature_' + sphere + '.txt', hop_2_feature, fmt='%d')
            np.savetxt(multi_hop_feature_outdir + '/' + str(subject) + '_3hinge_3_hop_feature_' + sphere + '.txt', hop_3_feature, fmt='%d')
            generate_node_feature(one_hot_feature, hop_1_feature, hop_2_feature, hop_3_feature, node_outdir, subject + '_' + sphere)
            # pdb.set_trace()




import numpy as np

import os
import json

class bub_instant(object):
    def __init__(self, sub, vertex, sphere, coordinate, i, roi_name, input_data, emb, adj_mat, hop0_feature,
                 hop1_feature, hop2_feature, fib, fiber_file, fpoints_file, ball_file, thinkness, sulc, curv,
                 h3_dir='H3', info_dir='informations', fiber_dir='ft'):
        self.sub = str(sub)
        self.sphere = str(sphere)
        self.i = int(i)
        self.vertex = int(vertex)
        print(vertex)
        self.coordinate = coordinate
        self.roi_label = str(roi_name)
        # self.id = sub + '_' + sphere + '_' + str(self.vertex)
        self.id = self.sub + '_' + self.sphere + '_' + str(self.vertex)

        self.input_data = input_data  ## numpy array
        self.emb = emb  ## numpy array
        self.emb1hop = emb[0:2, :]  ## numpy array
        self.emb2hop = emb[0:3, :]  ## numpy array
        self.hop_1_flat = emb[0:2, :].flatten()  ## numpy array
        self.hop_2_flat = emb[0:3, :].flatten()  ## numpy array
        self.hop_1_sum = np.sum(emb[0:2, :], axis=1)  ## numpy array
        self.adj_row = adj_mat[self.i, :]  ## numpy array
        self.hop0 = hop0_feature[self.i, :]  ## numpy array
        self.hop1 = hop1_feature[self.i, :]  ## numpy array
        self.hop2 = hop2_feature[self.i, :]  ## numpy array
        self.trace_map = fib  ## numpy array

        self.folder = h3_dir  # where bubble vtk saved
        self.fiber_dir = fiber_dir  # where fiber vtk and fiber features saved
        self.info_dir = info_dir  # meta information saved
        self.fiber_vtk = fiber_file
        self.fibPoints_vtk = fpoints_file
        self.ball_vtk = ball_file

        self.thinkness = float(thinkness)
        self.sulc = float(sulc)
        self.curv = float(curv)

        self.neighbor_vertex = []
        self.neibghber_labels = []
        self.neibghber_distance = []

        self.duplicate = False
        self.template_spot = None
        self.emb_sim = 0.0
        self.fib_sim = 0.0

        self.cluster = None
        self.match = False
        self.trait0 = None
        self.trait1 = None
        self.trait2 = None
        self.trait3 = None
        self.trait4 = None

        self.message = None
        self.date = '22-7-6'
        self.location0 = None
        self.location1 = None

    def __repr__(self):
        repr_dict = {}
        repr_dict['id'] = self.id
        repr_dict['date'] = self.date
        repr_dict['message'] = self.message
        repr_dict['roi'] = self.roi_label
        repr_dict['emb'] = self.emb1hop.tolist()
        # repr_dict['input'] = self.input_data
        repr_dict['subject'] = self.sub
        repr_dict['vertex'] = self.vertex
        repr_dict['trace map'] = self.trace_map.tolist()
        repr_dict['neighbers'] = self.neighbor_vertex
        repr_dict['bubble'] = self.ball_vtk
        repr_dict['fiber'] = self.fiber_vtk
        repr_dict['sulc'] = self.sulc
        return json.dumps(repr_dict)

    def __str__(self):
        return '3H (' + self.id + ',' + self.roi_label + ')'

    def find_neighber(self, row):
        print('------------ locating neighbers ----------', '\n')
        self.neighbor_vertex.append(row.iloc[2])
        self.neighbor_vertex.append(row.iloc[3])
        self.neighbor_vertex.append(row.iloc[4])
        self.neibghber_labels.append(row.iloc[5])
        self.neibghber_labels.append(row.iloc[6])
        self.neibghber_labels.append(row.iloc[7])
        l_0 = len(row.iloc[8].split(','))
        l_1 = len(row.iloc[9].split(','))
        l_2 = len(row.iloc[10].split(','))
        self.neibghber_distance.append(l_0)
        self.neibghber_distance.append(l_1)
        self.neibghber_distance.append(l_2)
        return 0

    def get_fiber_vtk(self, new_path):
        return os.path.join(new_path, str(self.sub), self.fiber_dir, str(self.fiber_vtk))

    def get_bubble_vtk(self, new_path):
        #print(new_path)
        return os.path.join(new_path, str(self.sub), self.folder, str(self.ball_vtk))

    def get_fpoints_vtk(self, new_path):
        return os.path.join(new_path, str(self.sub), self.fiber_dir, str(self.fibPoints_vtk))


class brain(object):
    def __init__(self, id, w_surf_lh, w_surf_rh, fiber, space='MNI', fiber_dir='fib', surf_dir='Surf', roi_dir='ROIs_combination'):
        self.id = str(id)
        self.surf_dir = surf_dir
        self.rois_dir = roi_dir
        self.fiber_dir = fiber_dir
        self.white_lh = w_surf_lh
        self.white_rh = w_surf_rh
        self.pial_lh = None
        self.pial_rh = None
        self.inflat_lh = None
        self.inflat_rh = None
        self.g_net_lh = None
        self.g_net_rh = None
        self.T1 = None
        self.fa = None
        self.stat = None
        self.fiber = fiber
        self.space = space
        self.nodes = {}
        self.gyrul = {}
        self.sulci = {}
        self.all_hings = {}
        self.bad_hings = {}
        self.good_hings = {}
        self.hing_number = len(self.good_hings)

        self.template = False
        self.work_temp = []
        self.trait0 = None
        self.trait1 = None
        self.trait2 = None
        self.trait3 = None
        self.trait4 = None

        self.message = None
        self.date = '22-7-6'
        self.location0 = None
        self.location1 = None

    def __repr__(self):
        repr_dict = {}
        repr_dict['id'] = self.id
        repr_dict['date'] = self.date
        repr_dict['message'] = self.message
        repr_dict['white_lh'] = self.white_lh
        repr_dict['white_rh'] = self.white_rh
        repr_dict['fiber'] = self.fiber
        repr_dict['coord system of space'] = self.space
        repr_dict['all 3 hings'] = len(self.all_hings)
        repr_dict['good 3 hings'] = len(self.good_hings)
        return json.dumps(repr_dict)

    def __str__(self):
        return 'Brain (' + self.id + ', ' + self.white_lh + ', ' + self.white_rh + ')'

    def get_lh_vtk(self, root_path):
        return os.path.join(root_path, self.id,  self.surf_dir, str(self.white_lh))
    def get_rh_vtk(self, root_path):
        return os.path.join(root_path, self.id, self.surf_dir, str(self.white_rh))
    def get_allROIs_vtk(self, root_path):
        return os.path.join(root_path, self.id, self.rois_dir, 'ALL_color_combine.white.vtk')
    def get_fiber_vtk(self, root_path):
        return os.path.join(root_path, self.id, self.fiber_dir, 'fiber_fix.vtk')

    def get_ratio(self):
        return str(len(self.good_hings))+' / '+str(len(self.all_hings))

    def check_duplicon(self, hings_dict):
        print('------------ check duplication ----------', '\n')
        assert len(self.all_hings) > 3
        short_threshold = 12
        in_set = []
        for n in self.all_hings.keys():
            nn = hings_dict[n]
            print(nn.neibghber_distance)
            if any(np.array(nn.neibghber_distance) < short_threshold):
                ids_ = [n, ]
                sulc_ = [float(nn.sulc), ]
                for k, d in enumerate(nn.neibghber_distance):
                    if d < short_threshold:
                        neib_id_ = self.id + '_' + nn.sphere + '_' + str(nn.neighbor_vertex[k])
                        if neib_id_ in self.all_hings.keys():
                            ids_.append(neib_id_)
                            sulc_.append(float(hings_dict[neib_id_].sulc))
                print(ids_)
                print(sulc_)
                b_ = np.argmax(np.array(sulc_))
                bb = ids_[b_]
                #in_set.append(bb)
                if bb in in_set:
                    pass
                else:
                    in_set.append(bb)
            else:
                in_set.append(n)
        in_list = list(set(in_set))
        print(len(in_list), ' / ', len(self.all_hings))

        for h in in_list:
            self.good_hings[h] = self.all_hings[h]
        rest_ones = list(set(list(self.all_hings.keys())).difference(in_set))
        for h in rest_ones:
            self.bad_hings[h] = self.all_hings[h]
        # for n in self.all_hings.keys():
        #     if n in in_list:
        #         self.good_hings[n] = self.all_hings[n]
        #     else:
        #         self.bad_hings[n] = self.all_hings[n]
        print('len_good', len(in_list), len(self.good_hings))
        assert len(in_list) == len(self.good_hings)
        print(self.get_ratio())
        return 0


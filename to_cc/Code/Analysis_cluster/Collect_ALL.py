from scipy import spatial
import pickle as pickle
import numpy as np
import vtk

import os
import shutil
import json
import openpyxl
import xlrd
#import xlsxwriter
import pandas as pd
import sys

from Brain_and_Hings import *

########################################################################################################################
def read_vtk_file(vtk_file):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    Header = reader.GetHeader()
    polydata = reader.GetOutput()
    nCells = polydata.GetNumberOfCells()
    nPolys = polydata.GetNumberOfPolys()
    nLines = polydata.GetNumberOfLines()
    nStrips = polydata.GetNumberOfStrips()
    nPieces = polydata.GetNumberOfPieces()
    nVerts = polydata.GetNumberOfVerts()
    nPoints = polydata.GetNumberOfPoints()
    Points = polydata.GetPoints()
    Point = polydata.GetPoints().GetPoint(0)

    return polydata

def calculate_coordinate(index, vtk_file):
    vtk = vtk_file
    polydata = read_vtk_file(vtk)
    center = np.zeros(shape=(3,))
    coordinate = polydata.GetPoints().GetPoint(index)
    return coordinate

def drawBubble(center, radius, color, bubble_output):
    print(bubble_output)
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(8*radius)
    sphere.SetCenter(center)
    sphere.SetThetaResolution(40)
    sphere.SetPhiResolution(40)
    sphere.Update()
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(sphere.GetOutput())
    writer.SetFileName(bubble_output)
    writer.Update()

    points = vtk.vtkPoints()
    polygons = vtk.vtkCellArray()

    polydata = read_vtk_file(bubble_output)
    CellArray = polydata.GetPolys()
    Polygons = CellArray.GetData()

    for point in range(polydata.GetNumberOfPoints()):
        coordinate = polydata.GetPoints().GetPoint(point)
        points.InsertNextPoint(coordinate)

    for i in range(0, CellArray.GetNumberOfCells()):
        triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
        polygon = vtk.vtkPolygon()

        polygon.GetPointIds().SetNumberOfIds(3)
        polygon.GetPointIds().SetId(0, triangle[0])
        polygon.GetPointIds().SetId(1, triangle[1])
        polygon.GetPointIds().SetId(2, triangle[2])

        polygons.InsertNextCell(polygon)
        Colors.InsertNextTuple3(float(color[0]), float(color[1]), float(color[2]))

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)
    polygonPolyData.GetCellData().SetScalars(Colors)
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(bubble_output)
    writer.Write()

def get_colors():
    color_list = [(128,0,0),(220,20,60),(205,92,92),(255,160,122),(255,215,0),(189,183,107),(154,205,50),(127,255,0),
                  (34,139,34),(152,251,152),(46,139,87),(47,79,79),(0,255,255),(72,209,204),(95,158,160),(30,144,255),
                  (25,25,112),(0,0,255),(72,61,139),(139,0,139),(128,0,128),(255,0,255),(255,20,147),(250,235,215),
                  (245,222,179),(255,255,224),(205,133,63),(188,143,143),(255,228,225),(255,239,213),(119,136,153),
                  (240,248,255),(240,255,255),(128,128,128),(220,220,220)]
    color_list = [(240,0,0)] * 500
    return color_list

def _collect_init(data_dir, adj_feature_matrix_folder, surf_dir='Surf', color=[11,225,11], radius=0.4, h3_dir = 'H3'):
    sbs_list = os.listdir(data_dir)
    for sub in sbs_list:
        if os.path.exists(os.path.join(data_dir, sub, h3_dir)):
            shutil.rmtree(os.path.join(data_dir, sub, h3_dir))
        os.mkdir(os.path.join(data_dir, sub, h3_dir))
        print(os.path.join(data_folder, sub, h3_dir))

        lh_file = os.path.join(adj_feature_matrix_folder, sub+'_3hinge_ids_lh.txt')
        rh_file = os.path.join(adj_feature_matrix_folder, sub+'_3hinge_ids_rh.txt')
        with open(lh_file) as file:
            # /mnt/disk2/work_2022_5/Rotate_HCP/Graph_embedding/Common_3hinge_results/Graph_embedding_data_500/adj_feature_matrix/136732_3hinge_ids_lh.txt
            lh_nodes = [int(line.rstrip().split(' 	 ')[0]) for line in file]
        with open(rh_file) as file:
            # /mnt/disk2/work_2022_5/Rotate_HCP/Graph_embedding/Common_3hinge_results/Graph_embedding_data_500/adj_feature_matrix/136732_3hinge_ids_lh.txt
            rh_nodes = [int(line.rstrip().split(' 	 ')[0]) for line in file]

        np.savetxt(os.path.join(data_dir, sub, 'lh_collect.txt'), np.stack(lh_nodes), fmt='%i', delimiter='\t')
        np.savetxt(os.path.join(data_dir, sub, 'rh_collect.txt'), np.stack(rh_nodes), fmt='%i', delimiter='\t')

        ## plot
        lh_s = os.path.join(data_folder, sub, surf_dir, 'lh_fix.white.vtk')
        print(lh_s)
        if os.path.exists(lh_s):
            for i, vertex in enumerate(lh_nodes):
                print('vertex is ', vertex)
                coordinate = calculate_coordinate(int(vertex), lh_s)
                color = color
                bubble_output = os.path.join(data_folder, sub, h3_dir, sub + '_lh_' + str(vertex) + '.vtk')
                # bubble
                drawBubble(coordinate, radius, color, bubble_output)
        rh_s = os.path.join(data_folder, sub, 'Surf', 'rh_fix.white.vtk')
        print(rh_s)
        if os.path.exists(rh_s):
            for i, vertex in enumerate(rh_nodes):
                print('vertex is ', vertex)
                coordinate = calculate_coordinate(int(vertex), rh_s)
                color = color
                bubble_output = os.path.join(data_folder, sub, h3_dir, sub + '_rh_' + str(vertex) + '.vtk')
                print(bubble_output)
                # bubble
                drawBubble(coordinate, radius, color, bubble_output)

    return 0

########################################################################################################################

## HERE do the 'trace_map_script' before 'collect next'

########################################################################################################################

def _collect_brains(data_folder, fiber_dir='fib', surf_dir='Surf'):
    print('------------ start collect brains ----------', '\n')
    sbs_list = os.listdir(data_folder)
    d = {}
    for sub in sbs_list:
        #(self, id, w_surf_lh, w_surf_rh, fiber, space='MNI', surf_dir='Surf', roi_dir='ROIs_combination')
        id = sub
        w_surf_lh = 'lh_fix.white.vtk'
        w_surf_rh = 'rh_fix.white.vtk'
        fiber = 'fiber_fix.vtk'
        s_obj = brain(id, w_surf_lh, w_surf_rh, fiber)
        d.setdefault(s_obj.id, s_obj)

    return d

def _collect_next(data_dir, multi_hop_feature_folder, adj_feature_matrix_folder, node_input_folder, model_results_embedding_folder, excel_folder, brain_dict, h3_dir='H3', info_dir='informations', fiber_dir='ft'):
    # (self, sub, vertex, sphere, coordinate, i, roi_name, input_data, emb, adj_mat, hop0_feature, hop1_feature, hop2_feature, fib, fiber_file, fpoints_file, ball_file, thinkness, sulc, curv, h3_dir='H3',info_dir='informations')
    print('------------ start collect 3 hings ----------', '\n')
    error_log = []
    all_dict = {}
    sbs_list = os.listdir(data_dir)
    for sub in sbs_list:
        print('------------ work on ', sub,  '----------', '\n')
        paths = [os.path.join(data_dir, sub, fiber_dir, str(sub)+'_lh_TracemapFeatures.txt'), os.path.join(data_dir, sub, fiber_dir, str(sub)+'_rh_TracemapFeatures.txt')]
        if os.path.exists(paths[0]) and os.path.exists(paths[1]):
            pass
        else:
            error_log.append(str(sub))
            continue

        # for lh
        print('------------ lh ----------', '\n')
        lh_list = []
        lh_nodes = np.loadtxt(os.path.join(data_dir, sub ,'lh_collect.txt'))
        lh_nodes.astype(int)
        excel = os.path.join(excel_folder, sub + '_info_' + 'lh' + '.xlsx')
        print(excel)
        lh_s = os.path.join(data_folder, sub, 'Surf', 'lh_fix.white.vtk')
        hop0 = np.loadtxt(os.path.join(multi_hop_feature_folder, str(sub)+'_3hinge_0_hop_feature_lh.txt'))
        hop1 = np.loadtxt(os.path.join(multi_hop_feature_folder, str(sub)+'_3hinge_1_hop_feature_lh.txt'))
        hop2 = np.loadtxt(os.path.join(multi_hop_feature_folder, str(sub)+'_3hinge_2_hop_feature_lh.txt'))
        adjmat = np.loadtxt(os.path.join(adj_feature_matrix_folder, str(sub)+'_3hinge_adj_lh_1_hop.txt'))
        Tracemap = np.loadtxt(os.path.join(data_dir, sub, fiber_dir, str(sub)+'_lh_TracemapFeatures.txt'))
        df = pd.read_excel(excel, engine='openpyxl')

        for i in range(lh_nodes.shape[0]):
            vertex = int(lh_nodes[i])
            print(vertex)
            coordinate = calculate_coordinate(int(vertex), lh_s)
            emb = np.loadtxt(os.path.join(model_results_embedding_folder, str(sub)+'_lh_' + str(i) + '_embedding.txt'))
            #vec = np.loadtxt(os.path.join(vec_dir, sub + '_' + sphere + '_' + str(i) + '_embedding.txt'))
            input_data = np.loadtxt(os.path.join(node_input_folder, str(sub)+'_lh_' + str(i) + '.txt'))
            fib = Tracemap[i,:]
            # fiber_file = os.path.join(data_dir, sub, fiber_dir, str(sub)+'_lh.seed.'+str(vertex)+'.vtk')
            # fpoints_file = os.path.join(data_dir, sub, fiber_dir, str(sub)+'_lh.seed.'+str(vertex)+'.TracemapPoints.vtk')
            # ball_file = os.path.join(data_folder, sub, h3_dir, sub + '_lh_' + str(vertex) + '.vtk')
            fiber_file = str(sub) + '_lh.seed.' + str(vertex) + '.vtk'
            fpoints_file = str(sub) + '_lh.seed.' + str(vertex) + '.TracemapPoints.vtk'
            ball_file = sub + '_lh_' + str(vertex) + '.vtk'

            for index, row in df.iterrows():
                # print(row.iloc[0])
                if row.iloc[0] == int(vertex):
                    info = row
                    roi_name, thinkness, sulc, curv = row.iloc[1], row.iloc[17], row.iloc[18], row.iloc[19]

            node=bub_instant(sub, vertex, 'lh', coordinate, i, roi_name, input_data, emb, adjmat, hop0, hop1, hop2, fib, fiber_file, fpoints_file, ball_file, thinkness, sulc, curv, h3_dir='H3',info_dir='informations')
            node.find_neighber(info)
            lh_list.append(node)

        # for rh
        print('------------ rh ----------', '\n')
        rh_list = []
        rh_nodes = np.loadtxt(os.path.join(data_dir, sub, 'rh_collect.txt'))
        rh_nodes.astype(int)
        excel = os.path.join(excel_folder, sub + '_info_' + 'rh' + '.xlsx')
        print(excel)
        rh_s = os.path.join(data_folder, sub, 'Surf', 'rh_fix.white.vtk')
        hop0 = np.loadtxt(os.path.join(multi_hop_feature_folder, str(sub) + '_3hinge_0_hop_feature_rh.txt'))
        hop1 = np.loadtxt(os.path.join(multi_hop_feature_folder, str(sub) + '_3hinge_1_hop_feature_rh.txt'))
        hop2 = np.loadtxt(os.path.join(multi_hop_feature_folder, str(sub) + '_3hinge_2_hop_feature_rh.txt'))
        adjmat = np.loadtxt(os.path.join(adj_feature_matrix_folder, str(sub) + '_3hinge_adj_rh_1_hop.txt'))
        Tracemap = np.loadtxt(os.path.join(data_dir, sub, fiber_dir, str(sub) + '_rh_TracemapFeatures.txt'))
        df = pd.read_excel(excel, engine='openpyxl')

        for i in range(rh_nodes.shape[0]):
            vertex = int(rh_nodes[i])
            print(vertex)
            coordinate = calculate_coordinate(int(vertex), rh_s)
            emb = np.loadtxt(os.path.join(model_results_embedding_folder, str(sub) + '_rh_' + str(i) + '_embedding.txt'))
            input_data = np.loadtxt(os.path.join(node_input_folder, str(sub) + '_rh_' + str(i) + '.txt'))
            fib = Tracemap[i,:]
            # fiber_file = os.path.join(data_dir, sub, fiber_dir, str(sub) + '_rh.seed.' + str(vertex) + '.vtk')
            # fpoints_file = os.path.join(data_dir, sub, fiber_dir, str(sub) + '_rh.seed.' + str(vertex) + '.TracemapPoints.vtk')
            # ball_file = os.path.join(data_folder, sub, h3_dir, sub + '_rh_' + str(vertex) + '.vtk')
            fiber_file = str(sub) + '_rh.seed.' + str(vertex) + '.vtk'
            fpoints_file = str(sub) + '_rh.seed.' + str(vertex) + '.TracemapPoints.vtk'
            ball_file = sub + '_rh_' + str(vertex) + '.vtk'

            for index, row in df.iterrows():
                # print(row.iloc[0])
                if row.iloc[0] == int(vertex):
                    info = row
                    roi_name, thinkness, sulc, curv = row.iloc[1], row.iloc[17], row.iloc[18], row.iloc[19]

            node = bub_instant(sub, vertex, 'rh', coordinate, i, roi_name, input_data, emb, adjmat, hop0, hop1, hop2,
                               fib, fiber_file, fpoints_file, ball_file, thinkness, sulc, curv, h3_dir='H3',
                               info_dir='informations')
            node.find_neighber(info)
            rh_list.append(node)

        for n in lh_list:
            print(brain_dict[sub])
            print(n.id)
            brain_dict[sub].all_hings.setdefault(n.id, (n.id, n.vertex))
            #brain_dict[sub].all_hings[n.id] =(n.id, n.vertex)
        for n in rh_list:
            print(brain_dict[sub])
            brain_dict[sub].all_hings.setdefault(n.id, (n.id, n.vertex))

        for h in lh_list+rh_list:
            all_dict.setdefault(h.id, h)

    return all_dict, brain_dict, error_log


if __name__ == '__main__':
    #data_folder = '/mnt/disk1/HCP_luzhang_do/Select_10'
    data_folder = '/mnt/disk1/HCP_luzhang_do/HCP_200_project_use'
    adj_feature_matrix_folder = '/mnt/disk1/HCP_luzhang_do/Graph_embedding/Common_3hinge_results/Graph_embedding_data_500/adj_feature_matrix'
    multi_hop_feature_folder = '/mnt/disk1/HCP_luzhang_do/Graph_embedding/Common_3hinge_results/Graph_embedding_data_500/multi_hop_feature_matrix'
    node_input_folder = '/mnt/disk1/HCP_luzhang_do/Graph_embedding/Common_3hinge_results/Graph_embedding_data_500/node_input_data'
    model_results_embedding_folder = '/mnt/disk1/HCP_luzhang_do/Graph_embedding/Common_3hinge_results/exp_6'
    excel_folder = '/mnt/disk1/HCP_luzhang_do/Analysis/HCP_GyralNet_files'
    output = '/home/yan/Craftable/HCP_zl_project/data_saves'

    act = input('Do step 0 or 1 ?')
    print('act:', act, '\n')
    if int(act) in [0, 1]:
        if int(act) == 0:
            print('-----------------------------------------------First step------------------------------------', '\n')
            _collect_init(data_folder, adj_feature_matrix_folder)
            print('\n', 'Do trace map before step to next stage, the defaults trace-map dir under subjects\' folder is \"ft\" ...')
        elif int(act) == 1:
            print('-----------------------------------------------Next step-------------------------------------', '\n')
            brain_dict = _collect_brains(data_folder)
            print(len(brain_dict))
            NODES_dict, SUBS_dict, E = _collect_next(data_folder, multi_hop_feature_folder, adj_feature_matrix_folder, node_input_folder, model_results_embedding_folder, excel_folder, brain_dict, h3_dir='H3', info_dir='informations', fiber_dir='ft')
            for k in SUBS_dict.keys():
                print(SUBS_dict[k])
                SUBS_dict[k].check_duplicon(NODES_dict)

            ## review
            c = 0
            for k in SUBS_dict.keys():
                if c < 5:
                    print(repr(SUBS_dict[k]))
                    c += 1
            c = 0
            for k in NODES_dict.keys():
                if c < 10:
                    print(repr(NODES_dict[k]))
                    c += 1
            ## save
            print('------------ start saving ----------', '\n')
            with open(os.path.join(output,'3hing_objects.pickle'), "wb") as nf:
                pickle.dump(NODES_dict, nf, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(output,'brains.pickle'), "wb") as bf:
                pickle.dump(SUBS_dict, bf, protocol=pickle.HIGHEST_PROTOCOL)
            print('Warning Follow subject missing items: ', E)

        else:
            print('---------------------------------------------unknown step------------------------------------', '\n')
            print('Abort')
            sys.exit(1)
    else:
        print('Abort')
        sys.exit(1)


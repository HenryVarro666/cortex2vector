import math
import vtk
import numpy as np
import nibabel as nib
from nibabel import freesurfer as nfs
# from __future__ import division
from scipy.stats import pearsonr

def read_meta_header(mhd_file):
    """Return a dictionary of meta data from meta header file"""
    fileIN = open(mhd_file, "r")
    line = fileIN.readline()

    meta_dict = {}
    tag_set = []
    tag_set.extend(['ObjectType', 'NDims', 'DimSize', 'ElementType', 'ElementDataFile', 'ElementNumberOfChannels'])
    tag_set.extend(['BinaryData', 'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize'])
    tag_set.extend(['Offset', 'CenterOfRotation', 'AnatomicalOrientation', 'ElementSpacing', 'TransformMatrix'])
    tag_set.extend(['Comment', 'SeriesDescription', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime'])

    tag_flag = [False] * len(tag_set)
    while line:
        tags = str.split(line, '=')
        # print(tags[0])
        for i in range(len(tag_set)):
            tag = tag_set[i]
            if (str.strip(tags[0]) == tag) and (not tag_flag[i]):
                # print(tags[1])
                content = str.strip(tags[1])
                if tag in ['ElementSpacing', 'Offset', 'CenterOfRotation', 'TransformMatrix']:
                    meta_dict[tag] = [float(s) for s in content.split()]
                elif tag in ['NDims', 'ElementNumberOfChannels']:
                    meta_dict[tag] = int(content)
                elif tag in ['DimSize']:
                    meta_dict[tag] = [int(s) for s in content.split()]
                elif tag in ['BinaryData', 'BinaryDataByteOrderMSB', 'CompressedData']:
                    if content == "True":
                        meta_dict[tag] = True
                    else:
                        meta_dict[tag] = False
                else:
                    meta_dict[tag] = content
                tag_flag[i] = True
        line = fileIN.readline()
    # print(comment)
    fileIN.close()
    return meta_dict


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


def read_nifti_file(nifti_file):
    nii_image = nib.load(nifti_file)
    nii_data = nii_image.get_data()
    return nii_data


def surf_region_index_list(lh_annot_file, rh_annot_file):
    lh = nfs.read_annot(lh_annot_file)
    rh = nfs.read_annot(rh_annot_file)

    ragion_index_list = []
    regions = []

    for id in np.unique(lh[0]):
        if id != -1:
            index = [i for i in range(len(lh[0])) if lh[0][i] == id]
            ragion_index_list.append(index)
            regions.append('lh_' + lh[2][id])
            print('%-36s' % ('lh_' + lh[2][id]), 'id = %-3d, sum = %-6d' % (id, (lh[0] == id).sum()), 'len(index) = %-6d' % len(index))
        else:
            print('%-36s' % 'lh_Unknown', 'id = %-3d, sum = %-6d' % (id, (lh[0] == id).sum()))

    for id in np.unique(rh[0]):
        if id != -1:
            index = [i for i in range(len(rh[0])) if rh[0][i] == id]
            ragion_index_list.append(index)
            regions.append('rh_' + lh[2][id])
            print('%-36s' % ('rh_' + lh[2][id]), 'id = %-3d, sum = %-6d' % (id, (rh[0] == id).sum()), 'len(index) = %-6d' % len(index))
        else:
            print('%-36s' % 'rh_Unknown', 'id = %-3d, sum = %-6d' % (id, (rh[0] == id).sum()))

    # for id in range(1, 76):
    for id in range(1, 36):
        if id not in np.unique(lh[0]):
            index = []
            ragion_index_list.append(index)
            regions.append('lh_' + lh[2][id])
            print('%-36s' % ('lh_' + lh[2][id]), 'id = %-3d, sum = %-6d' % (id, (lh[0] == id).sum()), 'len(index) = %-6d' % len(index))

    # for id in range(1, 76):
    for id in range(1, 36):
        if id not in np.unique(rh[0]):
            index = []
            ragion_index_list.append(index)
            regions.append('rh_' + rh[2][id])
            print('%-36s' % ('rh_' + rh[2][id]), 'id = %-3d, sum = %-6d' % (id, (rh[0] == id).sum()), 'len(index) = %-6d' % len(index))

    # print regions

    region_index_list = zip(regions, ragion_index_list)

    return region_index_list


def surf_id_region_index_dicts_whole(lh_annot_file, rh_annot_file):
    lh = nfs.read_annot(lh_annot_file)
    rh = nfs.read_annot(rh_annot_file)

    id_region_dict = {}
    region_index_dict = {}

    for id in np.unique(lh[0]):
        if id != -1:
            for i in range(len(lh[0])):
                if lh[0][i] == id:
                    id_region_dict.setdefault(id, 'lh_' + lh[2][id].decode("utf-8"))
                    region_index_dict.setdefault('lh_' + lh[2][id].decode("utf-8"), []).append(i)
        else:
            for i in range(len(lh[0])):
                if lh[0][i] == -1:
                    id_region_dict.setdefault(-1, 'lh_' + lh[2][0].decode("utf-8"))
                    region_index_dict.setdefault('lh_' + lh[2][0].decode("utf-8"), []).append(i)
        # print id, id_region_dic[id], len(region_index_dic[id_region_dic[id]])

    for id in np.unique(rh[0]):
        if id != -1:
            for i in range(len(rh[0])):
                if rh[0][i] == id:
                    id_region_dict.setdefault(id + 100, 'rh_' + rh[2][id].decode("utf-8"))
                    region_index_dict.setdefault('rh_' + rh[2][id].decode("utf-8"), []).append(i)

        else:
            for i in range(len(rh[0])):
                if rh[0][i] == -1:
                    id_region_dict.setdefault(-1 + 100, 'rh_' + rh[2][0].decode("utf-8"))
                    region_index_dict.setdefault('rh_' + rh[2][0].decode("utf-8"), []).append(i)
        # print id + 100, id_region_dic[id + 100], len(region_index_dic[id_region_dic[id + 100]])

    # for id in range(1, 76):
    for id in range(1, 36):
        if id not in np.unique(lh[0]):
            id_region_dict.setdefault(id, 'lh_' + lh[2][id].decode("utf-8"))
            region_index_dict.setdefault('lh_' + lh[2][id].decode("utf-8"), [])

    # for id in range(1, 76):
    for id in range(1, 36):
        if id not in np.unique(rh[0]):
            id_region_dict.setdefault(id + 100, 'rh_' + rh[2][id].decode("utf-8"))
            region_index_dict.setdefault('rh_' + rh[2][id].decode("utf-8"), [])

    # print 'surf_id_region_index_dicts_whole``````````````', id_region_dict[42], region_index_dict[id_region_dict[42]]
    surf_id_region_index_dicts_whole = [id_region_dict, region_index_dict]

    return surf_id_region_index_dicts_whole


def surf_id_region_index_dicts_deUknown(lh_annot_file, rh_annot_file):
    lh = nfs.read_annot(lh_annot_file)
    rh = nfs.read_annot(rh_annot_file)

    id_region_dict = {}
    region_index_dict = {}

    for id in np.unique(lh[0]):
        if id != -1:
            for i in range(len(lh[0])):
                if lh[0][i] == id:
                    id_region_dict.setdefault(id, 'lh_' + lh[2][id].decode("utf-8"))
                    region_index_dict.setdefault('lh_' + lh[2][id].decode("utf-8"), []).append(i)
            # print id, id_region_dic[id], len(region_index_dic[id_region_dic[id]])

    for id in np.unique(rh[0]):
        if id != -1:
            for i in range(len(rh[0])):
                if rh[0][i] == id:
                    id_region_dict.setdefault(id + 100, 'rh_' + rh[2][id].decode("utf-8"))
                    region_index_dict.setdefault('rh_' + rh[2][id].decode("utf-8"), []).append(i)
            # print id+100, id_region_dic[id+100], len(region_index_dic[id_region_dic[id+100]])

    # for id in range(1, 76):
    for id in range(1, 36):
        if id not in np.unique(lh[0]):
            id_region_dict.setdefault(id, 'lh_' + lh[2][id].decode("utf-8"))
            region_index_dict.setdefault('lh_' + lh[2][id].decode("utf-8"), [])

    # for id in range(1, 76):
    for id in range(1, 36):
        if id not in np.unique(rh[0]):
            id_region_dict.setdefault(id + 100, 'rh_' + rh[2][id].decode("utf-8"))
            region_index_dict.setdefault('rh_' + rh[2][id].decode("utf-8"), [])

    # print 'surf_id_region_index_dicts_Uknown``````````````', id_region_dict[42], region_index_dict[id_region_dict[42]]

    surf_id_region_index_dicts_deUknown = [id_region_dict, region_index_dict]

    # print surf_id_region_index_dicts_deUknown[0]

    return surf_id_region_index_dicts_deUknown


def surf_id_region_coordinate_dicts(lh_surf_vtk_polydata, rh_surf_vtk_polydata, surf_id_region_index_dicts):
    id_region_dict = surf_id_region_index_dicts[0]
    region_index_dict = surf_id_region_index_dicts[1]

    region_coordinate_dict = {}

    for id in id_region_dict.keys():
        if id < 99:
            if not len(region_index_dict[id_region_dict[id]]):
                region_coordinate_dict.setdefault(id_region_dict[id], [])
            else:
                for index in region_index_dict[id_region_dict[id]]:
                    Point = lh_surf_vtk_polydata.GetPoints().GetPoint(index)
                    region_coordinate_dict.setdefault(id_region_dict[id], []).append(Point)
        else:
            if not len(region_index_dict[id_region_dict[id]]):
                region_coordinate_dict.setdefault(id_region_dict[id], [])
            else:
                for index in region_index_dict[id_region_dict[id]]:
                    Point = rh_surf_vtk_polydata.GetPoints().GetPoint(index)
                    region_coordinate_dict.setdefault(id_region_dict[id], []).append(Point)

    surf_id_region_coordinate_dicts = [id_region_dict, region_coordinate_dict]

    return surf_id_region_coordinate_dicts


def coordinate_2_voxel(coordinate, mhd_dict):
    Offset = mhd_dict['Offset']
    ElementSpacing = mhd_dict['ElementSpacing']

    voxel_point = [0 for j in range(3)]
    voxel_point[0] = int(math.floor((coordinate[0] - Offset[0]) / ElementSpacing[0]))
    voxel_point[1] = int(math.floor((coordinate[1] - Offset[1]) / ElementSpacing[1]))
    voxel_point[2] = int(math.floor((coordinate[2] - Offset[2]) / ElementSpacing[2]))

    return voxel_point


def freesurferCoordinate_2_voxel(freesurferCoordinate, mhd_dict):
    ElementSpacing = mhd_dict['ElementSpacing']
    Dim = mhd_dict['DimSize']

    voxel_point = [0 for j in range(3)]
    voxel_point[0] = int(math.floor((-freesurferCoordinate[0]) / ElementSpacing[0] + Dim[0] / 2))
    voxel_point[1] = int(math.floor(freesurferCoordinate[1] / ElementSpacing[1] + Dim[1] / 2))
    voxel_point[2] = int(math.floor(freesurferCoordinate[2] / ElementSpacing[2] + Dim[2] / 2))

    return voxel_point


def freesurferCoordinate_2_coordinate(freesurferCoordinate, mhd_dict):
    Offset = mhd_dict['Offset']
    ElementSpacing = mhd_dict['ElementSpacing']
    Dim = mhd_dict['DimSize']

    coordinate = [0 for j in range(3)]
    coordinate[0] = int(math.floor((-freesurferCoordinate[0]) + Dim[0] / 2 * ElementSpacing[0] + Offset[0]))
    coordinate[1] = int(math.floor(freesurferCoordinate[1] + Dim[1] / 2 * ElementSpacing[1] + Offset[1]))
    coordinate[2] = int(math.floor(freesurferCoordinate[2] + Dim[2] / 2 * ElementSpacing[2] + Offset[2]))

    return coordinate


def coordinate_2_freesurferCoordinate(coordinate, mhd_dict):
    Offset = mhd_dict['Offset']
    ElementSpacing = mhd_dict['ElementSpacing']
    Dim = mhd_dict['DimSize']

    freesurferCoordinate = [0 for j in range(3)]
    freesurferCoordinate[0] = int(math.floor((-coordinate[0]) + Dim[0] / 2 * ElementSpacing[0] + Offset[0]))
    freesurferCoordinate[1] = int(math.floor(coordinate[1] - Dim[1] / 2 * ElementSpacing[1] - Offset[1]))
    freesurferCoordinate[2] = int(math.floor(coordinate[2] - Dim[2] / 2 * ElementSpacing[2] - Offset[2]))

    return coordinate


def surf_id_region_voxel_dicts(surf_id_region_coordinate_dicts, mhd_dict):
    id_region_dict = surf_id_region_coordinate_dicts[0]
    region_coordinate_dict = surf_id_region_coordinate_dicts[1]

    region_voxel_dict = {}

    for id in id_region_dict.keys():
        region_voxel_list = []
        for Point in region_coordinate_dict[id_region_dict[id]]:
            voxel_point = freesurferCoordinate_2_voxel(Point, mhd_dict)
            if voxel_point not in region_voxel_list:
                region_voxel_list.append(voxel_point)
        region_voxel_dict.setdefault(id_region_dict[id], region_voxel_list)

    surf_id_region_voxel_dicts = [id_region_dict, region_voxel_dict]

    return surf_id_region_voxel_dicts


def fiber_points_list(fiber_polydata):
    # ============================= Get Fiber-Points Set ============================
    # fiber_point_list[] is a 3D list;
    # fiber_point_list[i] is the total points of ith fiber;
    # fiber_point_list[i][j] is the jth point of fiber(i);
    # fiber_point_list[i][j][0] is the x coordinate of the jth point of fiber(i);
    # fiber_point_list[i][j][1] is the y coordinate of the jth point of fiber(i);
    # fiber_point_list[i][j][2] is the z coordinate of the jth point of fiber(i);
    # ===========================================================================

    fiber_point_list = [[] for i in range(fiber_polydata.GetNumberOfCells())]

    for fiber in range(fiber_polydata.GetNumberOfCells()):
        pts = fiber_polydata.GetCell(fiber).GetPoints()
        fiber_point_list[fiber] = [pts.GetPoint(point) for point in range(pts.GetNumberOfPoints())]
    return fiber_point_list


def fiber_voxel_list(fiber_point_list, mhd_dict, fiber_polydata):

    fiber_voxel_list = [[] for i in range(fiber_polydata.GetNumberOfCells())]

    for fiber in range(fiber_polydata.GetNumberOfCells()):
        for point in fiber_point_list[fiber]:
            voxel_point = coordinate_2_voxel(point, mhd_dict)
            if voxel_point not in fiber_voxel_list[fiber]:
                fiber_voxel_list[fiber].append(voxel_point)

    return fiber_voxel_list


def voxel_fiber_list(fiber_point_list, mhd_dict, fiber_polydata):

    Dim = mhd_dict['DimSize']

    # be careful about the index of fiber_index_list, the order is z,y,x!! fiber_index_list[z][y][x]
    voxel_fiber_list = [[[[] for x in range(Dim[0])] for y in range(Dim[1])] for z in range(Dim[2])]

    for fiber in range(fiber_polydata.GetNumberOfCells()):
        for point in fiber_point_list[fiber]:
            voxel_point = coordinate_2_voxel(point, mhd_dict)
            if fiber not in voxel_fiber_list[voxel_point[2]][voxel_point[1]][voxel_point[0]]:
                voxel_fiber_list[voxel_point[2]][voxel_point[1]][voxel_point[0]].append(fiber)
            # fiber_index_list[z][y][x], pay attention to the order of z,y,x!!!

    return voxel_fiber_list


def region_fiber_dicts(surf_id_region_voxel_dicts, voxel_fiber_list):
    id_region_dict = surf_id_region_voxel_dicts[0]
    region_voxel_dict = surf_id_region_voxel_dicts[1]

    region_fiber_dict = {}

    for id in id_region_dict.keys():
        region_fiber_list = []
        for voxel in region_voxel_dict[id_region_dict[id]]:
            for fiber in voxel_fiber_list[voxel[2]][voxel[1]][voxel[0]]:
                if fiber not in region_fiber_list:
                    region_fiber_list.append(fiber)
        region_fiber_dict.setdefault(id_region_dict[id], region_fiber_list)

    region_fiber_dicts = [id_region_dict, region_fiber_dict]

    return region_fiber_dicts


def lists_operations(listA,listB):
    retA = [i for i in listA if i in listB]
    retB = list(set(listA).intersection(set(listB)))

    print("retA is: ",retA)
    print("retB is: ",retB)

    retC = list(set(listA).union(set(listB)))
    print("retC1 is: ",retC)

    retD = list(set(listB).difference(set(listA)))
    print("retD is: ",retD)

    retE = [i for i in listB if i not in listA]
    print("retE is: ",retE)


def common_fibers_of_2regions(region_fiber_dicts):
    id_region_dict = region_fiber_dicts[0]
    region_fiber_dict = region_fiber_dicts[1]

    common_fiber_of_2regions_dict = {}
    for id1 in id_region_dict.keys():
        for id2 in id_region_dict.keys():
            region1 = id_region_dict[id1]
            region2 = id_region_dict[id2]
            common_fiber_of_2regions_dict.setdefault(str(id1)+'_'+str(id2)+'_'+region1+'_and_'+region2, list(set(region_fiber_dict[region1]).intersection(set(region_fiber_dict[region2]))))

    common_fiber_of_2regions_dicts = [id_region_dict, common_fiber_of_2regions_dict]

    return common_fiber_of_2regions_dicts


def common_fiber_matrix(common_fiber_of_2regions_dicts):
    id_region_dict = common_fiber_of_2regions_dicts[0]
    region_fiber_dict = common_fiber_of_2regions_dicts[1]
    matrixSize = len(id_region_dict.keys())
    common_fiber_matrix = np.zeros(shape=(matrixSize, matrixSize), dtype=np.float64)
    id_region_key_list = sorted(id_region_dict.keys())
    print(id_region_key_list)
    print(id_region_dict.keys())
    for row in range(matrixSize):
        for col in range(matrixSize):
            id1 = id_region_key_list[row]
            id2 = id_region_key_list[col]
            regionPair = str(id1)+'_'+str(id2)+'_'+id_region_dict[id1]+'_and_'+id_region_dict[id2]
            common_fiber_matrix[row][col] = len(region_fiber_dict[regionPair])

    np.save("common_fiber_matrix.npy", common_fiber_matrix)
    np.savetxt("common_fiber_matrix.txt", common_fiber_matrix, fmt='%-8.1f')

    return common_fiber_matrix


def normalization(matrix, accuracy):
    max_data = matrix.max()
    min_data = matrix.min()
    print("=========Raw data======")
    print("max_value, min_value: ", matrix.max(), matrix.min())
    if max_data == 0 and min_data == 0:
        normalized_matrix = matrix
    else:
        normalized_matrix = (matrix - min_data) / (max_data - min_data)

    normalized_matrix = np.around(normalized_matrix, decimals=accuracy)

    print("=========Normalized data======")
    print("max_value, min_value: ", normalized_matrix.max(), normalized_matrix.min())

    return normalized_matrix


def region_average_fmri_signal_dicts(surf_id_region_voxel_dicts, fmri_nii_data):
    id_region_dict = surf_id_region_voxel_dicts[0]
    region_voxel_dict = surf_id_region_voxel_dicts[1]
    region_average_fmri_signal_dict = {}

    timePoints = fmri_nii_data.shape[3]
    for id in id_region_dict.keys():
        signal_sum = np.zeros(shape=(timePoints,))
        if len(region_voxel_dict[id_region_dict[id]]) == 0:
            signal_average = signal_sum
        else:
            for voxel in region_voxel_dict[id_region_dict[id]]:
                signal_sum = signal_sum + fmri_nii_data[voxel[0]][voxel[1]][voxel[2]][:]
            print(signal_sum)
            print(len(region_voxel_dict[id_region_dict[id]]))
            signal_average = signal_sum / len(region_voxel_dict[id_region_dict[id]])
        region_average_fmri_signal_dict.setdefault(id_region_dict[id], signal_average)

    region_average_fmri_signal_dicts = [id_region_dict, region_average_fmri_signal_dict]

    return region_average_fmri_signal_dicts


def signal_corelation(signal1, signal2):
    if  np.all(signal1 == 0) or np.all(signal2 == 0):
        pcc = 0
    else:
        pcc, p_value = pearsonr(signal1, signal2)
    return pcc


def region_fmri_feature_matrix(region_average_fmri_signal_dicts, TimeWindow):
    id_region_dict = region_average_fmri_signal_dicts[0]
    region_average_fmri_signal_dict = region_average_fmri_signal_dicts[1]
    id_region_key_list = sorted(id_region_dict.keys())
    print (id_region_key_list)
    print (id_region_dict.keys())

    nodeNumber = len(id_region_dict.keys())
    timePoints = len(region_average_fmri_signal_dict[id_region_dict[6]])

    print ("the nodeNumber is:", nodeNumber)
    print ("the timePoints is:", timePoints)

    if 0 >= TimeWindow:
        print ("An wrong TimeWindow!")
    if 1 == TimeWindow:
        for timepoint in range(timePoints):
            region_fmri_feature_matrix = np.zeros(shape=(nodeNumber, 1))
            for node in range(nodeNumber):
                id = id_region_key_list[node]
                region = id_region_dict[id]
                region_fmri_feature_matrix[node] = region_average_fmri_signal_dict[region][timepoint]
            np.save("raw_fmri_feature_matrix_"+str(timepoint)+".npy", region_fmri_feature_matrix)
            np.savetxt("raw_fmri_feature_matrix_"+str(timepoint)+".txt", region_fmri_feature_matrix, fmt='%8.8f')

    if 1 < TimeWindow:
        timePeriods = int(math.floor(timePoints / TimeWindow))
        print ("the timePeriods is:", timePeriods)
        for timeperiod in range(timePeriods):
            region_fmri_feature_matrix = np.zeros(shape=(nodeNumber, nodeNumber))
            for node1 in range(nodeNumber):
                for node2 in range(nodeNumber):
                    id1 = id_region_key_list[node1]
                    region1 = id_region_dict[id1]
                    id2 = id_region_key_list[node2]
                    region2 = id_region_dict[id2]
                    signal1 = region_average_fmri_signal_dict[region1][(timeperiod * TimeWindow):((timeperiod+1) * TimeWindow)]
                    signal2 = region_average_fmri_signal_dict[region2][(timeperiod * TimeWindow):((timeperiod+1) * TimeWindow)]
                    region_fmri_feature_matrix[node1][node2] = signal_corelation(signal1, signal2)
            np.save("pcc_fmri_feature_matrix_"+str(timeperiod)+".npy", region_fmri_feature_matrix)
            np.savetxt("pcc_fmri_feature_matrix_"+str(timeperiod)+".txt", region_fmri_feature_matrix, fmt='%12.8f')


def matrix_2_hotMap():
    return


def test():
    '''
    """test surf_region_coordinate_dict"""
    lh = nfs.read_annot(lh_annot_file)
    rh = nfs.read_annot(rh_annot_file)

    for id in np.unique(lh[0]):
        if id == 75:
            index = [i for i in range(len(lh[0])) if lh[0][i] == id]
            Point = lh_polydata.GetPoints().GetPoint(index[6])
            print Point
    for id in np.unique(rh[0]):
        if id == 75:
            index = [i for i in range(len(rh[0])) if rh[0][i] == id]
            Point = rh_polydata.GetPoints().GetPoint(index[6])
            print Point


    """test fiber_voxel_list"""
    #subjID:   ADNI 002_S_0413
    #fiberID:  36, 72, 11111

    T1_nii = nib.load(T1_nifti_file)
    test_fiber_list = [36, 72, 11111]
    for test_fiber in test_fiber_list:
        matrix = np.zeros(T1_nii.shape)
        print 'test fiber_voxel: ', test_fiber
        for voxel in fiber_voxel_list[test_fiber]:
            x = voxel[0]
            y = voxel[1]
            z = voxel[2]
            matrix[x][y][z] = 36
        new_img = nib.Nifti1Image(matrix, T1_nii.affine, T1_nii.header)
        output_nii_file = "test_fiber_voxel_" + str(test_fiber) + ".nii.gz"
        nib.save(new_img, output_nii_file)


    """test surf_id_region_voxel_dicts"""
    #subjID:   ADNI 002_S_0413
    #regionID 16, 27, 116, 127

    id_region_dict = surf_id_region_voxel_dicts[0]
    region_voxel_dict = surf_id_region_voxel_dicts[1]
    T1_nii = nib.load(T1_nifti_file)
    test_region_list = [16, 27, 116, 127]
    for test_region in test_region_list:
        matrix = np.zeros(T1_nii.shape)
        print 'test_region_voxel: ', test_region, id_region_dict[test_region]
        for voxel in region_voxel_dict[id_region_dict[test_region]]:
            x = voxel[0]
            y = voxel[1]
            z = voxel[2]
            matrix[x][y][z] = 36
        new_img = nib.Nifti1Image(matrix, T1_nii.affine, T1_nii.header)
        output_nii_file = "test_region_voxel_" + str(id_region_dict[test_region]) + '_' + str(test_region) + ".nii.gz"
        nib.save(new_img, output_nii_file)


    """test region_fiber_dicts"""
    #subjID:   ADNI 002_S_0413
    #regionID 16, 116

    id_region_dict = region_fiber_dicts[0]
    region_fiber_dict = region_fiber_dicts[1]
    T1_nii = nib.load(T1_nifti_file)
    matrix = np.zeros(T1_nii.shape)
    test_region_list = [27, 127]
    for test_region in test_region_list:
        print 'test_region_fiber: ', test_region, id_region_dict[test_region]
        matrix = np.zeros(T1_nii.shape)
        for fiber in region_fiber_dict[id_region_dict[test_region]]:
            print fiber
            for voxel in fiber_voxel_list[fiber]:
                x = voxel[0]
                y = voxel[1]
                z = voxel[2]
                matrix[x][y][z] = 36
        new_img = nib.Nifti1Image(matrix, T1_nii.affine, T1_nii.header)
        output_nii_file = "test_region_fiber_" + str(id_region_dict[test_region]) + '_' + str(test_region) + ".nii.gz"
        nib.save(new_img, output_nii_file)


    """test common_fibers_of_2_regions"""
    # subjID:   ADNI 002_S_0413

    id_region_dict = common_fiber_of_2regions_dicts[0]
    common_fiber_of_2regions_dict = common_fiber_of_2regions_dicts[1]
    T1_nii = nib.load(T1_nifti_file)

    test_regionsPair_list = [[16, 116], [27, 127], [16, 27]]
    for regionsPair in test_regionsPair_list:
        test_region1 = id_region_dict[regionsPair[0]]
        test_region2 = id_region_dict[regionsPair[1]]

        print 'region1:', regionsPair[0], test_region1
        print 'region2:', regionsPair[1], test_region2
        matrix = np.zeros(T1_nii.shape)
        for fiber in common_fiber_of_2regions_dict[test_region1+'_and_'+test_region2]:
            print fiber
            for voxel in fiber_voxel_list[fiber]:
                x = voxel[0]
                y = voxel[1]
                z = voxel[2]
                matrix[x][y][z] = 36
        new_img = nib.Nifti1Image(matrix, T1_nii.affine, T1_nii.header)
        output_nii_file = "test_common_fibers_of_2_regions_" + str(regionsPair[0])+'_and_'+str(regionsPair[1]) + ".nii.gz"
        nib.save(new_img, output_nii_file)


    """test common_fiber_matrix"""
    # subjID:   ADNI 002_S_0413

    id_region_dict = common_fiber_of_2regions_dicts[0]
    common_fiber_of_2regions_dict = common_fiber_of_2regions_dicts[1]
    for i in range(150):
        for j in range(150):
            print common_fiber_matrix[i][j]
            id1 = id_region_dict.keys()[i]
            id2 = id_region_dict.keys()[j]
            region1 = id_region_dict[id1]
            region2 = id_region_dict[id2]
            print len(common_fiber_of_2regions_dict[str(id1)+'_'+str(id2)+'_'+region1+'_and_'+region2])

    '''

if __name__ == '__main__':
    lh_surf_vtk_file = '/Users/zl/Programing//002_S_0413/Surf/vtk/lh.white.vtk'
    # rh_surf_vtk_file = '/Users/zl/Programing/002_S_0413/Surf/vtk/rh.white.vtk'
    # fiber_vtk_file = '/Users/zl/Programing/002_S_0413/Fiber/preprocessing/fiber/MedINRIA_StreamLine/fiber.vtk'
    # lh_polydata = read_vtk_file(lh_surf_vtk_file)
    # rh_polydata = read_vtk_file(rh_surf_vtk_file)
    # fiber_polydata = read_vtk_file(fiber_vtk_file)
    #
    # lh_annot_file = '/Users/zl/Programing/002_S_0413/Surf/label/lh.aparc.a2009s.annot'
    # rh_annot_file = '/Users/zl/Programing/002_S_0413/Surf/label/rh.aparc.a2009s.annot'
    # # surf_region_index_list = surf_region_index_list(lh_annot_file, rh_annot_file)
    # surf_id_region_index_dicts_deUknown = surf_id_region_index_dicts_deUknown(lh_annot_file, rh_annot_file)
    # surf_id_region_index_dicts_whole = surf_id_region_index_dicts_whole(lh_annot_file, rh_annot_file)
    # print surf_id_region_index_dicts_deUknown[0]
    # print surf_id_region_index_dicts_whole[0]

    # mhd_file = '/data/lu_zhang/Documents/Data/ADNI_preprocessed/002_S_0413/Surf/mhd/T12dti_b0_l.mhd'
    # mhd_dict = read_meta_header(mhd_file)
    #
    # T1_nifti_file = '/data/lu_zhang/Documents/Data/ADNI_preprocessed/002_S_0413/NIFTI/Accelerated_Sagittal_MPRAGE/T12dti_b0_l.nii.gz'
    # fmri_nifti_file = '/data/lu_zhang/Documents/Data/ADNI_preprocessed/002_S_0413/fMRI/rsfProcess/fmriReadyRetrend.nii.gz'
    # T1_nii_data = read_nifti_file(T1_nifti_file)
    # fmri_nii_data = read_nifti_file(fmri_nifti_file)
    #
    # surf_id_region_coordinate_dicts = surf_id_region_coordinate_dicts(lh_polydata, rh_polydata, surf_id_region_index_dicts_deUknown)
    # surf_id_region_voxel_dicts = surf_id_region_voxel_dicts(surf_id_region_coordinate_dicts, mhd_dict)
    #
    # fiber_point_list = fiber_points_list(fiber_polydata)
    # fiber_voxel_list = fiber_voxel_list(fiber_point_list, mhd_dict, fiber_polydata)
    # voxel_fiber_list = voxel_fiber_list(fiber_point_list, mhd_dict, fiber_polydata)
    # region_fiber_dicts = region_fiber_dicts(surf_id_region_voxel_dicts, voxel_fiber_list)
    # common_fiber_of_2regions_dicts = common_fibers_of_2regions(region_fiber_dicts)
    # common_fiber_matrix = common_fiber_matrix(common_fiber_of_2regions_dicts)
    # # normalized_common_fiber_matrix = normalization(common_fiber_matrix, 6)
    #
    # normalized_fmri_nii_data = normalization(fmri_nii_data, 8)
    # region_average_fmri_signal_dicts = region_average_fmri_signal_dicts(surf_id_region_voxel_dicts, normalized_fmri_nii_data)
    # region_fmri_feature_matrix(region_average_fmri_signal_dicts, 30)

    # test()

    # {1: 'lh_G&S_frontomargin', 2: 'lh_G&S_occipital_inf', 3: 'lh_G&S_paracentral', 4: 'lh_G&S_subcentral', 5: 'lh_G&S_transv_frontopol', 6: 'lh_G&S_cingul-Ant', 7: 'lh_G&S_cingul-Mid-Ant', 8: 'lh_G&S_cingul-Mid-Post', 9: 'lh_G_cingul-Post-dorsal', 10: 'lh_G_cingul-Post-ventral', 11: 'lh_G_cuneus', 12: 'lh_G_front_inf-Opercular', 13: 'lh_G_front_inf-Orbital', 14: 'lh_G_front_inf-Triangul', 15: 'lh_G_front_middle', 16: 'lh_G_front_sup', 17: 'lh_G_Ins_lg&S_cent_ins', 18: 'lh_G_insular_short', 19: 'lh_G_occipital_middle', 20: 'lh_G_occipital_sup', 21: 'lh_G_oc-temp_lat-fusifor', 22: 'lh_G_oc-temp_med-Lingual', 23: 'lh_G_oc-temp_med-Parahip', 24: 'lh_G_orbital', 25: 'lh_G_pariet_inf-Angular', 26: 'lh_G_pariet_inf-Supramar', 27: 'lh_G_parietal_sup', 28: 'lh_G_postcentral', 29: 'lh_G_precentral', 30: 'lh_G_precuneus', 31: 'lh_G_rectus', 32: 'lh_G_subcallosal', 33: 'lh_G_temp_sup-G_T_transv', 34: 'lh_G_temp_sup-Lateral', 35: 'lh_G_temp_sup-Plan_polar', 36: 'lh_G_temp_sup-Plan_tempo', 37: 'lh_G_temporal_inf', 38: 'lh_G_temporal_middle', 39: 'lh_Lat_Fis-ant-Horizont', 40: 'lh_Lat_Fis-ant-Vertical', 41: 'lh_Lat_Fis-post', 42: 'lh_Medial_wall', 43: 'lh_Pole_occipital', 44: 'lh_Pole_temporal', 45: 'lh_S_calcarine', 46: 'lh_S_central', 47: 'lh_S_cingul-Marginalis', 48: 'lh_S_circular_insula_ant', 49: 'lh_S_circular_insula_inf', 50: 'lh_S_circular_insula_sup', 51: 'lh_S_collat_transv_ant', 52: 'lh_S_collat_transv_post', 53: 'lh_S_front_inf', 54: 'lh_S_front_middle', 55: 'lh_S_front_sup', 56: 'lh_S_interm_prim-Jensen', 57: 'lh_S_intrapariet&P_trans', 58: 'lh_S_oc_middle&Lunatus', 59: 'lh_S_oc_sup&transversal', 60: 'lh_S_occipital_ant', 61: 'lh_S_oc-temp_lat', 62: 'lh_S_oc-temp_med&Lingual', 63: 'lh_S_orbital_lateral', 64: 'lh_S_orbital_med-olfact', 65: 'lh_S_orbital-H_Shaped', 66: 'lh_S_parieto_occipital', 67: 'lh_S_pericallosal', 68: 'lh_S_postcentral', 69: 'lh_S_precentral-inf-part', 70: 'lh_S_precentral-sup-part', 71: 'lh_S_suborbital', 72: 'lh_S_subparietal', 73: 'lh_S_temporal_inf', 74: 'lh_S_temporal_sup', 75: 'lh_S_temporal_transverse', 101: 'rh_G&S_frontomargin', 102: 'rh_G&S_occipital_inf', 103: 'rh_G&S_paracentral', 104: 'rh_G&S_subcentral', 105: 'rh_G&S_transv_frontopol', 106: 'rh_G&S_cingul-Ant', 107: 'rh_G&S_cingul-Mid-Ant', 108: 'rh_G&S_cingul-Mid-Post', 109: 'rh_G_cingul-Post-dorsal', 110: 'rh_G_cingul-Post-ventral', 111: 'rh_G_cuneus', 112: 'rh_G_front_inf-Opercular', 113: 'rh_G_front_inf-Orbital', 114: 'rh_G_front_inf-Triangul', 115: 'rh_G_front_middle', 116: 'rh_G_front_sup', 117: 'rh_G_Ins_lg&S_cent_ins', 118: 'rh_G_insular_short', 119: 'rh_G_occipital_middle', 120: 'rh_G_occipital_sup', 121: 'rh_G_oc-temp_lat-fusifor', 122: 'rh_G_oc-temp_med-Lingual', 123: 'rh_G_oc-temp_med-Parahip', 124: 'rh_G_orbital', 125: 'rh_G_pariet_inf-Angular', 126: 'rh_G_pariet_inf-Supramar', 127: 'rh_G_parietal_sup', 128: 'rh_G_postcentral', 129: 'rh_G_precentral', 130: 'rh_G_precuneus', 131: 'rh_G_rectus', 132: 'rh_G_subcallosal', 133: 'rh_G_temp_sup-G_T_transv', 134: 'rh_G_temp_sup-Lateral', 135: 'rh_G_temp_sup-Plan_polar', 136: 'rh_G_temp_sup-Plan_tempo', 137: 'rh_G_temporal_inf', 138: 'rh_G_temporal_middle', 139: 'rh_Lat_Fis-ant-Horizont', 140: 'rh_Lat_Fis-ant-Vertical', 141: 'rh_Lat_Fis-post', 142: 'rh_Medial_wall', 143: 'rh_Pole_occipital', 144: 'rh_Pole_temporal', 145: 'rh_S_calcarine', 146: 'rh_S_central', 147: 'rh_S_cingul-Marginalis', 148: 'rh_S_circular_insula_ant', 149: 'rh_S_circular_insula_inf', 150: 'rh_S_circular_insula_sup', 151: 'rh_S_collat_transv_ant', 152: 'rh_S_collat_transv_post', 153: 'rh_S_front_inf', 154: 'rh_S_front_middle', 155: 'rh_S_front_sup', 156: 'rh_S_interm_prim-Jensen', 157: 'rh_S_intrapariet&P_trans', 158: 'rh_S_oc_middle&Lunatus', 159: 'rh_S_oc_sup&transversal', 160: 'rh_S_occipital_ant', 161: 'rh_S_oc-temp_lat', 162: 'rh_S_oc-temp_med&Lingual', 163: 'rh_S_orbital_lateral', 164: 'rh_S_orbital_med-olfact', 165: 'rh_S_orbital-H_Shaped', 166: 'rh_S_parieto_occipital', 167: 'rh_S_pericallosal', 168: 'rh_S_postcentral', 169: 'rh_S_precentral-inf-part', 170: 'rh_S_precentral-sup-part', 171: 'rh_S_suborbital', 172: 'rh_S_subparietal', 173: 'rh_S_temporal_inf', 174: 'rh_S_temporal_sup', 175: 'rh_S_temporal_transverse'}


    # "156 subjects average diff threshold(1.0) threshold(0.66)"
    # array([ 15,  15,  26,  37,  53,  55,  65,  72,  79,  89,  89, 100, 100,
    #        103, 111, 129, 139, 146]), array([ 53,  89,  55,  72,  15,  26, 139,  37,  89,  15,  79, 103, 129,
    #        100, 146, 100,  65, 111]))

    # CN
    # (array([ 15,  15,  26,  53,  55,  65,  89,  89, 100, 127, 129, 139]), array([ 53,  89,  55,  15,  26, 139,  15, 127, 129,  89, 100,  65]))

    # MCI
    # (array([  6,  15,  15,  26,  26,  27,  27,  29,  37,  44,  55,  65,  66,
    #         72,  79,  80,  89,  89,  89, 100, 100, 101, 102, 103, 111, 118,
    #        118, 129, 139, 146]), array([ 15,   6,  89,  29,  55,  44,  66,  26,  72,  27,  26, 139,  27,
    #         37,  89,  89,  15,  79,  80, 103, 129, 118, 118, 100, 146, 101,
    #        102, 100,  65, 111]))

    #
    #
    #   0  Unknown                           0   0   0    0
    #   1  G&S_frontomargin                 23 220  60    0
    #   2  G&S_occipital_inf                23  60 180    0
    #   3  G&S_paracentral                  63 100  60    0
    #   4  G&S_subcentral                   63  20 220    0
    #   5  G&S_transv_frontopol             13   0 250    0
    #   6  G&S_cingul-Ant                   26  60   0    0
    #   7  G&S_cingul-Mid-Ant               26  60  75    0
    #   8  G&S_cingul-Mid-Post              26  60 150    0
    #   9  G_cingul-Post-dorsal             25  60 250    0
    #  10  G_cingul-Post-ventral            60  25  25    0
    #  11  G_cuneus                        180  20  20    0
    #  12  G_front_inf-Opercular           220  20 100    0
    #  13  G_front_inf-Orbital             140  60  60    0
    #  14  G_front_inf-Triangul            180 220 140    0
    #  15  G_front_middle                  140 100 180    0
    #  16  G_front_sup                     180  20 140    0
    #  17  G_Ins_lg&S_cent_ins              23  10  10    0
    #  18  G_insular_short                 225 140 140    0
    #  19  G_occipital_middle              180  60 180    0
    #  20  G_occipital_sup                  20 220  60    0
    #  21  G_oc-temp_lat-fusifor            60  20 140    0
    #  22  G_oc-temp_med-Lingual           220 180 140    0
    #  23  G_oc-temp_med-Parahip            65 100  20    0
    #  24  G_orbital                       220  60  20    0
    #  25  G_pariet_inf-Angular             20  60 220    0
    #  26  G_pariet_inf-Supramar           100 100  60    0
    #  27  G_parietal_sup                  220 180 220    0
    #  28  G_postcentral                    20 180 140    0
    #  29  G_precentral                     60 140 180    0
    #  30  G_precuneus                      25  20 140    0
    #  31  G_rectus                         20  60 100    0
    #  32  G_subcallosal                    60 220  20    0
    #  33  G_temp_sup-G_T_transv            60  60 220    0
    #  34  G_temp_sup-Lateral              220  60 220    0
    #  35  G_temp_sup-Plan_polar            65 220  60    0
    #  36  G_temp_sup-Plan_tempo            25 140  20    0
    #  37  G_temporal_inf                  220 220 100    0
    #  38  G_temporal_middle               180  60  60    0
    #  39  Lat_Fis-ant-Horizont             61  20 220    0
    #  40  Lat_Fis-ant-Vertical             61  20  60    0
    #  41  Lat_Fis-post                     61  60 100    0
    #  42  Medial_wall                      25  25  25    0
    #  43  Pole_occipital                  140  20  60    0
    #  44  Pole_temporal                   220 180  20    0
    #  45  S_calcarine                      63 180 180    0
    #  46  S_central                       221  20  10    0
    #  47  S_cingul-Marginalis             221  20 100    0
    #  48  S_circular_insula_ant           221  60 140    0
    #  49  S_circular_insula_inf           221  20 220    0
    #  50  S_circular_insula_sup            61 220 220    0
    #  51  S_collat_transv_ant             100 200 200    0
    #  52  S_collat_transv_post             10 200 200    0
    #  53  S_front_inf                     221 220  20    0
    #  54  S_front_middle                  141  20 100    0
    #  55  S_front_sup                      61 220 100    0
    #  56  S_interm_prim-Jensen            141  60  20    0
    #  57  S_intrapariet&P_trans           143  20 220    0
    #  58  S_oc_middle&Lunatus             101  60 220    0
    #  59  S_oc_sup&transversal             21  20 140    0
    #  60  S_occipital_ant                  61  20 180    0
    #  61  S_oc-temp_lat                   221 140  20    0
    #  62  S_oc-temp_med&Lingual           141 100 220    0
    #  63  S_orbital_lateral               221 100  20    0
    #  64  S_orbital_med-olfact            181 200  20    0
    #  65  S_orbital-H_Shaped              101  20  20    0
    #  66  S_parieto_occipital             101 100 180    0
    #  67  S_pericallosal                  181 220  20    0
    #  68  S_postcentral                    21 140 200    0
    #  69  S_precentral-inf-part            21  20 240    0
    #  70  S_precentral-sup-part            21  20 200    0
    #  71  S_suborbital                     21  20  60    0
    #  72  S_subparietal                   101  60  60    0
    #  73  S_temporal_inf                   21 180 180    0
    #  74  S_temporal_sup                  223 220  60    0
    #  75  S_temporal_transverse           221  60  60    0
    #

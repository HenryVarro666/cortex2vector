import nibabel
import BrainLibrary
from BrainLibrary import *
import os
import shutil
import pdb


def colorDict(ctab_file):

    color_dict = {}

    with open(ctab_file) as ctab:
        ctab = ctab.read()
    ctab_list = ctab.split()

    if len(ctab_list)%6 == 0:
        label_num = len(ctab_list) / 6
    else:
        print ("ctab error!")
        exit()

    for label in range(int(label_num)):
        color_dict.setdefault(ctab_list[label*6+1], []).append(ctab_list[label*6+2:label*6+5])

    return color_dict


def loadLabels(labelFilePath):
    label_file_name_list = [x for x in os.listdir(labelFilePath) if not x.startswith('.')]
    return label_file_name_list


def nameMapping(label_file_name):
    label_name = label_file_name.split('.')[0] + '_' + label_file_name.split('.')[1]
    color_name = label_file_name.split('.')[1]
    vtk_name = label_file_name.split('.')[0]
    return label_name, color_name, vtk_name


def creatVTK(vtk_file, label_file, color, label_name, labelIndex, output_path):
    '''Points part add points and create the dictionary'''
    points = vtk.vtkPoints()
    polygons = vtk.vtkCellArray()

    ROIPointsIndex = nibabel.freesurfer.io.read_label(label_file, read_scalars=False)
    polydata = read_vtk_file(vtk_file)

    for point in range(ROIPointsIndex.shape[0]):
        coordinate = polydata.GetPoints().GetPoint(ROIPointsIndex[point])
        points.InsertNextPoint(coordinate)

    ROIPointsDict = {}
    for point in range(ROIPointsIndex.shape[0]):
        ROIPointsDict.setdefault(ROIPointsIndex[point], point)

    '''Triangle part add the triangles'''
    CellArray = polydata.GetPolys()
    Polygons = CellArray.GetData()
    Cell_num = CellArray.GetNumberOfCells()
    triangle_list_whole = []
    triangle_list_label = []

    for i in range(0, CellArray.GetNumberOfCells()):
        triangle_list_whole.append([Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)])

    # sometimes, want a smaller region to show in paraview, smaller one round, outer round
    # '''a smaller lable region'''
    # for triangle in triangle_list_whole:
    #     if triangle[0] in ROIPointsIndex and triangle[1] in ROIPointsIndex and triangle[2] in ROIPointsIndex:
    #         triangleInLabel = [ROIPointsDict[triangle[0]], ROIPointsDict[triangle[1]], ROIPointsDict[triangle[2]]]
    #         triangle_list_label.append(triangleInLabel)

    '''a lager lable region'''
    for triangle in triangle_list_whole:
        if triangle[0] in ROIPointsIndex or triangle[1] in ROIPointsIndex or triangle[2] in ROIPointsIndex:

            '''add the points belong to a triangle but doesn't belong to this label'''
            if not triangle[0] in ROIPointsDict.keys():
                coordinate = polydata.GetPoints().GetPoint(triangle[0])
                points.InsertNextPoint(coordinate)
                ROIPointsDict.setdefault(triangle[0], len(ROIPointsDict))
            if not triangle[1] in ROIPointsDict.keys():
                coordinate = polydata.GetPoints().GetPoint(triangle[1])
                points.InsertNextPoint(coordinate)
                ROIPointsDict.setdefault(triangle[1], len(ROIPointsDict))
            if not triangle[2] in ROIPointsDict.keys():
                coordinate = polydata.GetPoints().GetPoint(triangle[2])
                points.InsertNextPoint(coordinate)
                ROIPointsDict.setdefault(triangle[2], len(ROIPointsDict))

            triangleInLabel = [ROIPointsDict[triangle[0]], ROIPointsDict[triangle[1]], ROIPointsDict[triangle[2]]]
            triangle_list_label.append(triangleInLabel)

    Colors = vtk.vtkUnsignedCharArray();
    Colors.SetNumberOfComponents(3);
    Colors.SetName("Colors");

    '''Setup four points'''
    for triangle in triangle_list_label:
        # Create the polygon
        polygon = vtk.vtkPolygon()

        '''make a quad'''
        polygon.GetPointIds().SetNumberOfIds(3)
        polygon.GetPointIds().SetId(0, triangle[0])
        polygon.GetPointIds().SetId(1, triangle[1])
        polygon.GetPointIds().SetId(2, triangle[2])

        ''' Add the polygon to a list of polygons'''
        polygons.InsertNextCell(polygon)
        Colors.InsertNextTuple3(float(color[0]), float(color[1]), float(color[2]));

    # setup colors (setting the name to "Colors" is nice but not necessary)
    '''Create a PolyData'''
    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)
    polygonPolyData.GetCellData().SetScalars(Colors);
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    # '''write to a vtk file'''
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output_path + '/' + label_name +'_lager_' + str(labelIndex) + '.vtk')
    writer.Write()


if __name__ == '__main__':
    # Open a file
    rpath = "/mnt/disk1/HCP_luzhang_do/Select_30"
    dirs = os.listdir(rpath)
    for ID in dirs:
        vtk_file_path = rpath + '/' + ID + '/Surf/'
        vtk_file = 'white.vtk'
        ctab_file = rpath + '/' + ID + '/Surf/label/aparc.annot.a2009s.ctab'
        lh_annot_file = rpath + '/' + ID + '/Surf/label/lh.aparc.a2009s.annot'
        rh_annot_file = rpath + '/' + ID + '/Surf/label/rh.aparc.a2009s.annot'
        label_file_path = rpath + '/' + ID +'/Surf/labels' #/media/disk2/%Proposal_data_2021/HCP_data/HCP_1064/100408/labels
        output_path =rpath + '/' + ID +'/ROIs'

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        shutil.copyfile(vtk_file_path+'/'+'lh_fix.white.vtk', output_path+'/'+'lh.white.vtk')
        shutil.copyfile(vtk_file_path+'/'+'rh_fix.white.vtk', output_path+'/'+'rh.white.vtk')

        label_file_name_list = loadLabels(label_file_path)
        color_dict = colorDict(ctab_file)
        surf_id_region_index_dicts_whole = BrainLibrary.surf_id_region_index_dicts_whole(lh_annot_file, rh_annot_file)
        id_region_dict = surf_id_region_index_dicts_whole[0]

        for label_file_name in label_file_name_list:
            print(label_file_name)

            label_name, color_name, vtk_name = nameMapping(label_file_name)
            print(label_name, color_name, vtk_name)
            #pdb.set_trace()
            #color_name = color_name.replace("&", "_and_") ############################ tempoary added ################
            #label_name = label_name.replace("&", "_and_")  ############################ tempoary added #################
            color = color_dict[color_name][0]
            print(color)
            # exit()

            # returns the labelIndex which is also the string name of region?
            labelIndex = list(id_region_dict.keys())[list(id_region_dict.values()).index(label_name)]
            #if labelIndex in [12, 15, 16, 27, 29, 30, 55, 68, 69, 70, 112, 115, 116, 127, 129, 130, 155, 168, 169, 170]:
            print(label_name, label_file_name)
                # exit()
            creatVTK(output_path + '/' + vtk_name + '.' + vtk_file, label_file_path + '/' + label_file_name, color, label_name, labelIndex, output_path)
                # exit()









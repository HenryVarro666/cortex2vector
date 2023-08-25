import os
import vtk
import shutil
import nibabel
from collections import defaultdict
import pdb
#from vtkmodules.vtkCommonColor import vtkNamedColors

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


def combine_rois(root_dir, roi_list, except_list, output_dir, name):
    points_new = vtk.vtkPoints()
    polygons_new = vtk.vtkCellArray()
    num = 0
    for roi in roi_list:
        if roi.split('_')[-1] in except_list:
            print(roi)
        else:
            roi_polydata = read_vtk_file(root_dir + '/' + roi)
            point_num = roi_polydata.GetNumberOfPoints()
            for i in range(point_num):
                coordinate = roi_polydata.GetPoints().GetPoint(i)
                points_new.InsertNextPoint(coordinate)

            cell_num = roi_polydata.GetNumberOfCells()
            CellArray = roi_polydata.GetPolys()
            Polygons = CellArray.GetData()
            for i in range(0, cell_num):
                triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(3)
                polygon.GetPointIds().SetId(0, triangle[0] + num)
                polygon.GetPointIds().SetId(1, triangle[1] + num)
                polygon.GetPointIds().SetId(2, triangle[2] + num)
                polygons_new.InsertNextCell(polygon)
            num = num + point_num

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points_new)
    polygonPolyData.SetPolys(polygons_new)
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output_dir + '/' + name)
    writer.Write()


def combine_rois_colorful(root_dir, roi_list, except_list, output_dir, name):
    points_new = vtk.vtkPoints()
    polygons_new = vtk.vtkCellArray()
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    num = 0
    for roi in roi_list:
        if roi.split('_')[-1] in except_list:
            print(roi)
            roi_polydata = read_vtk_file(root_dir + '/' + roi)
            point_num = roi_polydata.GetNumberOfPoints()
            for i in range(point_num):
                coordinate = roi_polydata.GetPoints().GetPoint(i)
                points_new.InsertNextPoint(coordinate)

            cell_num = roi_polydata.GetNumberOfCells()
            CellArray = roi_polydata.GetPolys()
            Polygons = CellArray.GetData()
            for i in range(0, cell_num):
                triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(3)
                polygon.GetPointIds().SetId(0, triangle[0] + num)
                polygon.GetPointIds().SetId(1, triangle[1] + num)
                polygon.GetPointIds().SetId(2, triangle[2] + num)
                polygons_new.InsertNextCell(polygon)
                cell_scalars = roi_polydata.GetCellData().GetArray('Colors').GetTuple(i)
                Colors.InsertNextTuple3(cell_scalars[0], cell_scalars[1], cell_scalars[2])
            num = num + point_num

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points_new)
    polygonPolyData.SetPolys(polygons_new)
    polygonPolyData.GetCellData().SetScalars(Colors)
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output_dir + '/' + name)
    writer.Write()

def combine_rois_colorful_more(root_dir, roi_list, except_list, output_dir, name):
    #colors = vtkNamedColors()
    points_new = vtk.vtkPoints()
    polygons_new = vtk.vtkCellArray()
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    num = 0
    for roi in roi_list:
        if roi.split('_')[-1] in except_list:
            print(roi)
            roi_polydata = read_vtk_file(root_dir + '/' + roi)
            point_num = roi_polydata.GetNumberOfPoints()
            for i in range(point_num):
                coordinate = roi_polydata.GetPoints().GetPoint(i)
                points_new.InsertNextPoint(coordinate)

            cell_num = roi_polydata.GetNumberOfCells()
            CellArray = roi_polydata.GetPolys()
            Polygons = CellArray.GetData()
            for i in range(0, cell_num):
                triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(3)
                polygon.GetPointIds().SetId(0, triangle[0] + num)
                polygon.GetPointIds().SetId(1, triangle[1] + num)
                polygon.GetPointIds().SetId(2, triangle[2] + num)
                polygons_new.InsertNextCell(polygon)
                cell_scalars = roi_polydata.GetCellData().GetArray('Colors').GetTuple(i)
                Colors.InsertNextTuple3(cell_scalars[0], cell_scalars[1], cell_scalars[2])
            num = num + point_num
        else:
            roi_polydata = read_vtk_file(root_dir + '/' + roi)
            point_num = roi_polydata.GetNumberOfPoints()
            for i in range(point_num):
                coordinate = roi_polydata.GetPoints().GetPoint(i)
                points_new.InsertNextPoint(coordinate)

            cell_num = roi_polydata.GetNumberOfCells()
            CellArray = roi_polydata.GetPolys()
            Polygons = CellArray.GetData()
            for i in range(0, cell_num):
                triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(3)
                polygon.GetPointIds().SetId(0, triangle[0] + num)
                polygon.GetPointIds().SetId(1, triangle[1] + num)
                polygon.GetPointIds().SetId(2, triangle[2] + num)
                polygons_new.InsertNextCell(polygon)
                Colors.InsertNextTuple3(224, 224, 224)
            num = num + point_num

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points_new)
    polygonPolyData.SetPolys(polygons_new)
    polygonPolyData.GetCellData().SetScalars(Colors)
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output_dir + '/' + name)
    writer.Write()

def combine_rois_colorful_all(root_dir, roi_list, output_dir, name):
    points_new = vtk.vtkPoints()
    polygons_new = vtk.vtkCellArray()
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    num = 0
    for roi in roi_list:
        print(roi)
        roi_polydata = read_vtk_file(root_dir + '/' + roi)
        point_num = roi_polydata.GetNumberOfPoints()
        for i in range(point_num):
            coordinate = roi_polydata.GetPoints().GetPoint(i)
            points_new.InsertNextPoint(coordinate)

        cell_num = roi_polydata.GetNumberOfCells()
        CellArray = roi_polydata.GetPolys()
        Polygons = CellArray.GetData()
        for i in range(0, cell_num):
            triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(3)
            polygon.GetPointIds().SetId(0, triangle[0] + num)
            polygon.GetPointIds().SetId(1, triangle[1] + num)
            polygon.GetPointIds().SetId(2, triangle[2] + num)
            polygons_new.InsertNextCell(polygon)
            cell_scalars = roi_polydata.GetCellData().GetArray('Colors').GetTuple(i)
            Colors.InsertNextTuple3(cell_scalars[0], cell_scalars[1], cell_scalars[2])
        num = num + point_num

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points_new)
    polygonPolyData.SetPolys(polygons_new)
    polygonPolyData.GetCellData().SetScalars(Colors)
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output_dir + '/' + name)
    writer.Write()

def delete_roi_from_whole(surf_root, surf, label_root, roi_list, except_list, output_dir, name):
    sphere_list = ['lh', 'rh']
    # except_ID = defaultdict(list)
    except_ID = dict()
    for roi in roi_list:
        if roi.split('_')[-1] in except_list:
            print(roi)
            if roi.split('_')[0] not in sphere_list:
                print(roi.split('_')[0])
                print('label name error')
                exit()
            else:
                label_name = roi.split('_')[0] + '.' + '_'.join(roi.split('_')[1:-2]) + '.label'
                ROIPointsIndex = nibabel.freesurfer.io.read_label(label_root + '/' + label_name, read_scalars=False)
                if roi.split('_')[0] in except_ID.keys():
                    # pdb.set_trace()
                    except_ID[roi.split('_')[0]] = except_ID[roi.split('_')[0]] + list(ROIPointsIndex)
                else:
                    except_ID[roi.split('_')[0]] = list(ROIPointsIndex)

    points_new = vtk.vtkPoints()
    polygons_new = vtk.vtkCellArray()
    count = 0
    print(except_ID.keys())
    for sphere in sphere_list:
        print(surf_root + '/' + sphere + surf)
        surf_polydata = read_vtk_file(surf_root + '/' + sphere + surf)
        point_num = surf_polydata.GetNumberOfPoints()
        for i in range(point_num):
            coordinate = surf_polydata.GetPoints().GetPoint(i)
            points_new.InsertNextPoint(coordinate)

        cell_num = surf_polydata.GetNumberOfCells()
        CellArray = surf_polydata.GetPolys()
        Polygons = CellArray.GetData()
        for i in range(0, cell_num):
            triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
            if not (triangle[0] in except_ID[sphere] and triangle[1] in except_ID[sphere] and triangle[2] in except_ID[sphere]):
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(3)
                polygon.GetPointIds().SetId(0, triangle[0] + count)
                polygon.GetPointIds().SetId(1, triangle[1] + count)
                polygon.GetPointIds().SetId(2, triangle[2] + count)
                polygons_new.InsertNextCell(polygon)
        count = count + point_num

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points_new)
    polygonPolyData.SetPolys(polygons_new)
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output_dir + '/' + name)
    writer.Write()


if __name__ == '__main__':
    root_dir = '/mnt/disk1/HCP_luzhang_do/Select_30'
    subject_list = [sub for sub in os.listdir(root_dir) if not sub.startswith('.') and not sub.endswith('.xlsx')]

    for subject in subject_list:
        print(subject)
        surf_dir = root_dir + '/' + subject + '/' + 'ROIs'
        label_root = root_dir + '/' + subject + '/' + 'Surf/labels'
        roi_output_dir = root_dir + '/' + subject + '/' + 'ROIs_combination'
        if os.path.exists(roi_output_dir):
            shutil.rmtree(roi_output_dir)
        os.makedirs(roi_output_dir)
        roi_input_dir = root_dir + '/' + subject + '/' + 'ROIs'
        roi_list = [roi for roi in os.listdir(roi_input_dir) if not roi.endswith('.') and not roi.endswith('white.vtk')]
        # for hinge in roi_except_dict.keys():
        #     roi_except_list = roi_except_dict[hinge]
            # combine_rois(roi_input_dir, roi_list, roi_except_list, roi_output_dir, '3hing_' + str(hinge) + '_combine.vtk')
        combine_rois_colorful_all(roi_input_dir, roi_list, roi_output_dir, 'ALL_color_combine.white.vtk')

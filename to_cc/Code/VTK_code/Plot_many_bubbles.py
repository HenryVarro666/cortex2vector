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

from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkIOPLY import vtkPLYReader
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersCore import vtkAppendFilter

from vtkmodules.vtkChartsCore import vtkCategoryLegend
from vtkmodules.vtkCommonCore import (
    vtkUnsignedCharArray,
    vtkLookupTable,
    vtkVariantArray,
    vtkPoints
)
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    vtkCellTypes,
    vtkGenericCell
)
from vtkmodules.vtkFiltersExtraction import vtkExtractEdges
from vtkmodules.vtkFiltersGeneral import vtkShrinkFilter, vtkAxes
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

from vtkmodules.vtkFiltersCore import (
    vtkTubeFilter,
    vtkAppendPolyData,
    vtkCleanPolyData
)

from vtkmodules.vtkFiltersSources import (
    vtkConeSource,
    vtkSphereSource
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkIOGeometry import (
    vtkBYUReader,
    vtkOBJReader,
    vtkSTLReader
)
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

########################################################################################################################
from Brain_and_Hings import *

class HTMLToFromRGBAColor:
    @staticmethod
    def RGBToHTMLColor(rgb):
        """
        Convert an [R, G, B] list to #RRGGBB.
        :param: rgb - The elements of the array rgb are unsigned chars (0..255).
        :return: The html color.
        """
        hexcolor = "#" + ''.join(['{:02x}'.format(int(x)) for x in rgb])
        return hexcolor

    @staticmethod
    def HTMLColorToRGB(colorString):
        """
        Convert #RRGGBB to a [R, G, B] list.
        :param: colorString a string in the form: #RRGGBB where RR, GG, BB are hexadecimal.
        The elements of the array rgb are unsigned chars (0..255).
        :return: The red, green and blue components as a list.
        """
        colorString = colorString.strip()
        if colorString[0] == '#':
            colorString = colorString[1:]
        if len(colorString) != 6:
            raise ValueError("Input #%s is not in #RRGGBB format" % colorString)
        r, g, b = colorString[:2], colorString[2:4], colorString[4:]
        r, g, b = [int(n, 16) for n in (r, g, b)]
        return [r, g, b]

    @staticmethod
    def RGBToLumaCCIR601(rgb):
        """
        RGB -> Luma conversion
        Digital CCIR601 (gives less weight to the R and B components)
        :param: rgb - The elements of the array rgb are unsigned chars (0..255).
        :return: The luminance.
        """
        Y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return Y

    @staticmethod
    def FormatRGBForHTML(rgb):
        """
        Format the rgb colors for display on a html table.
        :param: rgb - The elements of the array rgb are unsigned chars (0..255).
        :return: A formatted string for the html table.
        """
        s = ','.join(['{:3d}'.format(x) for x in rgb])
        s = s.replace(' ', '&#160;')
        s = s.replace(',', '&#160;&#160;')
        return s

def ReadPolyData(file_name):
    import os
    path, extension = os.path.splitext(file_name)
    extension = extension.lower()
    if extension == '.ply':
        reader = vtkPLYReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.vtp':
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.obj':
        reader = vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.stl':
        reader = vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.vtk':
        #reader = vtkUnstructuredGridReader()
        reader = vtk.vtkPolyDataReader()
        #reader = vtkXMLPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.g':
        reader = vtkBYUReader()
        reader.SetGeometryFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    else:
        raise "unknown"
    return reader

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

########################################################################################################################
########################################################################################################################

def combine_multi_v(file_list:list, rgb:list, scale:list, smooth_list:list, filename):
    # add many vtk file together into vtp file
    colors = vtkNamedColors()
    color_list = []
    scale_list = []
    #Colors.InsertNextTuple3(*colors.GetColor3ub('Red'))

    appendFilter = vtkAppendFilter()

    points_new = vtk.vtkPoints()
    polygons_new = vtk.vtkCellArray()

    num = 0
    for k, f in enumerate(file_list):
        reader = ReadPolyData(f)

        # ugrid = vtk.vtkUnstructuredGrid()
        # ugrid.SetPoints(points)
        # ugrid.InsertNextCell(voxel.GetCellType(), voxel.GetPointIds())
        # gfilter = vtk.vtkGeometryFilter()
        # gfilter.SetInput(ugrid)
        # gfilter.Update()
        # output_port = reader.GetOutputPort()

        if smooth_list[k]:
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputConnection(reader.GetOutputPort())
            smoother.SetNumberOfIterations(200)
            #smoother.FeatureEdgeSmoothingOn()
            #smoother.SetFeatureAngle(60.0)
            #smoother.SetEdgeAngle(60.0)
            #smoother.SetBoundarySmoothing(1)
            #smoother.SetFeatureEdgeSmoothing(1)
            #smoother.SetRelaxationFactor(0.5)
            smoother.Update()
            #output_port=smoother.GetOutputPort()
            output = smoother.GetOutput()
        else:
            #output_port = reader.GetOutputPort()
            output = reader.GetOutput()

        # appendFilter = vtk.vtkAppendPolyData()
        # appendFilter.AddInputData(output1)
        # appendFilter.AddInputData(output2)
        # appendFilter.Update()
        # cleanFilter = vtkCleanPolyData()
        # cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
        # cleanFilter.Update()
        # polygonMapper = vtkPolyDataMapper()
        # polygonMapper.SetInputConnection(cleanFilter.GetOutputPort())
        point_num = output.GetNumberOfPoints()
        cell_num = output.GetNumberOfCells()
        ##
        print(point_num, cell_num, type(output))

        for i in range(point_num):
            coordinate = output.GetPoints().GetPoint(i)
            points_new.InsertNextPoint(coordinate)

        if rgb[k]:
            CellArray = output.GetPolys()
            #CellArray = output.GetCells()
            Polygons = CellArray.GetData()
            for i in range(0, cell_num):
                triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(3)
                polygon.GetPointIds().SetId(0, triangle[0] + num)
                polygon.GetPointIds().SetId(1, triangle[1] + num)
                polygon.GetPointIds().SetId(2, triangle[2] + num)
                polygons_new.InsertNextCell(polygon)
                #cell_scalars = roi_polydata.GetCellData().GetArray('Colors').GetTuple(i)
                #Colors.InsertNextTuple3(cell_scalars[0], cell_scalars[1], cell_scalars[2])
                color_list.append([int(rgb[k][0]), int(rgb[k][1]), int(rgb[k][2])])
                if scale[k]:
                    scale_list.append(scale[k])
                else:
                    scale_list.append(0)
        else:
            CellArray = output.GetPolys()
            #CellArray = output.GetCells()
            Polygons = CellArray.GetData()
            for i in range(0, cell_num):
                triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(3)
                polygon.GetPointIds().SetId(0, triangle[0] + num)
                polygon.GetPointIds().SetId(1, triangle[1] + num)
                polygon.GetPointIds().SetId(2, triangle[2] + num)
                polygons_new.InsertNextCell(polygon)
                cell_scalars = output.GetCellData().GetArray('Colors').GetTuple(i)
                #print('vtk colors',cell_scalars)
                #Colors.InsertNextTuple3(cell_scalars[0], cell_scalars[1], cell_scalars[2])
                color_list.append([int(cell_scalars[0]), int(cell_scalars[1]), int(cell_scalars[2])])
                if scale[k]:
                    scale_list.append(scale[k])
                else:
                    scale_list.append(0)

        num = num + point_num

    # add colors
    Colors = vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetNumberOfTuples(len(color_list))
    Colors.SetName("Colors")
    for i, c in enumerate(color_list):
        #print('new colors', c)
        #Colors.InsertNextTuple3(c[0], c[1], c[2])
        #hexc = HTMLToFromRGBAColor.RGBToHTMLColor(c)
        Colors.SetTuple3(i, c[0], c[1], c[2])

    Scales = vtkUnsignedCharArray()
    Scales.SetNumberOfComponents(1)
    print("Scales", len(scale_list))
    #Scales.SetNumberOfTuples(len(scale_list))
    Scales.SetName("Scales")
    nn = 0
    for s in scale_list:
        #print(nn, '-', s)
        Scales.InsertNextValue(s)
        nn += 1

    polydata = vtkPolyData()
    polydata.SetPoints(points_new)
    polydata.SetPolys(polygons_new)
    # * Add them to point or cell data with AddArray:
    polydata.GetCellData().AddArray(Scales)
    polydata.GetCellData().AddArray(Colors)
    # * Set one of them to be the default scalars:
    polydata.GetCellData().SetActiveScalars('Colors')
    # Alternatively, you can add the default scalars with
    # polydata.GetCellData().SetScalars(Colors)
    polydata.Modified()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Update()

    # Combine the meshes
    #appendFilter.AddInputData(output)
    #appendFilter.Update()
    return 0

if __name__ == '__main__':
    data_subjects = '/home/yan/Craftable/HCP_zl_project/data_saves/brains.pickle'
    data_hings = '/home/yan/Craftable/HCP_zl_project/data_saves/3hing_objects.pickle'

    with open(data_subjects,'rb') as file:
        brains_dict = pickle.load(file)

    with open(data_hings,'rb') as file:
        hings_dict = pickle.load(file)

    for key in brains_dict.keys():
        print(len(brains_dict[key].good_hings), ' / ', len(brains_dict[key].all_hings))

    # print(brains_dict['137532'])
    # b137532 = brains_dict['137532']
    # good_Hs = b137532.good_hings
    #print(good_Hs)

    for m in brains_dict.keys():
        b = brains_dict[m]
        surf = b.get_allROIs_vtk('/home/yan/Craftable/HCP_zl_project/Select_10')
        plot_list = [surf,]
        cc_list = [None, ]
        ss_list = [255, ]
        smoo_list = [True, ]

        good_Hs = b.good_hings

        for key in good_Hs.keys():
            print(repr(hings_dict[key]))
            vtk_file = hings_dict[key].get_bubble_vtk('/home/yan/Craftable/HCP_zl_project/Select_10')
            print('bub vtk', vtk_file)
            i = hings_dict[key].i
            plot_list.append(vtk_file)
            cc_list.append(None)
            ss_list.append(i)
            smoo_list.append(False)

        print(len(plot_list), len(cc_list), len(ss_list), len(smoo_list))
        #exit()

        outfile = '/home/yan/Craftable/HCP_zl_project/Code/' + b.id+ '_.vtk'
        combine_multi_v(plot_list, cc_list, ss_list, smoo_list, outfile)



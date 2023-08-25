#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:37:40 2021

@author: yan
"""

import sys
import pdb
import numpy as np
import math
import nibabel as nib
from nibabel import freesurfer as nfs
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import pdb
from nibabel import trackvis

def tractography_from_trackvis_file(filename):
    tracts_and_data, header = trackvis.read(filename, points_space='rasmm')
    tracts, scalars, properties = list(zip(*tracts_and_data)) # zip(*) is unzip something
    scalar_names = [n for n in header['scalar_name'] if len(n) > 0]

    tracts_data = {}
    for i, sn in enumerate(scalar_names):
        if hasattr(sn, 'decode'):
            sn = sn.decode()
        tracts_data[sn] = [scalar[:, i][:, None] for scalar in scalars]
    affine = header['vox_to_ras']
    image_dims = header['dim']

    return (tracts, scalars, properties, affine, image_dims)

def assign_color(fib):
    array1 = np.stack(fib[0:-1])
    array2 = np.stack(fib[1:])
    
    p0 = np.array(list(fib[0]))
    pe = np.array(list(fib[-1]))   
    vec = pe - p0
    bvec = np.abs(vec/np.sum(np.abs(vec)))
    cc = np.floor(bvec*255)
    print(cc)
    #xyz = XYZColor(abs(vec[0]), abs(vec[1]), abs(vec[2]))
    tanc = np.array(list(map(np.arctan, bvec))) # 0 - pi/4

    #print(tanc)
    # hsl = convert_color(xyz, HSLColor, through_rgb_type=AdobeRGBColor)
    # print(hsl)
    # rgb_ = colorsys.hls_to_rgb(hsl.hsl_h, hsl.hsl_l, hsl.hsl_s)
    # print(rgb_)
    
    #rgb_ = matplotlib.colors.hsv_to_rgb(bvec)
    #print('rgb is ', rgb_)    
    return cc

def calculate_points_color_for(fiber_coordinate_list, p_id_list):
    array1 = np.stack(fiber_coordinate_list[0:-1])
    array2 = np.stack(fiber_coordinate_list[1:])
    color = abs(array1-array2)/np.repeat(np.sqrt(np.sum((array1-array2)**2, axis=1)).reshape((array1.shape[0], 1)), 3, axis=1)
    #print('fiber_coordinate_list', len(fiber_coordinate_list), 'color.shape', color.shape[0])
    assert len(fiber_coordinate_list) == color.shape[0] + 1
    point_id_color_dict = dict()
    for i, fib in enumerate(fiber_coordinate_list[0:-1]):
    #for i, fib in enumerate(fiber_coordinate_list):
        point_id_color_dict[p_id_list[i]] = color[i]
    point_id_color_dict[p_id_list[-1]] = color[-1]
    print('fiber_coordinate_list', len(fiber_coordinate_list), 'point_id_color_dict', len(point_id_color_dict.keys()))
    #assert len(fiber_coordinate_list) == len(point_id_color_dict.keys())
    return point_id_color_dict

def extend_fib(fib):
    if len(fib) > 39:
        p0 = np.array(fib[0])
        p1 = np.array(fib[1])
        p2 = np.array(fib[-2])
        p3 = np.array(fib[-1])
        p_new_head = p0 + (p0 - p1)
        p_new_head = (p_new_head[0], p_new_head[1], p_new_head[2])
        p_new_tail = p3 + (p3 - p2)
        p_new_tail = (p_new_tail[0], p_new_tail[1], p_new_tail[2])
        return [p_new_head] + fib + [p_new_tail]
    else:
        return fib

def read_tracts(tracts):
    #fileIN = open(filename, "r")
    Fibs = []
    Color_access = []
    for t in tracts:
        # 100 * 3
        fib =[]
        k = t.shape[0]
        for i in range(k):
            xyz = (float(t[i,0]), float(t[i,1]), float(t[i,2]))
            fib.append(xyz)
        Fibs.append(fib)
        fib_rgb = assign_color(fib)
        Color_access.append(fib_rgb)

    return Fibs

def calculate_coordinate_in_dti_space_for_DStudio_fiber(coordinate_in_DStudio_list, t1_hdr):
    new_list = []
    mat = t1_hdr.get_sform()

    Offset = mat[:,-1][0:3]
    ElementSpacing = t1_hdr.get_zooms()
    Dim = t1_hdr.get_data_shape()
    # print(Dim)
    # print(Offset)
    # print(ElementSpacing)
    # print(mat)
    # print(len(coordinate_in_DStudio_list))
    #sys.exit()
    if mat[0,0] < 0 and mat[1,1] < 0 and mat[2,2] > 0:
        for coordinate_in_DStudio in coordinate_in_DStudio_list:
            new_coordinate = np.zeros(shape=coordinate_in_DStudio.shape)
            print(ElementSpacing, Dim, Offset)
            # break
            #new_coordinate[0] = -(-ElementSpacing[0] * Dim[0] - coordinate_in_DStudio[0] + Offset[0])
            #new_coordinate[1] = coordinate_in_DStudio[1] + Offset[1] + ElementSpacing[1] * Dim[1]
            #new_coordinate[2] = coordinate_in_DStudio[2] + Offset[2]
            new_coordinate[0] = coordinate_in_DStudio[0] + Offset[0]
            new_coordinate[1] = coordinate_in_DStudio[1] + Offset[1]
            new_coordinate[2] = coordinate_in_DStudio[2] + Offset[2]
            new_list.append((new_coordinate[0], new_coordinate[1], new_coordinate[2]))

    elif mat[0,0] > 0 and mat[1,1] > 0 and mat[2,2] > 0:
        for coordinate_in_DStudio in coordinate_in_DStudio_list:
            new_coordinate = np.zeros(shape=coordinate_in_DStudio.shape)
            print(ElementSpacing, Dim, Offset)
            new_coordinate[0] = ElementSpacing[0] * Dim[0] + coordinate_in_DStudio[0] - Offset[0]
            new_coordinate[1] = coordinate_in_DStudio[1] + Offset[1] + ElementSpacing[1] * Dim[1]
            new_coordinate[2] = coordinate_in_DStudio[2] + Offset[2]
            new_list.append((new_coordinate[0], new_coordinate[1], new_coordinate[2]))
        
    else:
         sys.exit('The data structure not supported')

    return new_list

if __name__ == '__main__':
    input_fib, output, = sys.argv[1], sys.argv[2] # input .trk and output vtk
    b0 = sys.argv[3] # source dwi file in .nii.gz
    
    image2 = nib.load(b0)
    hd2 = image2.header
    m_v2r = hd2.get_best_affine()
    
    tracts,_,_,_,_ = tractography_from_trackvis_file(input_fib)
    fib_set = read_tracts(tracts)
    
    '''fib_set_e = []
    for fs in fib_set:
        efs = extend_fib(fs)
        fib_set_e.append(efs)
    fib_set = fib_set_e'''
    
    # colors_collect = []
    # for f in fib_set:
    #     colors_dictionary = calculate_points_color_for(fib_set)
    
    point_dict = {}
    points_colors_dict = {}
    # point_dict.setdefault(key, idx)
    Wpolydata = vtk.vtkPolyData()
    WPoints = vtk.vtkPoints()
    
    linepoly = vtk.vtkPolyData()
    lines = vtk.vtkCellArray()
    
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName('Colors')

    pp_list = []
    for i, fib in enumerate(fib_set):
        print('Fiber: ', i, " / ", len(fib_set))
        k = len(fib)
        lines.InsertNextCell(k) ##################### work well for line
        
        fib_numpy_list = []
        fib_id_list = []
        for index in range(k):
            p = fib[index]
            fib_numpy_list.append(np.array(p))
            dict_num = len(point_dict)
            if p in point_dict.keys():
                fib_id_list.append(point_dict[p])
                pass
            else:
                # pdb.set_trace()
                point_dict.setdefault(p, dict_num)
                fib_id_list.append(point_dict[p])
                #WPoints.InsertNextPoint(t_mat.dot(np.append(np.array(p), np.array([1.])).T).T[0:3])
                pp_list.append(np.array(p))
            
            lines.InsertCellPoint(point_dict[p]) ##################### work well for line
            
        color_dictinary_parts = calculate_points_color_for(fib_numpy_list, fib_id_list)
        points_colors_dict.update(color_dictinary_parts)
                # bad try
                #line.InsertNextPoint(point_dict[p])
                #line.GetPointIds().SetId(i, )
            #lines.InsertNextCell(line)
            
    assert len(pp_list) == len(points_colors_dict)
    ## Rotate points
    ## Keep the order in point lists
    new_pp_list = calculate_coordinate_in_dti_space_for_DStudio_fiber(pp_list, hd2)

    ## Insert points, Dye points
        
    for i, p in enumerate(new_pp_list):
        print(i, " / ", len(new_pp_list))
        WPoints.InsertNextPoint(p)
        cc = points_colors_dict[i]
        Colors.InsertNextTuple3(int(cc[0] * 255), int(cc[1] * 255), int(cc[2] * 255))
        #Colors.InsertNextTuple3(int(cc[0]*100)/100, int(cc[1]*100)/100, int(cc[2]*100)/100)
    
    Wpolydata.SetPoints(WPoints)  ##
    Wpolydata.Modified()
    Wpolydata.SetLines(lines)
    Wpolydata.Modified()
    Wpolydata.GetPointData().SetScalars(Colors)
    Wpolydata.Modified()
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(Wpolydata)
    writer.SetFileName(output)
    writer.Write()

# Colors = vtk.vtkUnsignedCharArray()
# Colors.SetNumberOfComponents(3)
# Colors.SetName(“Colors”)
# Colors.InsertNextTuple3(255,0,0)

# for i in range(len(pc)):
# Points.InsertNextPoint(pc[i,&#32;0], pc[i,&#32;1], pc[i,&#32;2])
# Colors.InsertNextTuple3(pc_color[i,&#32;0] / 255, pc_color[i,&#32;1] / 255, pc_color[i,&#32;2] / 255)


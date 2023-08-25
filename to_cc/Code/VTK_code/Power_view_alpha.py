#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:49:43 2021

@author: yanjunlyu
@acknowledge LuZhang
"""

import os
import argparse
import threading
import time
#import multiprocessing.pool
import sys

import nibabel as nib
import numpy as np
import vtk
#from vtk.util.numpy_support import vtk_to_numpy
import pdb
from nibabel import trackvis
import scipy.io
import datetime

###############################################################################
## assitant functions
###############################################################################
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

# def assign_color(fib):
#     p0 = np.array(list(fib[0]))
#     pe = np.array(list(fib[-1]))   
#     vec = pe - p0
#     bvec = np.abs(vec/np.sum(np.abs(vec)))
#     cc = np.floor(bvec*255)
#     print(cc)
#     #xyz = XYZColor(abs(vec[0]), abs(vec[1]), abs(vec[2]))
#     tanc = np.array(list(map(np.arctan, bvec))) # 0 - pi/4

#     #print(tanc)
#     # hsl = convert_color(xyz, HSLColor, through_rgb_type=AdobeRGBColor)
#     # print(hsl)
#     # rgb_ = colorsys.hls_to_rgb(hsl.hsl_h, hsl.hsl_l, hsl.hsl_s)
#     # print(rgb_)
    
#     #rgb_ = matplotlib.colors.hsv_to_rgb(bvec)
#     #print('rgb is ', rgb_)    
#     return cc

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
        #fib_rgb = assign_color(fib)
        #Color_access.append(fib_rgb)
    return Fibs

'''
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
    if mat[0,0] < 0 and mat[1,1] > 0 and mat[2,2] > 0:
        for coordinate_in_DStudio in coordinate_in_DStudio_list:
            new_coordinate = np.zeros(shape=coordinate_in_DStudio.shape)
            # print(new_coordinate, ElementSpacing, Dim, coordinate_in_DStudio, Offset)
            # break
            new_coordinate[0] = -(-ElementSpacing[0] * Dim[0] - coordinate_in_DStudio[0] + Offset[0])
            new_coordinate[1] = coordinate_in_DStudio[1] + Offset[1] + ElementSpacing[1] * Dim[1]
            new_coordinate[2] = coordinate_in_DStudio[2] + Offset[2]
        
            new_list.append((new_coordinate[0], new_coordinate[1], new_coordinate[2]))
    else:
         sys.exit('The data structure not supported')
        #######################################################################
        # if mat[0,0] < 0:
        #     # print(new_coordinate, ElementSpacing, Dim, coordinate_in_DStudio, Offset)
        #     # break
        #     new_coordinate[0] =  coordinate_in_DStudio[0]/ElementSpacing[0] #+ Offset[0]
        # else:
        #     new_coordinate[0] = coordinate_in_DStudio[0]/ElementSpacing[0] #+ Offset[0]
        # if mat[1,1] < 0:
        #     new_coordinate[1] =  coordinate_in_DStudio[1]/ElementSpacing[1] #+ Offset[1]
        # else:
        #     new_coordinate[1] = coordinate_in_DStudio[1]/ElementSpacing[1] #+ Offset[1]
        # if mat[2,2] < 0:
        #     new_coordinate[2] =  coordinate_in_DStudio[2]/ElementSpacing[2] #+ Offset[2]
        # else:
        #     new_coordinate[2] = coordinate_in_DStudio[2]/ElementSpacing[2] #+ Offset[2]
            
        # new_list.append((new_coordinate[0], new_coordinate[1], new_coordinate[2]))
        
    return new_list
'''

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

def Cells_Ids_2_Array(polydata):
        nCells = polydata.GetNumberOfCells()
        pts = polydata.GetCellData()
        stacks = list()
        
        for i in range(nCells):
            pts = polydata.GetCell(i)
            ids = pts.GetPointIds()
            n = ids.GetNumberOfIds()
            array = np.array([ids.GetId(v) for v in range(n)])
            
            stacks.append(array)
        
        return stacks

'''
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
'''

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

def read_tracts(tracts):
    #fileIN = open(filename, "r")
    Fibs = []
    #Color_access = []
    for t in tracts:
        # 100 * 3
        fib =[]
        k = t.shape[0]
        for i in range(k):
            xyz = (float(t[i,0]), float(t[i,1]), float(t[i,2]))
            fib.append(xyz)
        Fibs.append(fib)
        #fib_rgb = assign_color(fib)
        #Color_access.append(fib_rgb)
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

def is_pd(K):
    try:
        np.linalg.cholesky(K)
        return True 
    except np.linalg.linalg.LinAlgError as err:
        #error.orig.message, error.params
        if 'Matrix is not positive definite' in err.args[0]:
            return False
        else:
            raise 

def is_hermitian_positive_semidefinite(X):
    if X.shape[0] != X.shape[1]: # must be a square matrix
        return False

    if not np.all( X - X.T == 0 ): # must be a symmetric or hermitian matrix
        return False

    try: # Cholesky decomposition fails for matrices that are NOT positive definite.

        # But since the matrix may be positive SEMI-definite due to rank deficiency
        # we must regularize.
        regularized_X = X + np.eye(X.shape[0]) * 1e-14

        np.linalg.cholesky(regularized_X)
    except np.linalg.LinAlgError:
        return False

    return True

def native_voxel_in_fsl(head_mat, A, x_N):
    # X is 4 by n, which is vtk.numpy.array.T
    'Initially, please note that the coordinate system used internally within FSL is a scaled mm system \
    (the voxel coordinates multiplied by the voxel dimensions) but is also of fixed handedness, \
    which is achieved by swapping the x voxel coordinate (x goes to N-1-x) if the voxel-to-mm mapping in the qform/sform has positive determinant.'
    eps = 1e-9
    D0 = np.diag(head_mat)
    abs_dim = np.absolute(D0)
    try:
        assert (abs_dim==0).any() == False
    except:
        print('\nCheck input: image orientaion should be (R/L),(A/P),(S/I)\n')
        sys.exit(1)

    Dim = np.diag(abs_dim)

    qsform = head_mat[0:-1, 0:-1]

    pd = is_pd(qsform)
    
    if pd:
        A_xswap = A.copy()
        A_xswap[0,:] = x_N * np.ones(A.shape[1]) - np.ones(A.shape[1]) - A[0,:]
    else:
        A_xswap = A.copy()
    #A_xswap = 
    A_mm = Dim.dot(A_xswap)
    return A_mm

def fsl_in_native_voxel(head_mat, A_mm, x_N):
    # X is 4 by n, which is vtk.numpy.array.T
    'Initially, please note that the coordinate system used internally within FSL is a scaled mm system \
    (the voxel coordinates multiplied by the voxel dimensions) but is also of fixed handedness, \
    which is achieved by swapping the x voxel coordinate (x goes to N-1-x) if the voxel-to-mm mapping in the qform/sform has positive determinant.'
    eps = 1e-9
    D0 = np.diag(head_mat)
    abs_dim = np.absolute(D0)
    try:
        assert (abs_dim==0).any() == False
    except:
        print('\nCheck input: image orientaion should be (R/L),(A/P),(S/I)\n')
        sys.exit(1)

    Dim = np.diag(abs_dim)
    Dim_ = np.linalg.inv(Dim)

    qsform = head_mat[0:-1, 0:-1]

    pd = is_pd(qsform)
    
    if pd:
        A_sawaped = Dim_.dot(A_mm)
        A_ = A_sawaped.copy()
        A_[0,:] = x_N * np.ones(A_mm.shape[1]) - np.ones(A_mm.shape[1]) - A_sawaped[0,:]
    else:
        A_ = Dim_.dot(A_mm)
    return A_

def flirt_explination():
    pass
'''
Initially, please note that the coordinate system used internally within FSL is a scaled mm system (the voxel coordinates multiplied by the voxel dimensions) but is also of fixed handedness, which is achieved by swapping the x voxel coordinate (x goes to N-1-x) if the voxel-to-mm mapping in the qform/sform has positive determinant.

The affine matrix used by FLIRT is constructed in the following way:

mat = rot_mat * skew_mat * scale_mat
The scale_mat has the form:

sx  0  0  0  

 0 sy  0  0  

 0  0 sz  0  

 0  0  0  1

where sx, sy, sz are the three scaling parameters.

The skew_mat has the form:

    1 kxy  kxz  0

    0  1   kyz  0

    0  0    1   0

    0  0    0   1

where kxy, kxz, kyz are the three skew parameters.

The rot_mat has the form:

rot_mat = Rx*Ry*Rz
where Rx has the form:

    1  0  0  0

    0  c  s  0

    0 -s  c  0

    0  0  0  1

where c=cos(theta) and s=sin(theta) and similarly for Ry and Rz (with appropriate axis/dimension swapping).

The FSL convention for transformation matrices uses an implicit centre of transformation - that is, a point that will be unmoved by that transformation, which is an arbitrary choice in general. This arbitrary centre of the transformation for FSL is at the mm origin (0,0,0) which is at the centre of the corner voxel of the image.

When using the transformation parameters from FLIRT, there is an additional complication in that the parameters are calculated in a way that uses a different centre convention: the centre of mass of the volume. The effect of this is that each of the three matrices above end up with an adjustment in the fourth column (top three elements only) that represents a shift between the corner origin and the centre of mass, while the rest of the matrix (first three columns) is unaffected. Once that is done the matrices are multiplied together, as indicated above, and you get your final matrix.

The full 12 parameters are often listed by FLIRT in the following order:

rx ry rz tx ty tz sx sy sz kxy kxz kyz
where rx, ry, rz are the rotation angles in radians (for the matrices Rx, Ry and Rz respectively) and tx, ty, tz are the translations in mm.

'''

###############################################################################
def nii_2_vtk(inputfile, outputname, downsample=True, color_index=0):
    # reconstruct image in RAS order
    image = nib.load(inputfile)

    hd = image.header
    print(hd)
    sp = hd.get_data_shape()
    hd.get_data_offset()
    m_v2r = hd.get_best_affine()
    m_v2base = hd.get_base_affine()
    #m_v2r.dot(np.array([91,109,91,1]))
    #m_v2base.dot(np.array([91,109,91,1]))[:-1]
    vol = np.copy(image.get_data())
    
    points_nii = []
    points_sift = []
    points_basic = []
    points_values = []
    for i in range(0,sp[0]):
        for j in range(0,sp[1]):
            for k in range(0,sp[2]):
                if vol[i,j,k] > 0.0:
                    xyz1_v = np.array([i,j,k,1])
                    points_nii.append((i,j,k))
                    #
                    xyz_sift = m_v2r.dot(xyz1_v)[:-1]
                    points_sift.append(xyz_sift)
                    #
                    xyz_ras_base = m_v2base.dot(xyz1_v)[:-1]
                    points_basic.append(xyz_ras_base)
                    #
                    points_values.append(vol[i,j,k])
    
    # down sample data by 2
    ds_points_nii = []
    ds_points_sift = []
    ds_points_basic = []
    ds_points_values = []
    for i in range(0,sp[0]):
        for j in range(0,sp[1]):
            for k in range(0,sp[2]):
                if i%2 == 0 and j%2 == 0 and k%2 == 0 and vol[i,j,k] > 0.0:
                    xyz1_v = np.array([i,j,k,1])
                    ds_points_nii.append((i,j,k))
                    #
                    xyz_sift = m_v2r.dot(xyz1_v)[:-1]
                    ds_points_sift.append(xyz_sift)
                    #
                    xyz_ras_base = m_v2base.dot(xyz1_v)[:-1]
                    ds_points_basic.append(xyz_ras_base)
                    #
                    ds_points_values.append(vol[i,j,k])
    # make vtk
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    #colors = vtk.vtkNamedColors()
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    
    if downsample:
        data = ds_points_sift
        pv = ds_points_values
    else:
        data = points_sift
        pv = points_values
    
    for n, p in enumerate(data):
        cc = [100,100,100]
        points.InsertNextPoint(p)
        cc[color_index] = int(pv[n])
        #Colors.InsertTuple3(n, cc[0],cc[1],cc[2])
        Colors.InsertNextTuple3(cc[0], cc[1], cc[2])
        print(n , '/',  len(data))
        
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, n)
        
        vertices.InsertNextCell(vertex)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.Modified()
    polydata.SetVerts(vertices)
    polydata.Modified()
    #polydata.GetPointIds()
    polydata.GetCellData().SetScalars(Colors)
    polydata.Modified()
    '''
    appendFilter = vtk.vtkAppendPolyData()
    if vtk.VTK_MAJOR_VERSION <= 5:
        appendFilter.AddInputConnection(polydata.GetProducerPort())
    else:
        appendFilter.AddInputData(polydata)
    
    appendFilter.Update()
    
    #  Remove any duplicate points.
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
    cleanFilter.Update()
    '''
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polydata)
    #writer.SetFileName(bubble_output + '/' + 'Sphere_' + str(Index) + '_' + group + '.vtk')
    writer.SetFileName(outputname)
    #writer.SetFileName(filename)
    writer.Write()
    
    return 0

def nii_2_vtk_vol(inputfile, outputname, downsample=True, color_index=0):
    # reconstruct in original volume order
    image = nib.load(inputfile)

    hd = image.header
    print(hd)
    sp = hd.get_data_shape()
    hd.get_data_offset()
    m_v2r = hd.get_best_affine()
    m_v2base = hd.get_base_affine()
    #m_v2r.dot(np.array([91,109,91,1]))
    #m_v2base.dot(np.array([91,109,91,1]))[:-1]
    vol = np.copy(image.get_data())
    
    points_nii = []
    points_sift = []
    points_basic = []
    points_values = []
    for i in range(0,sp[0]):
        for j in range(0,sp[1]):
            for k in range(0,sp[2]):
                if vol[i,j,k] > 0.0:
                    xyz1_v = np.array([i,j,k,1])
                    points_nii.append((i,j,k))
                    #
                    xyz_sift = m_v2r.dot(xyz1_v)[:-1]
                    points_sift.append(xyz_sift)
                    #
                    xyz_ras_base = m_v2base.dot(xyz1_v)[:-1]
                    points_basic.append(xyz_ras_base)
                    #
                    points_values.append(vol[i,j,k])
    
    # down sample data by 2
    ds_points_nii = []
    ds_points_sift = []
    ds_points_basic = []
    ds_points_values = []
    for i in range(0,sp[0]):
        for j in range(0,sp[1]):
            for k in range(0,sp[2]):
                if i%2 == 0 and j%2 == 0 and k%2 == 0 and vol[i,j,k] > 0.0:
                    xyz1_v = np.array([i,j,k,1])
                    ds_points_nii.append((i,j,k))
                    #
                    xyz_sift = m_v2r.dot(xyz1_v)[:-1]
                    ds_points_sift.append(xyz_sift)
                    #
                    xyz_ras_base = m_v2base.dot(xyz1_v)[:-1]
                    ds_points_basic.append(xyz_ras_base)
                    #
                    ds_points_values.append(vol[i,j,k])
    # make vtk
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    #colors = vtk.vtkNamedColors()
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    
    if downsample:
        data = ds_points_nii
        pv = ds_points_values
    else:
        data = points_nii
        pv = points_values
    
    for n, p in enumerate(data):
        cc = [100,100,100]
        points.InsertNextPoint(p)
        cc[color_index] = int(pv[n])
        #Colors.InsertTuple3(n, cc[0],cc[1],cc[2])
        Colors.InsertNextTuple3(cc[0], cc[1], cc[2])
        print(n , '/',  len(data))
        
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, n)
        
        vertices.InsertNextCell(vertex)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.Modified()
    polydata.SetVerts(vertices)
    polydata.Modified()
    #polydata.GetPointIds()
    polydata.GetCellData().SetScalars(Colors)
    polydata.Modified()
    '''
    appendFilter = vtk.vtkAppendPolyData()
    if vtk.VTK_MAJOR_VERSION <= 5:
        appendFilter.AddInputConnection(polydata.GetProducerPort())
    else:
        appendFilter.AddInputData(polydata)
    
    appendFilter.Update()
    
    #  Remove any duplicate points.
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
    cleanFilter.Update()
    '''
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polydata)
    #writer.SetFileName(bubble_output + '/' + 'Sphere_' + str(Index) + '_' + group + '.vtk')
    writer.SetFileName(outputname)
    #writer.SetFileName(filename)
    writer.Write()
    
    return 0

def mgz_2_vtk(inputfile, outputname, downsample=True, color_index=0):
    import nibabel.freesurfer.mghformat as mgh
    mgh_file = mgh.load(inputfile)

    hd = mgh_file.header
    print(hd)
    sp = mgh_file.get_data().shape
    hd.get_data_offset()
    m_v2r = hd.get_best_affine()
    m_v2rastkr = hd.get_vox2ras_tkr()
    #m_v2r.dot(np.array([91,109,91,1]))
    #m_v2base.dot(np.array([91,109,91,1]))[:-1]
    vol = np.copy(mgh_file.get_data())
    
    points_nii = []
    points_sift = []
    points_tkr = []
    points_values = []
    for i in range(0,sp[0]):
        for j in range(0,sp[1]):
            for k in range(0,sp[2]):
                if vol[i,j,k] > 0.0:
                    xyz1_v = np.array([i,j,k,1])
                    points_nii.append((i,j,k))
                    #
                    xyz_sift = m_v2r.dot(xyz1_v)[:-1]
                    points_sift.append(xyz_sift)
                    #
                    xyz_tkr = m_v2rastkr.dot(xyz1_v)[:-1]
                    points_tkr.append(xyz_tkr)
                    #
                    points_values.append(vol[i,j,k])
    
    # down sample data by 2
    ds_points_nii = []
    ds_points_sift = []
    ds_points_tkr = []
    ds_points_values = []
    for i in range(0,sp[0]):
        for j in range(0,sp[1]):
            for k in range(0,sp[2]):
                if i%2 == 0 and j%2 == 0 and k%2 == 0 and vol[i,j,k] > 0.0:
                    xyz1_v = np.array([i,j,k,1])
                    ds_points_nii.append((i,j,k))
                    #
                    xyz_sift = m_v2r.dot(xyz1_v)[:-1]
                    ds_points_sift.append(xyz_sift)
                    #
                    xyz_tkr = m_v2rastkr.dot(xyz1_v)[:-1]
                    ds_points_tkr.append(xyz_tkr)
                    #
                    ds_points_values.append(vol[i,j,k])
    # make vtk
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    #colors = vtk.vtkNamedColors()
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    
    if downsample:
        data = ds_points_sift
        pv = ds_points_values
    else:
        data = points_sift
        pv = points_values
    
    for n, p in enumerate(data):
        cc = [100,100,100]
        points.InsertNextPoint(p)
        cc[color_index] = int(pv[n])
        #Colors.InsertTuple3(n, cc[0],cc[1],cc[2])
        Colors.InsertNextTuple3(cc[0], cc[1], cc[2])
        print(n , '/',  len(data))
        
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, n)
        
        vertices.InsertNextCell(vertex)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.Modified()
    polydata.SetVerts(vertices)
    polydata.Modified()
    #polydata.GetPointIds()
    polydata.GetCellData().SetScalars(Colors)
    polydata.Modified()
    '''
    appendFilter = vtk.vtkAppendPolyData()
    if vtk.VTK_MAJOR_VERSION <= 5:
        appendFilter.AddInputConnection(polydata.GetProducerPort())
    else:
        appendFilter.AddInputData(polydata)
    
    appendFilter.Update()
    
    #  Remove any duplicate points.
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
    cleanFilter.Update()
    '''
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polydata)
    #writer.SetFileName(bubble_output + '/' + 'Sphere_' + str(Index) + '_' + group + '.vtk')
    writer.SetFileName(outputname)
    #writer.SetFileName(filename)
    writer.Write()
    
    return 0

def mgz_2_vtk_vol(inputfile, outputname, downsample=True, color_index=0):
    import nibabel.freesurfer.mghformat as mgh
    mgh_file = mgh.load(inputfile)

    hd = mgh_file.header
    print(hd)
    sp = mgh_file.get_data().shape
    hd.get_data_offset()
    m_v2r = hd.get_best_affine()
    m_v2rastkr = hd.get_vox2ras_tkr()
    #m_v2r.dot(np.array([91,109,91,1]))
    #m_v2base.dot(np.array([91,109,91,1]))[:-1]
    vol = np.copy(mgh_file.get_data())
    
    points_nii = []
    points_sift = []
    points_tkr = []
    points_values = []
    for i in range(0,sp[0]):
        for j in range(0,sp[1]):
            for k in range(0,sp[2]):
                if vol[i,j,k] > 0.0:
                    xyz1_v = np.array([i,j,k,1])
                    points_nii.append((i,j,k))
                    #
                    xyz_sift = m_v2r.dot(xyz1_v)[:-1]
                    points_sift.append(xyz_sift)
                    #
                    xyz_tkr = m_v2rastkr.dot(xyz1_v)[:-1]
                    points_tkr.append(xyz_tkr)
                    #
                    points_values.append(vol[i,j,k])
    
    # down sample data by 2
    ds_points_nii = []
    ds_points_sift = []
    ds_points_tkr = []
    ds_points_values = []
    for i in range(0,sp[0]):
        for j in range(0,sp[1]):
            for k in range(0,sp[2]):
                if i%2 == 0 and j%2 == 0 and k%2 == 0 and vol[i,j,k] > 0.0:
                    xyz1_v = np.array([i,j,k,1])
                    ds_points_nii.append((i,j,k))
                    #
                    xyz_sift = m_v2r.dot(xyz1_v)[:-1]
                    ds_points_sift.append(xyz_sift)
                    #
                    xyz_tkr = m_v2rastkr.dot(xyz1_v)[:-1]
                    ds_points_tkr.append(xyz_tkr)
                    #
                    ds_points_values.append(vol[i,j,k])
    # make vtk
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    #colors = vtk.vtkNamedColors()
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    
    if downsample:
        data = ds_points_nii
        pv = ds_points_values
    else:
        data = points_nii
        pv = points_values
    
    for n, p in enumerate(data):
        cc = [100,100,100]
        points.InsertNextPoint(p)
        cc[color_index] = int(pv[n])
        #Colors.InsertTuple3(n, cc[0],cc[1],cc[2])
        Colors.InsertNextTuple3(cc[0], cc[1], cc[2])
        print(n , '/',  len(data))
        
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, n)
        
        vertices.InsertNextCell(vertex)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.Modified()
    polydata.SetVerts(vertices)
    polydata.Modified()
    #polydata.GetPointIds()
    polydata.GetCellData().SetScalars(Colors)
    polydata.Modified()
    '''
    appendFilter = vtk.vtkAppendPolyData()
    if vtk.VTK_MAJOR_VERSION <= 5:
        appendFilter.AddInputConnection(polydata.GetProducerPort())
    else:
        appendFilter.AddInputData(polydata)
    
    appendFilter.Update()
    
    #  Remove any duplicate points.
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
    cleanFilter.Update()
    '''
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polydata)
    #writer.SetFileName(bubble_output + '/' + 'Sphere_' + str(Index) + '_' + group + '.vtk')
    writer.SetFileName(outputname)
    #writer.SetFileName(filename)
    writer.Write()
    
    return 0

###############################################################################

def surface_return_t1_ras(surf, T1_mgz_template, outputvtk):
    # use header info of T1.mgz to map surf xyz back to standard RAS space
    import nibabel.freesurfer.mghformat as mgh
    mgh_file = mgh.load(T1_mgz_template)
    hd = mgh_file.header
    print(hd)
    
    '''
    For The freesurfer surf, the xyz cooridinate is in a special RAS_TKR space
    in the corrisponding T1.mgz file, their is a tranfer matrix in the header Info.,
    showing that the transfer fron voxel xyz to RAS_TKR space
    therefore, first is apply inverse on surf cooridinate to restore the voxel space
    1, (inv(xyz_surf) -> xyz0)
    second is apply voxel to RAS matrix on xyz0 to get the cooridinate in RAS space
    2, (xyz0 -> xyz_RAS)
    '''
    sp = mgh_file.get_data().shape
    hd.get_data_offset()
    m_v2ras = hd.get_best_affine()
    m_v2rastkr = hd.get_vox2ras_tkr()
    m_s2v = np.linalg.inv(m_v2rastkr)
    
    import nibabel.freesurfer.io as fsio
    (coords, faces) = fsio.read_geometry(surf)
    
    #(type(coords), coords.shape, coords.dtype)
    ##=> (numpy.ndarray, (150676, 3), dtype('float64'))
    #(type(faces), faces.shape, faces.dtype)
    ##=> (numpy.ndarray, (301348, 3), dtype('>i4'))
    (n, _) = coords.shape
    #ssc = coords.T.shape
    array_n = np.ones(n)
    array_n = np.expand_dims(array_n, axis=1)
    #ssa = array_n.T.shape
    Dat_S = np.concatenate((coords, array_n),axis=1)# n * 4
    #Dat_S.shape
    Dat_V = m_s2v.dot(Dat_S.T)
    Dat_RAS = m_v2ras.dot(Dat_V)
    
    Dat_new = Dat_RAS[:-1,:].copy() # 3 * n
    Dat_new = Dat_new.T # n * 3
    
    
    WPoints = vtk.vtkPoints()
    WTriangles = vtk.vtkCellArray()
    WTriangle = vtk.vtkTriangle()
    for i in range(len(Dat_new)):
        WPoints.InsertNextPoint(Dat_new[i])
        
    for i in range(len(faces)):
        #temp = faces[i,:]  
        WTriangle.GetPointIds().SetId(0, faces[i,0])
        WTriangle.GetPointIds().SetId(1, faces[i,1])
        WTriangle.GetPointIds().SetId(2, faces[i,2])
        WTriangles.InsertNextCell(WTriangle)

    Wpolydata = vtk.vtkPolyData()
    Wpolydata.SetPoints(WPoints)
    Wpolydata.SetPolys(WTriangles)
    Wpolydata.Modified()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(Wpolydata)
    writer.SetFileName(outputvtk)
    writer.Write()

def lines_return_t1_ras(surf_vtk, T1_mgz_template, outputvtk):
    # use header info of T1.mgz to map surf xyz back to standard RAS space
    print('\nCaution !! This function is only used for testing, not validated Yet\n')
    import nibabel.freesurfer.mghformat as mgh
    mgh_file = mgh.load(T1_mgz_template)
    hd = mgh_file.header
    print(hd)
    
    '''
    For The freesurfer surf, the xyz cooridinate is in a special RAS_TKR space
    in the corrisponding T1.mgz file, their is a tranfer matrix in the header Info.,
    showing that the transfer fron voxel xyz to RAS_TKR space
    therefore, first is apply inverse on surf cooridinate to restore the voxel space
    1, (inv(xyz_surf) -> xyz0)
    second is apply voxel to RAS matrix on xyz0 to get the cooridinate in RAS space
    2, (xyz0 -> xyz_RAS)
    '''
    sp = mgh_file.get_data().shape
    hd.get_data_offset()
    m_v2ras = hd.get_best_affine()
    m_v2rastkr = hd.get_vox2ras_tkr()
    m_s2v = np.linalg.inv(m_v2rastkr)
   
    # parse VTK
    polydata = read_vtk_file(surf_vtk)
    Points = polydata.GetPoints()
    array = Points.GetData()
    numpy_nodes = vtk_to_numpy(array)
    Colors = polydata.GetCellData().GetScalars()
    
    cell_stacks = Cells_Ids_2_Array(polydata)
    
    lines = vtk.vtkCellArray()
    (n, _) = numpy_nodes.shape
    #ssc = coords.T.shape
    array_n = np.ones(n)
    array_n = np.expand_dims(array_n, axis=1)
    #ssa = array_n.T.shape
    Dat_S = np.concatenate((numpy_nodes, array_n),axis=1)# n * 4
    Dat_V = m_s2v.dot(Dat_S.T)
    Dat_RAS = m_v2ras.dot(Dat_V)
    Dat_new = Dat_RAS[:-1,:].copy() # 3 * n
    Dat_new = Dat_new.T # n * 3
    print(Dat_new[0:5,:])

    WPoints = vtk.vtkPoints()
    for i in range(Dat_new.shape[0]):
         WPoints.InsertNextPoint(Dat_new[i,:])

    for i, fib in enumerate(cell_stacks):
        k = fib.shape[0]
        lines.InsertNextCell(k)  ##################### work well for line
        for index in range(k):
            lines.InsertCellPoint(fib[index])
        
    Wpolydata = vtk.vtkPolyData()
    Wpolydata.SetPoints(WPoints)  ##
    Wpolydata.Modified()
    Wpolydata.SetLines(lines)
    Wpolydata.Modified()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(Wpolydata)
    writer.SetFileName(outputvtk)
    writer.Write()
    
def surface_return_mgz_vol(surf, T1_mgz_template, outputvtk):
    '''
     These alignment matrices align the volume with what called “FreeSurfer native”. 
     The FreeSurfer native coordinate system is identical to the vertex coordinate system, 
     except for a small translation that is different in every subject.
    '''
    # use header info of T1.mgz to map surf xyz back to standard RAS space
    import nibabel.freesurfer.mghformat as mgh
    mgh_file = mgh.load(T1_mgz_template)
    hd = mgh_file.header
    print(hd)
   
    sp = mgh_file.get_data().shape
    hd.get_data_offset()
    m_v2ras = hd.get_best_affine()
    m_v2rastkr = hd.get_vox2ras_tkr()
    m_s2v = np.linalg.inv(m_v2rastkr)
   
    import nibabel.freesurfer.io as fsio
    (coords, faces) = fsio.read_geometry(surf)

    #(type(coords), coords.shape, coords.dtype)
    ##=> (numpy.ndarray, (150676, 3), dtype('float64'))
    #(type(faces), faces.shape, faces.dtype)
    ##=> (numpy.ndarray, (301348, 3), dtype('>i4'))
    (n, _) = coords.shape
    #ssc = coords.T.shape
    array_n = np.ones(n)
    array_n = np.expand_dims(array_n, axis=1)
    #ssa = array_n.T.shape
    Dat_S = np.concatenate((coords, array_n),axis=1)# n * 4
    #Dat_S.shape
    Dat_V = m_s2v.dot(Dat_S.T)
    #Dat_RAS = m_v2ras.dot(Dat_V)
    
    Dat_new = Dat_V[:-1,:].copy() # 3 * n
    Dat_new = Dat_new.T # n * 3
    
    WPoints = vtk.vtkPoints()
    WTriangles = vtk.vtkCellArray()
    WTriangle = vtk.vtkTriangle()
    for i in range(len(Dat_new)):
        WPoints.InsertNextPoint(Dat_new[i])
        
    for i in range(len(faces)):
        #temp = faces[i,:]  
        WTriangle.GetPointIds().SetId(0, faces[i,0])
        WTriangle.GetPointIds().SetId(1, faces[i,1])
        WTriangle.GetPointIds().SetId(2, faces[i,2])
        WTriangles.InsertNextCell(WTriangle)

    Wpolydata = vtk.vtkPolyData()
    Wpolydata.SetPoints(WPoints)
    Wpolydata.SetPolys(WTriangles)
    Wpolydata.Modified()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(Wpolydata)
    writer.SetFileName(outputvtk)
    writer.Write()

def dsi_fib_to_fa(input_fib, output, fa):
    ''' Works only for HCP or ADNI fa image, which oritation is [-1, 1, 1]'''
    image2 = nib.load(fa)
    hd2 = image2.header
    m_v2r = hd2.get_best_affine()
    '''
    diag=np.diag(m_v2r)
    #print(hd2)
    hd2_shape = np.append(np.array(hd2.get_data_shape()), np.array([0.]))
    #support_mat = np.expand_dims(((np.abs(diag) - diag / 2) * hd2_shape.T), axis=1)
    support_mat = np.diag((np.abs(diag) - diag / 2) * hd2_shape.T)
    a = np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,0]])
    b = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,1]])
    
    _t_mat = support_mat.dot(a) + np.diag(np.diag(m_v2r.dot(b))).dot(a)
    t_mat = np.eye(4,4) + _t_mat
    '''
    ##
    tracts,_,_,_,_ = tractography_from_trackvis_file(input_fib)
    fib_set = read_tracts(tracts)
    
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
    
        #print('fib_id_list', len(fib_id_list))
        #print('fib_numpy_list', len(fib_numpy_list))
        color_dictinary_parts = calculate_points_color_for(fib_numpy_list, fib_id_list)
        points_colors_dict.update(color_dictinary_parts)
    # bad try
    #line.InsertNextPoint(point_dict[p])
    #line.GetPointIds().SetId(i, )
    #lines.InsertNextCell(line)
    print('point size: ',len(pp_list), '  color size: ',len(points_colors_dict))
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
    return 0

def vtk_restore_flirt(input_surf, output, moved_img, ref_image, flirt_mat):
    '''
    Move the vtk file witch matches the moved image back to its original pre-moved image
    output should match the orignal pre-moved image in its RAS
    Flirt suppose to transfer the Voxel to match the reference 
    And copy the Head of the reference image
    '''
    mm_mat = np.loadtxt(flirt_mat)
    mm_mat_inv = np.linalg.inv(mm_mat)

    image_move = nib.load(moved_img)
    hd_move = image_move.header
    print('move', hd_move)
    sp_move = hd_move.get_data_shape()
    m_v2r_move = hd_move.get_best_affine()
    m_v2base_move = hd_move.get_base_affine()

    image_ref = nib.load(ref_image)
    hd_ref = image_ref.header
    print('ref', hd_ref)
    sp_ref = hd_ref.get_data_shape()
    m_v2r_ref = hd_ref.get_best_affine()
    m_v2r_ref_inv = np.linalg.inv(m_v2r_ref)
    m_v2base_ref = hd_ref.get_base_affine()

    points_new = vtk.vtkPoints()
    polygons_new = vtk.vtkCellArray()

    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    # do surf vtk
    polydata = read_vtk_file(input_surf)
    point_num = polydata.GetNumberOfPoints()
    nodes_list = []
    for i in range(point_num):
        coordinate = polydata.GetPoints().GetPoint(i)
        nodes_list.append(coordinate)

    numpy_nodes = np.stack(nodes_list)
    #(n, _) = numpy_nodes.shape
    #ssc = coords.T.shape
    array_n = np.ones(point_num)
    array_n = np.expand_dims(array_n, axis=1)
    #ssa = array_n.T.shape
    Dat_S = np.concatenate((numpy_nodes, array_n),axis=1)# n * 4
    Dat_REF = m_v2r_ref_inv.dot(Dat_S.T)
    Dat_REF_mm = native_voxel_in_fsl(m_v2r_ref , Dat_REF, int(hd_ref['dim'][1]))
    Dat_V_mm = mm_mat_inv.dot(Dat_REF_mm)
    Dat_V = fsl_in_native_voxel(m_v2r_move, Dat_V_mm, int(hd_move['dim'][1]))
    Dat_RAS = m_v2r_move.dot(Dat_V)
    Dat_new = Dat_RAS[:-1,:].copy() # 3 * n
    Dat_new = Dat_new.T # n * 3
    print('finish example\n', Dat_new[0:5,:])

    for i in range(Dat_new.shape[0]):
         points_new.InsertNextPoint(Dat_new[i,:])

    cell_num = polydata.GetNumberOfCells()
    CellArray = polydata.GetPolys()
    Polygons = CellArray.GetData()
    for i in range(0, cell_num):
        triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(3)
        polygon.GetPointIds().SetId(0, triangle[0])
        polygon.GetPointIds().SetId(1, triangle[1])
        polygon.GetPointIds().SetId(2, triangle[2])
        polygons_new.InsertNextCell(polygon)
        #cell_scalars = polydata.GetCellData().GetArray('Colors').GetTuple(i)
        #Colors.InsertNextTuple3(cell_scalars[0], cell_scalars[1], cell_scalars[2])
        
    Wpolydata = vtk.vtkPolyData()
    Wpolydata.SetPoints(points_new)
    Wpolydata.SetPolys(polygons_new)
    Wpolydata.Modified()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(Wpolydata)
    writer.SetFileName(output)
    writer.Write()

def vtk_apply_flirt(input_surf, output, move_img, ref_image, flirt_mat):
    '''
    Apply flirt on the vtk file, output should match the moved image in its RAS
    Flirt suppose to transfer the Voxel to match the reference 
    And copy the Head of the reference image
    '''
    mm_mat = np.loadtxt(flirt_mat)
    #mm_mat_inv = np.linalg.inv(mm_mat)

    image_move = nib.load(move_img)
    hd_move = image_move.header
    #print('move', hd_move)
    sp_move = hd_move.get_data_shape()
    m_v2r_move = hd_move.get_best_affine()
    print('move', hd_move)
    print('x_dim', int(hd_move['dim'][1]))
    m_v2base_move = hd_move.get_base_affine()
    m_v2r_move_inv = np.linalg.inv(m_v2r_move)

    image_ref = nib.load(ref_image)
    hd_ref = image_ref.header
    #print('ref', hd_ref)
    sp_ref = hd_ref.get_data_shape()
    m_v2r_ref = hd_ref.get_best_affine()
    print('ref', hd_ref)
    m_v2base_ref = hd_ref.get_base_affine()

    #sys.exit(1)
    points_new = vtk.vtkPoints()
    polygons_new = vtk.vtkCellArray()

    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    # do surf vtk
    polydata = read_vtk_file(input_surf)
    point_num = polydata.GetNumberOfPoints()
    nodes_list = []
    for i in range(point_num):
        coordinate = polydata.GetPoints().GetPoint(i)
        nodes_list.append(coordinate)

    numpy_nodes = np.stack(nodes_list)
    #(n, _) = numpy_nodes.shape
    #ssc = coords.T.shape
    array_n = np.ones(point_num)
    array_n = np.expand_dims(array_n, axis=1)
    #ssa = array_n.T.shape
    Dat_S = np.concatenate((numpy_nodes, array_n),axis=1)# n * 4
    Dat_V = m_v2r_move_inv.dot(Dat_S.T)
    Dat_mm = native_voxel_in_fsl(m_v2r_move , Dat_V, int(hd_move['dim'][1]))
    Dat_Move_mm = mm_mat.dot(Dat_mm)
    Dat_Move = fsl_in_native_voxel(m_v2r_ref, Dat_Move_mm, int(hd_ref['dim'][1]))
    Dat_RAS = m_v2r_ref.dot(Dat_Move)

    Dat_new = Dat_RAS[:-1,:].copy() # 3 * n
    Dat_new = Dat_new.T # n * 3
    print('finish example\n', Dat_new[0:5,:])

    for i in range(Dat_new.shape[0]):
         points_new.InsertNextPoint(Dat_new[i,:])

    cell_num = polydata.GetNumberOfCells()
    CellArray = polydata.GetPolys()
    Polygons = CellArray.GetData()
    for i in range(0, cell_num):
        triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(3)
        polygon.GetPointIds().SetId(0, triangle[0])
        polygon.GetPointIds().SetId(1, triangle[1])
        polygon.GetPointIds().SetId(2, triangle[2])
        polygons_new.InsertNextCell(polygon)
        #cell_scalars = polydata.GetCellData().GetArray('Colors').GetTuple(i)
        #Colors.InsertNextTuple3(cell_scalars[0], cell_scalars[1], cell_scalars[2])
        
    Wpolydata = vtk.vtkPolyData()
    Wpolydata.SetPoints(points_new)
    Wpolydata.SetPolys(polygons_new)
    Wpolydata.Modified()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(Wpolydata)
    writer.SetFileName(output)
    writer.Write()

def fib_apply_flirt(input_fib, output, move_img, ref_image, flirt_mat):
    print('start: ', datetime.datetime.now())

    mm_mat = np.loadtxt(flirt_mat)
    #mm_mat_inv = np.linalg.inv(mm_mat)

    image_move = nib.load(move_img)
    hd_move = image_move.header
    #print('move', hd_move)
    sp_move = hd_move.get_data_shape()
    m_v2r_move = hd_move.get_best_affine()
    print('move', hd_move)
    print('x_dim', int(hd_move['dim'][1]))
    m_v2base_move = hd_move.get_base_affine()
    m_v2r_move_inv = np.linalg.inv(m_v2r_move)

    image_ref = nib.load(ref_image)
    hd_ref = image_ref.header
    #print('ref', hd_ref)
    sp_ref = hd_ref.get_data_shape()
    m_v2r_ref = hd_ref.get_best_affine()
    print('ref', hd_ref)
    m_v2base_ref = hd_ref.get_base_affine()

    fiber_polydata = read_vtk_file(input_fib)
    fiber_point_num = fiber_polydata.GetNumberOfPoints()
    fiber_line_num = fiber_polydata.GetNumberOfLines()

    Lines = fiber_polydata.GetLines()
    line_scalars = fiber_polydata.GetCellData().GetArray('Colors')
    point_scalars = fiber_polydata.GetPointData().GetArray('Colors')

    Wpolydata = vtk.vtkPolyData()
    WPoints = vtk.vtkPoints()

    coordinate_temp = []
    for i in range(fiber_point_num):
        coordinate_temp.append(fiber_polydata.GetPoints().GetPoint(i))
    nodes_array = np.stack(coordinate_temp)
    print(nodes_array.shape)

    array_n = np.ones(fiber_point_num)
    array_n = np.expand_dims(array_n, axis=1)
    #ssa = array_n.T.shape
    Dat_S = np.concatenate((nodes_array, array_n),axis=1)# n * 4
    Dat_V = m_v2r_move_inv.dot(Dat_S.T)
    Dat_mm = native_voxel_in_fsl(m_v2r_move , Dat_V, int(hd_move['dim'][1]))
    Dat_Move_mm = mm_mat.dot(Dat_mm)
    Dat_Move = fsl_in_native_voxel(m_v2r_ref, Dat_Move_mm, int(hd_ref['dim'][1]))
    Dat_RAS = m_v2r_ref.dot(Dat_Move)

    Dat_new = Dat_RAS[:-1,:].copy() # 3 * n
    Dat_new = Dat_new.T # n * 3
    for i in range(Dat_new.shape[0]):
        #points_new.InsertNextPoint(Dat_new[i,:])
        #print(new_coordinate.shape)
        WPoints.InsertNextPoint(Dat_new[i, :])
        print('Relocating point:, ', i,'/',Dat_new.shape[0])

    Wpolydata.SetPoints(WPoints)  ##
    Wpolydata.Modified()
    Wpolydata.SetLines(Lines)
    Wpolydata.Modified()

    try:
        assert line_scalars.GetTuple((fiber_line_num-1)) is not None
        Wpolydata.GetCellData().SetScalars(line_scalars)
        Wpolydata.Modified()
    except:
        print('\nFound no colors in Line Cells\n')
        pass

    try:
        assert point_scalars.GetTuple((fiber_point_num-1)) is not None
        Wpolydata.GetPointData().SetScalars(point_scalars)
        Wpolydata.Modified()
    except:
        print('\nFound no colors in Points\n')
        pass
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(Wpolydata)
    writer.SetFileName(output)
    writer.Write()
    print('finished fiber: ', datetime.datetime.now())
    return 0

def write_value_on_vtk(input_vtk, output, data_mat,):
    print('to be continue')
    pass

def write_value_by_index(input_vtk, output, data_mat,):
    print('to be continue')
    pass

#################################################################################
## Main
#################################################################################
def Help(help_info):
        print(help_info)
        sys.exit(0)
def main(parser):
    args = parser.parse_args() 
    help_info = "\n python3 Power_view.py -m mod \
    -v <vtk file, for task C F L A Z>\
    -k <trk file, for task D>\
    -s <surf file, white/pial for task S>\
    -i <image vtk/nifti/mgz(always required)> \
    -o <output vtk(always required)> \
    -r <reference b0/T1/None> \
    -d <0/1/None (if do downsample), for task R V M W> \
    -c <0/1/2/None (color index), for task R V M W> \
    -a <matrix .mat file> \
    -x <data_matrix .txt/.npy file>, 'array matrix (n*4)/(n*2), 3D cooridinate and certain value/index value pair'\
    \n-------------------------------------------------------------------------\n \
    \n example: Reverse Filrt \
    python3 Power_view.py -m Z -v ./lh.white.vtk -o ./lh.white.back.vtk -i ./t1_bet.nii.gz -r ./t1_2_b0.nii.gz -a ./t1_2_b0.mat \
    \n example: Surf return 2 T1 in RAS \
    python3 Power_view.py -m S -s ./lh.white -o ./lh.white.vtk -r ./T1.mgz \
    \n example: T1 2 points-cloud vtk in RAS and downsample and die in blue\
    python3 Power_view.py -m R -i ./T1.nii.gz -o ./T1.vtk -d 1 -c 2 \
    \n example: Fiber trk 2 vtk which matches FA.nii in RAS \
    python3 Power_view.py -m D -k ./fib.trk -o ./fib.vtk -r ./fa.nii.gz \
    \n example: Apply filrt Mat to vtk \
    python3 Power_view.py -m A -v ./surf.trk -o ./surf.new_space.vtk -i ./T1_move.nii.gz -r ./T1_reference.nii.gz -a move_2_ref.mat\
    \
    "

    message = 'Nii to vtk in RAS (R),  '\
                 'Nii to vtk in native Volume xyz (V),  '\
                 'Mgz to vtk in RAS (M),  '\
                 'Mgz to vtk in native Volume xyz (G),  '\
                 'Surface to reference Mgz RAS space (S),  '\
                 'Surface to reference Mgz in native Volume space (C),  '\
                 'Dsi_studio Fiber return to fa (D),  '\
                 'Surface to reference Mgz RAS space but lines in VTK file (L),  '\
                 'HELP (H),  '\
                 'Reverse filrt from moved Image and back to vtk (Z), '\
                 'Apply filrt Mat to vtk (A), ' \
                 'Apply filrt Mat to fiber (F), ' \
                 'Write value from voxel to vtk (W), ' \
                 'Write value on vtk by index-value pair(I) \n'

    task_dict = {'R': nii_2_vtk, 'V':nii_2_vtk_vol, 'D':dsi_fib_to_fa,
                 'M':mgz_2_vtk, 'G':mgz_2_vtk_vol, 'L':lines_return_t1_ras,
                 'S':surface_return_t1_ras, 'C':surface_return_mgz_vol,
                 'Z':vtk_restore_flirt, 'A':vtk_apply_flirt, 'F':fib_apply_flirt,
                 'W':write_value_on_vtk, 'I':write_value_by_index, 'H':Help}

    if not args.mod in list(task_dict.keys()):
        #Help(help_info)
        print('Unsupport task\n')
        print('\n-------------------------------------------------------------------------\n')
        #parser.print_help()
        print(message)
        print('\n')
        print('Also check number of para\n')
        sys.exit(1)
    else:
        task = args.mod
        out_file =  args.output
        downsample = args.downsample
        color_index = args.color
        if task == 'H':
            parser.print_help()
            print('\n-------------------------------------------------------------------------\n')
            Help(help_info)
            print('\n-------------------------------------------------------------------------\n')
            print(message)
            sys.exit(0)
        elif task == 'D':
            try:
                assert (args.trk.endswith('trk') and (args.ref.endswith('nii.gz') or args.ref.endswith('nii')))
            except:
                print('Also check para\n')
                Help(help_info)
                sys.exit(1)
            task_dict[task](args.trk, args.output, args.ref)
            #print('Please use <New_Color_fib.py>')

        elif task == 'R':
            try:
                assert (args.image.endswith('nii.gz') or args.image.endswith('nii'))
            except: 
                print('Also check para\n')
                Help(help_info)
                sys.exit(1)
            task_dict[task](args.image, args.output, args.downsample, args.color)
            #nii_2_vtk(inputfile, outputname, downsample=True, color_index=0)

        elif task == 'V':
            try:
                assert (args.image.endswith('nii.gz') or args.image.endswith('nii'))
            except:
                print('Also check para\n')
                Help(help_info)
                sys.exit(1)
            task_dict[task](args.image, args.output, args.downsample, args.color)

        elif task == 'M':
            try:
                assert args.image.endswith('mgz')
            except:
                print('Also check para\n')
                Help(help_info)
                sys.exit(1)
            task_dict[task](args.image, args.output, args.downsample, args.color)
            
        elif task == 'G':
            try:
                assert args.image.endswith('mgz')
            except:
                print('Also check para\n')
                Help(help_info)
                sys.exit(1)
            task_dict[task](args.image, args.output, args.downsample, args.color)

        elif task == 'S':
            try:
                assert (args.surf.endswith('pial') or args.surf.endswith('white')) and args.ref.endswith('mgz')
            except:
                print('Also check para\n')
                Help(help_info)
                sys.exit(1)
            task_dict[task](args.surf, args.ref, args.output)

        elif task == 'L':
            try:
                assert args.vtk.endswith('vtk') and (args.ref.endswith('nii.gz') or args.ref.endswith('nii'))
            except:
                print('Also check para\n')
                Help(help_info)
                sys.exit(1)
            task_dict[task](args.vtk, args.ref, args.output)

        elif task == 'C':
            try:
                assert (args.surf.endswith('pial') or args.surf.endswith('white')) and args.ref.endswith('mgz')
            except:
                print('Also check para\n')
                Help(help_info)
                sys.exit(1)
            task_dict[task](args.surf, args.ref, args.output)

        elif task == 'Z':
            try:
                assert args.vtk.endswith('vtk')
                assert (args.ref.endswith('nii.gz') or args.ref.endswith('nii'))
                assert (args.image.endswith('nii.gz') or args.image.endswith('nii'))
                assert args.mat is not None
            except:
                print('Also check para\n')
                Help(help_info)
                sys.exit(1)
            task_dict[task](args.vtk, args.output, args.image, args.ref, args.mat)
            #vtk_apply_flirt(input_surf, output, move_img, ref_image, flirt_mat)

        elif task == 'A':
            try:
                assert args.vtk.endswith('vtk')
                assert (args.ref.endswith('nii.gz') or args.ref.endswith('nii'))
                assert (args.image.endswith('nii.gz') or args.image.endswith('nii'))
                assert args.mat is not None
            except:
                print('Also check para\n')
                Help(help_info)
                sys.exit(1)
            task_dict[task](args.vtk, args.output, args.image, args.ref, args.mat)

        elif task == 'F':
            try:
                assert args.vtk.endswith('vtk')
                assert (args.ref.endswith('nii.gz') or args.ref.endswith('nii'))
                assert (args.image.endswith('nii.gz') or args.image.endswith('nii'))
                assert args.mat is not None
            except:
                print('Also check para\n')
                Help(help_info)
                sys.exit(1)
            task_dict[task](args.vtk, args.output, args.image, args.ref, args.mat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Power_view 11 Functions in one, more under development !')
    parser.add_argument('-m', '--mod', type=str,
                        default='H', help='Nii to vtk in RAS (R),  '\
                 'Nii to vtk in native Volume xyz (V),  '\
                 'Mgz to vtk in RAS (M),  '\
                 'Mgz to vtk in native Volume xyz (G),  '\
                 'Surface to reference Mgz RAS space (S),  '\
                 'Surface to reference Mgz in native Volume xyz space (C),  '\
                 'Dsi_studio Fiber return to fa (D),  '\
                 'Surface to reference Mgz RAS space but lines in VTK file (L),  '\
                 'HELP (H),  '\
                 'Reverse filrt from moved Image and back to vtk (Z), '\
                 'Apply filrt Mat to vtk (A), ' \
                 'Apply flirt Mat to Fiber-Line (F), '\
                 'Write value from voxel to vtk (W), '\
                 'Write value onto vtk by index-value pair (I)'
                 )

    parser.add_argument('-i', '--image', type=str, default=None, help='input file: nifti/mgz')
    parser.add_argument('-s', '--surf', type=str, default=None, help='input file: freesurf surf file, .white or .pial')
    parser.add_argument('-k', '--trk', type=str, default=None, help='input file: fiber trk file, .trk')
    #parser.add_argument('-n', '--nii', type=str, default=None, help='input file: nifti')
    #parser.add_argument('-m', '--mgz', type=str, default=None, help='freesurfer T1 mgz file')
    parser.add_argument('-v', '--vtk', type=str, default=None, help='input file: vtk')
    parser.add_argument('-o', '--output', type=str, default='./output.vtk', help='out put')
    # parser.add_argument('-v', '--vtk', type=str,
    #                     default='/mnt/disk2/work_2022_5/Rotate_HCP/subs/141422/test/T1_bet_fa.mhd', help='mhd_t_dict')
    parser.add_argument('-r', '--ref', type=str, default=None, help='reference mgz/nifti')
    parser.add_argument('-a', '--mat', type=str, default=None, help='Affine matrix')

    parser.add_argument('-c', '--color', type=int, default=2, help='index of color, Color in RGB as 0 1 2')
    parser.add_argument('-d', '--downsample', type=bool, default=0, help='if do downsample')

    parser.add_argument('-x', '--xvalue', type=str, default=None, help='array matrix (n*4)/(n*2), 3D cooridinate and certain value/index value pair')

    #parser.add_argument('-h', '--help', type=bool, default=False, help='HELP')

    #args = parser.parse_args()

    main(parser)
    
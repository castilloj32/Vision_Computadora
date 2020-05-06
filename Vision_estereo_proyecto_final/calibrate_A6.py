#!/usr/bin/env python


from __future__ import print_function
import numpy as np
import cv2
import os
import sys
import getopt
from glob import glob
from common import splitfn

############################################################################################
############################################################################################

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

######Deteccion puntos camara 1##################################################################
############################################################################################

img_mask = './left*.ppm'  # default
img_names = glob(img_mask)

debug_dir = './output/'
if not os.path.isdir(debug_dir):
    os.mkdir(debug_dir)
square_size = float(1.0)

pattern_size = (9, 6)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []
h, w = 0, 0
img_names_undistort = []
for fn in img_names:
    print('processing %s... ' % fn, end='')
    img = cv2.imread(fn, 0)
    if img is None:
        print("Failed to load", fn)
        continue
    
    h, w = img.shape[:2]
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        img=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(img, pattern_size, corners, found)
        path, name, ext = splitfn(fn)
        outfile = debug_dir + 'points' + name + '.jpg'
        cv2.imwrite(outfile, img)
        if found:
            img_names_undistort.append(outfile)
            

    
    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)

#####Deteccion puntos camara 2###################################################################
############################################################################################
    
img_mask2 = './right*.ppm'  # default
img_names2 = glob(img_mask2)
debug_dir2 = './output/'
if not os.path.isdir(debug_dir2):
    os.mkdir(debug_dir2)
square_size2 = float(1.0)

pattern_size2 = (9, 6)
pattern_points2 = np.zeros((np.prod(pattern_size2), 3), np.float32)
pattern_points2[:, :2] = np.indices(pattern_size2).T.reshape(-1, 2)
pattern_points2 *= square_size2

obj_points2 = []
img_points2 = []
h2, w2 = 0, 0
img_names_undistort2 = []
for fn2 in img_names2:
    print('processing %s... ' % fn2, end='')
    img2 = cv2.imread(fn2, 0)
    if img2 is None:
        print("Failed to load", fn2)
        continue

    h2, w2 = img.shape[:2]
    found2, corners2 = cv2.findChessboardCorners(img2, pattern_size2)
    if found2:
        term2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img2, corners2, (5, 5), (-1, -1), term2)
        img2=cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(img2, pattern_size, corners2, found)
        path, name, ext = splitfn(fn2)
        outfile = debug_dir + 'points' + name + '.jpg'
        cv2.imwrite(outfile, img2)
        if found:
            img_names_undistort2.append(outfile)
    
    img_points2.append(corners2.reshape(-1, 2))
    obj_points2.append(pattern_points2)



############################################################################################
############################################################################################

#Se obtiene la matriz de cada camara (parametros intrinsecos)

camera_matrix1=cv2.initCameraMatrix2D(obj_points,img_points,(w,h))
camera_matrix2=cv2.initCameraMatrix2D(obj_points2,img_points2,(w2,h2))
term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)


#Se obtienen Parametros extrinsecos R y T de cada camara
flags = 0
#flags |= cv2.CALIB_FIX_INTRINSIC
#flags |= cv2.CALIB_USE_INTRINSIC_GUESS
#flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
#flags |= cv2.CALIB_FIX_FOCAL_LENGTH
flags |= cv2.CALIB_FIX_ASPECT_RATIO
flags |= cv2.CALIB_ZERO_TANGENT_DIST
flags |= cv2.CALIB_SAME_FOCAL_LENGTH
flags |= cv2.CALIB_RATIONAL_MODEL
flags |= cv2.CALIB_FIX_K3
flags |= cv2.CALIB_FIX_K4
flags |= cv2.CALIB_FIX_K5

dist_coeffs1=0
dist_coeffs2=0
(rms,camera_matrix1,dist_coeffs1,camera_matrix2,dist_coeffs2,R,T,E,F)=cv2.stereoCalibrate(obj_points,img_points,img_points2,camera_matrix1,dist_coeffs1,camera_matrix2,dist_coeffs2,(w,h),criteria=term_crit,flags=flags)


#Se obtiene Mapas de rectificacion R1,R2,P1,P2 y matriz de reproyeccion Q.

R1, R2, P1, P2, Q, roi1, roi2=cv2.stereoRectify(camera_matrix1,dist_coeffs1,camera_matrix2,dist_coeffs2,(w,h),R,T,alpha=0)


map11,map12=cv2.initUndistortRectifyMap(camera_matrix1,dist_coeffs1,R1,P1,(w,h),cv2.CV_16SC2)
map21,map22=cv2.initUndistortRectifyMap(camera_matrix2,dist_coeffs2,R2,P2,(w,h),cv2.CV_16SC2)


#########################Se aplica transformacion geometria a cada imagen###################
#########################Tomando los mapas de rectificacion##################################


print('loading images...')


num=0
while num <= (len(obj_points)):
    
    if num == len(obj_points):
        imgL=cv2.imread('imagen_izq.ppm')
        imgR=cv2.imread('imagen_der.ppm')
    else:    
        imgL=cv2.imread(img_names_undistort[num])
        imgR=cv2.imread(img_names_undistort2[num])
    
    rima1=cv2.remap(imgL,map11,map12,cv2.INTER_LINEAR)
    rima2=cv2.remap(imgR,map21,map22,cv2.INTER_LINEAR)
    concat = np.concatenate((rima1, rima2), axis=1)
    height,width=concat.shape[:2]
    count = 0
    while count < width:
    
        concat=cv2.line(concat,(0,count),(width,count),(0,255,0),1,8)
        count +=16
        
    concat2=cv2.resize(concat,(int(width*.7),int(height*.7)))
    cv2.imwrite(debug_dir2 + str(num) +'concat.jpg', concat)
    
    # Se indica el algoritmo a utilizar para calcular correspondencia estereo
    #SGBM
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )
    
    #Se hace el calculo de correspondencia en estereo y mapa de
    #disparidad
    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    
    print('generating 3d point cloud...',)
    h3, w3 = imgL.shape[:2]
    f = 0.8*w3                     
    
    #Se reproyecta mapa disparidad a 3D y se obtiene mapa de
    #profundidad.
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    
    write_ply(debug_dir + (str(num) + 'out.ply'), out_points, out_colors)
    print('%s saved' % 'out.ply')
    
    num+=1
    cv2.destroyAllWindows()
############################################################################################
############################################################################################

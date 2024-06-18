#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## BIOCV CALIBRATION TO OPENCAP CALIBRATION     ##
    ##################################################
    
    Convert bioCV calibration files (without extension) to 
    to OpenCap .pickle calibration files

    Usage: 
    - Put your bioCV calibration files in a folder calib_dir,
    - Run the script: python -m bioCV_to_opencap -i calib_dir
'''

import os
import pickle
import argparse
import numpy as np
import cv2


def biocv_to_opencap_func(*args):
    '''
    Put your bioCV calibration files in a folder calib_dir, 
    Run the script: python -m bioCV_to_opencap -i calib_dir
    '''

    # paths to bioCV calibration files
    calib_dir = args[0]['input_calib_folder']
    list_dir = os.listdir(calib_dir)
    list_dir_noext = [os.path.splitext(f)[0] for f in list_dir if os.path.splitext(f)[1]==''] # files with no extension
    file_to_convert_paths = [os.path.join(calib_dir,f) for f in list_dir_noext if os.path.isfile(os.path.join(calib_dir, f))]

    # read bioCV calib files
    ret, C, S, D, K, R, T = calib_biocv_fun(file_to_convert_paths)

    # write opencap calib files
    write_opencap_pickle(calib_dir, C, S, D, K, R, T)


def calib_biocv_fun(file_to_convert_paths):
    '''
    Convert bioCV calibration files.

    INPUTS:
    - file_to_convert_paths: path of the calibration files to convert (no extension)
    - binning_factor: always 1 with biocv calibration

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''
    

    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    for i, f_path in enumerate(file_to_convert_paths):
        with open(f_path) as f:
            calib_data = f.read().split('\n')
            ret += [np.nan]
            C += [f'cam_{str(i+1).zfill(2)}']
            S += [[int(calib_data[0]), int(calib_data[1])]]
            D += [[float(d) for d in calib_data[-2].split(' ')[:4]]]
            K += [np.array([k.strip().split(' ') for k in calib_data[2:5]], np.float32)]
            RT = np.array([k.strip().split(' ') for k in calib_data[6:9]], np.float32)
            R += [cv2.Rodrigues(RT[:,:3])[0].squeeze()]
            T += [RT[:,3]/1000]
                        
    return ret, C, S, D, K, R, T


def write_opencap_pickle(output_calibration_folder, C, S, D, K, R, T):
    '''
    Writes OpenCap .pickle calibration files

    Extrinsics in OpenCap are calculated with a vertical board for the world frame.
    As we want the world frame to be horizontal, we need to rotate cameras by -Pi/2 around x in the world frame. 
    T is good the way it is.

    INPUTS:
    - Path of the output calibration folder
    - C: list of camera names
    - S: list of image sizes
    - D: list of distortion coefficients
    - K: list of intrinsic parameters
    - R (extrinsic rotation),
    - T (extrinsic translation)
    '''
    
    for i in range(len(C)):
        # Transform rotation for vertical frame of reference (checkerboard vertical with OpenCap)
        R_mat = cv2.Rodrigues(R[i])[0] # transform in matrix
        R_w, T_w = RT_qca2cv(R_mat, T[i]) # transform in world centered perspective
        R_w_90, T_w_90 = rotate_cam(R_w, T_w, ang_x=-np.pi/2, ang_y=0, ang_z=np.pi) # rotate cam wrt world frame
        R_c, T_c = RT_qca2cv(R_w_90, T_w_90) # transform in camera centered perspective

        # retrieve data
        calib_data = {'distortion': np.append(D[i],np.array([0])),
                      'intrinsicMat': K[i],
                      'imageSize': np.expand_dims(S[i][::-1], axis=1)
                      }

        # write pickle
        with open(os.path.join(output_calibration_folder, f'cam{i:02d}.pickle'), 'wb') as f_out:
            pickle.dump(calib_data, f_out)
            
            
def RT_qca2cv(r, t):
    '''
    Converts rotation R and translation T 
    from Qualisys object centered perspective
    to OpenCV camera centered perspective
    and inversely.

    Qc = RQ+T --> Q = R-1.Qc - R-1.T
    '''

    r = r.T
    t = - r.dot(t) 

    return r, t


def rotate_cam(r, t, ang_x=0, ang_y=0, ang_z=0):
    '''
    Apply rotations around x, y, z in cameras coordinates
    Angle in radians
    '''

    r,t = np.array(r), np.array(t)
    if r.shape == (3,3):
        rt_h = np.block([[r,t.reshape(3,1)], [np.zeros(3), 1 ]]) 
    elif r.shape == (3,):
        rt_h = np.block([[cv2.Rodrigues(r)[0],t.reshape(3,1)], [np.zeros(3), 1 ]])
    
    r_ax_x = np.array([1,0,0, 0,np.cos(ang_x),-np.sin(ang_x), 0,np.sin(ang_x),np.cos(ang_x)]).reshape(3,3) 
    r_ax_y = np.array([np.cos(ang_y),0,np.sin(ang_y), 0,1,0, -np.sin(ang_y),0,np.cos(ang_y)]).reshape(3,3)
    r_ax_z = np.array([np.cos(ang_z),-np.sin(ang_z),0, np.sin(ang_z),np.cos(ang_z),0, 0,0,1]).reshape(3,3) 
    r_ax = r_ax_z.dot(r_ax_y).dot(r_ax_x)

    r_ax_h = np.block([[r_ax,np.zeros(3).reshape(3,1)], [np.zeros(3), 1]])
    r_ax_h__rt_h = r_ax_h.dot(rt_h)
    
    r = r_ax_h__rt_h[:3,:3]
    t = r_ax_h__rt_h[:3,3]

    return r, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_calib_folder', required = True, help='Directory of the bioCV calibration files')
    args = vars(parser.parse_args())
    
    biocv_to_opencap_func(args)
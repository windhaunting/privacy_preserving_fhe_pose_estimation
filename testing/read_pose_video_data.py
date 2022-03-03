#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:15:59 2021

@author: fubao
"""
import numpy as np
import pickle

# read pose estimation from this ground truth data

data_dir = "/var/fubao/video_analytics/input_output/one_person_diy_video_dataset/"

video_pose_gt_list = ["output_005_dance/"]

video_frame_list = ["005_dance_frames/"]


# resolution from ground truth
'''
'320x240'	0
'480x352'	1
'640x480'	2
'960x720'	3
'1120x832'	4
'''
# previous project gt coco format without neck  17 points
'''
0	nose
1	leftEye
2	rightEye
3	leftEar
4	rightEar
5	leftShoulder
6	rightShoulder
7	leftElbow
8	rightElbow
9	leftWrist
10	rightWrist
11	leftHip
12	rightHip
13	leftKnee
14	rightKnee
15	leftAnkle
16	rightAnkle
'''

        
def read_pickle_data(pickle_file):
    with open(pickle_file, "rb") as fp:   # Unpickling
        out = pickle.load(fp)
    
    return out      

def get_one_video_pose_gt(video_pose_gt_path):
    # get the one video ground truth pose keypoints
    confg_est_frm_arr = np.load(video_pose_gt_path, allow_pickle=True)

    print("confg_est_frm_arr: ", confg_est_frm_arr.shape, type(confg_est_frm_arr[0][0]))
    
    return confg_est_frm_arr


if __name__ == '__main__':
    pass
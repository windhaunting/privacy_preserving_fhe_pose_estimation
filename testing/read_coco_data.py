#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:33:37 2021

@author: fubao
"""
import sys
import math
import json
import numpy as np
import argparse

# read the coco dataset validation annotation dataset
# read the image path and ground truth
# the file is a json file generated from generate_json_mask.py


"""

 For COCO, part 1 is the neck position calculated by the mean of the two shoulders. 
 The part part 18 in COCO are background predictions.
 {{0, "Nose"}, //t
{1, "Neck"}, //f is not included in coco.  calculated by the mean of the two shoulders.
{2, "RShoulder"}, //t
{3, "RElbow"}, //t
{4, "RWrist"}, //t
{5, "LShoulder"}, //t
{6, "LElbow"}, //t
{7, "LWrist"}, //t
{8, "RHip"}, //t
{9, "RKnee"}, //t
{10, "RAnkle"}, //t
{11, "LHip"}, //t
{12, "LKnee"}, //t
{13, "LAnkle"}, //t
{14, "REye"}, //t
{15, "LEye"}, //t
{16, "REar"}, //t
{17, "LEar"}, //t
{18, "Bkg"}}, //f background ??
 """
 
def parse():
    """
    json_path(.json) is the save_path for the generated json file, which contains the information required for training.
    json file format:
        [{"filename": "/var/fubao/TSM_video_analytics/TSM_Pose_Estimation/input_data/coco/images/val2017/000000397133.jpg", "info": [{"pos": [443.365
, 208.73000000000002], "keypoints": [[433, 94, 1], [447.0, 130.5, 1], [474, 133, 1], [489, 173, 1], [0, 0, 2], [420, 128, 1], [396, 162, 1], 
[0, 0, 2], [458, 215, 1], [458, 273, 1], [465, 334, 1], [419, 214, 1], [411, 274, 1], [402, 333, 1], [0, 0, 2], [434, 90, 1], [0, 0, 2], [443
, 98, 1]], "scale": 0.7544021739130435}]}, {"filename": "/var/fubao/TSM_video_analytics/TSM_Pose_Estimation/input_data/coco/images/val2017/00
0000252219.jpg", "info": [{"pos": [361.9, 273.185], "keypoints": [[356, 198, 1], [358.0, 209.0, 1], [341, 211, 1], [336, 238, 1], [343, 242, 
1], [375, 207, 1], [388, 236, 1], [392, 263, 1], [347, 272, 1], [348, 318, 1], [355, 354, 1], [373, 271, 1], [372, 316, 1], [372, 353, 1], [3
51, 194, 1], [358, 193, 1], [346, 194, 1], [364, 192, 1]], "scale": 0.5360054347826086}, {"pos": [70.75999999999999, 280.28499999999997], "ke
ypoints": [[100, 190, 1], [77.5, 208.0, 1], [71, 208, 1], [59, 240, 1], [66, 271, 1], [84, 208, 1], [84, 245, 1], [115, 263, 1], [71, 264, 1]
, [99, 322, 1], [101, 377, 1], [64, 268, 1], [59, 324, 1], [18, 363, 1], [96, 185, 1], [0, 0, 2], [86, 188, 1], [0, 0, 2]], "scale": 0.615353
2608695652}, {"pos": [572.27, 279.15], "keypoints": [[536, 192, 0], [561.5, 207.5, 1], [555, 208, 1], [554, 246, 1], [550, 277, 1], [568, 207
, 1], [559, 243, 1], [542, 270, 1], [559, 274, 1], [541, 322, 1], [530, 361, 1], [573, 274, 1], [589, 323, 1], [617, 365, 1], [0, 0, 2], [538
, 188, 1], [0, 0, 2], [552, 190, 1]], "scale": 0.586304347826087}]}, 
                                                                                                                                          
    """
    
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--input_data_json_path', type=str,
                        dest='input_data_json_path', help='the save_path for the generated json file')

    return parser.parse_args()

def read_json(args):
    json_file_path = args.input_data_json_path
    json_list = json.load(json_file_path)
    
    print("json_list: ", json_list)

if __name__ == '__main__':

    args = parse()
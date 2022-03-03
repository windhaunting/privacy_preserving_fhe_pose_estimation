
import os
import glob
import time
import cv2
import unittest
import torch.nn as nn
from Pyfhel.Pyfhel import Pyfhel
import numpy as np

import nufhe


import sys
sys.path.append('..')


from read_pose_video_data import data_dir, video_frame_list


# test on the GPU nufhe library  for FHE

INPUT_FRM_WIDTH = 128       # 640  input imgage width 
INPUT_FRM_HEIGHT = 128       #320  input imgage height 


# test the encoded layer
class TestEncodedLayersGPU(object):
    def __init__(self):
        self.ctx = nufhe.Context()
        self.secret_key, self.cloud_key = self.ctx.make_key_pair()

    def encrypt_one_image(self, img_arr):
        encrypted_image = self.ctx.encrypt(self.secret_key, self.secret_key)
        return encrypted_image
        
    def encode_one_image_convolutionalLayer(self,frm_path):
        pass
        
        
    def test_multiple_frames(self, video_frm_parent_dir):
        #  detect a continous sequence of frames (video clips)
        # then calculate the accuracy and processing time per second/frame
        #  confg_est_frm_arr is the video's pose estimation array
        frame_path_list = sorted(glob.glob(video_frm_parent_dir + "*.jpg"))
        
        out_parent_dir = "/".join(video_frm_parent_dir.split('/')[:-2])
        video_name_dir = video_frm_parent_dir.split('/')[-2] + "_no_temporal_frame_out"
        
        output_video_dir = out_parent_dir + "/" + video_name_dir + "/"
        
        print("output_video_output: ", frame_path_list[0], output_video_dir)
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir)
        
        
        elpased_time = 0
        processed_frame_path_list = frame_path_list[:1]
        for i, frm_path in enumerate(processed_frame_path_list):
            # generate image with body parts
            origin_img = cv2.imread(frm_path)
            origin_img = cv2.resize(origin_img, (INPUT_FRM_WIDTH, INPUT_FRM_HEIGHT))  # resize for fast test  # '320x240'	0; '480x352'	1 ; '640x480'	2; '960x720'	3; '1120x832'	4
            #normed_img = self.normalize(origin_img)
            
            time_start = time.time()
            self.encrypt_one_image(origin_img)
            
            elpased_time += time.time() - time_start
        
        print("processed time {} second per frame: ".format(elpased_time/len(processed_frame_path_list)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
 
    obj_testEncodedLayers = TestEncodedLayersGPU()

    video_indx = 0     # to be detected video
    
    # get the pose estimation result
    video_frm_parent_dir = data_dir + video_frame_list[video_indx]
    obj_testEncodedLayers.test_multiple_frames(video_frm_parent_dir)
    
   
    

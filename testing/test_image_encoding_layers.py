
import os
import glob
import time
import cv2
import unittest
import torch.nn as nn
from Pyfhel.Pyfhel import Pyfhel
import numpy as np

#from Pyfhel import PyPtxt

import sys
sys.path.append('..')

from PyFHE_Encode.crypto.crypto import decode_matrix, decode_matrix, decode_matrix
from PyFHE_Encode.functional.convolutional_layer import ConvolutionalLayer
from PyFHE_Encode.functional.flatten_layer import FlattenLayer
from PyFHE_Encode.functional.rencryption_layer import RencryptionLayer
from PyFHE_Encode.functional.ReLU import ReLU
from PyFHE_Encode.functional.batchNormalization import BatchNormalization
from PyFHE_Encode.encoded_net_builder import build_from_pytorch
from PyFHE_Encode.crypto import crypto as cr

from read_pose_video_data import data_dir, video_frame_list

INPUT_FRM_WIDTH = 32       # 640  input imgage width 
INPUT_FRM_HEIGHT = 32       #320  input imgage height 


# test the encoded layer
class TestEncodedLayers(unittest.TestCase):
    def __init__(self):
        self.HE = Pyfhel()
        self.HE.contextGen(p=65537, m=2048, base=3, flagBatching=True) # (65537)    # (p=65537, m=2048, base=3, flagBatching=True)
        self.HE.keyGen()
        self.HE.relinKeyGen(20, 100)
    
    def encrypt_one_image(self, img_arr):
        encrypted_image = cr.encrypt_matrix(self.HE, img_arr)
        
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
            print("origin_img info: ", origin_img.shape)
            time_start = time.time()
            encrypted_image = self.encrypt_one_image(origin_img)
            print("encrypted_image info: ", type(encrypted_image), encrypted_image.shape, encrypted_image.dtype)
            
            time_start = time.time()
        
            
            """
            # convolutional layer encoding
            plain_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            )
            encoded_net = build_from_pytorch(self.HE, plain_net)
            print("encoded_net: ", encoded_net)
            self.assertTrue(np.allclose(plain_net[0].weight.detach().numpy(),
                        decode_matrix(self.HE, encoded_net[0].weights)))
            #self.assertEqual(plain_net[0].kernel_size, encoded_net[0].kernel_size)
            
            # convolutional layer encoding
            plain_net = nn.Sequential(
            #nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            encoded_net = build_from_pytorch(self.HE, plain_net)
            print("encoded_net: ", encoded_net, plain_net[0].kernel_size, encoded_net[0].kernel_size[0])
            #self.assertEqual(plain_net[0].kernel_size, encoded_net[0].kernel_size)
            """
            """
            # test relu
            plain_net = nn.Sequential(
            nn.ReLU()
            )
            encoded_net = build_from_pytorch(self.HE, plain_net)
            print("encoded_net: ", encoded_net[0], plain_net[0])
            func_vec = np.vectorize(encoded_net[0])
            relu_out = func_vec(encrypted_image)
            """
            
            # test batch normalizaiton
            
            plain_net = nn.Sequential(
            nn.BatchNorm2d(32)
            )
            encoded_net = build_from_pytorch(self.HE, plain_net)
            print("encoded_net: ", encoded_net[0], plain_net[0])
            batch_out = encoded_net[0](encrypted_image)
            decry_out = cr.decrypt_matrix(self.HE, batch_out)
            #print("decry_out: ", decry_out)
            
            
            """
            print("batch origin_img: ", batchnorm_forward(origin_img))
            """
            """
            x = 4.0
            ptxt_f1 = self.HE.encryptFrac(x)    # need to call encryption
            print("type ptxt_f1 :", type(ptxt_f1))
            encry_out = encoded_net[0](ptxt_f1)
            decry_out = self.HE.decryptFrac(encry_out) 
            print("encr_out: ", x, encry_out, decry_out)
            """
            
            
            elpased_time += time.time() - time_start
        
        print("processed time {} second per frame: ".format(elpased_time/len(processed_frame_path_list)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
 
    obj_testEncodedLayers = TestEncodedLayers()

    video_indx = 0     # to be detected video
    
    # get the pose estimation result
    video_frm_parent_dir = data_dir + video_frame_list[video_indx]
    obj_testEncodedLayers.test_multiple_frames(video_frm_parent_dir)
    
   
    

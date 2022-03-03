#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:21:05 2021

@author: fubao
"""

# test the encoded pose estimation model

# https://github.com/AlexMV12/PyCrCNN/blob/b8b08afc96a6f199f5fc8612eacb95ad98731298/pycrcnn/tests/test_encoded_net_builder.py


import os
import glob
import time
import math
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
import unittest
import gc
from threading import Thread
from queue import Queue

from common_utils import memory_monitor

from Pyfhel.Pyfhel import Pyfhel
from scipy.ndimage.filters import gaussian_filter

from common_utils import computeOKSAP, computeOKS_mat
from read_pose_video_data import data_dir, video_pose_gt_list, video_frame_list


#from Pyfhel import PyPtxt

import sys
sys.path.append('..')

from PyFHE_Encode_ckks.crypto.crypto import decode_matrix
from PyFHE_Encode_ckks.encoded_net_builder import build_from_pytorch
from PyFHE_Encode_ckks.crypto import crypto as cr

from read_pose_video_data import get_one_video_pose_gt
from read_pose_video_data import data_dir, video_frame_list

import pose_estimation_up_to_model22

# detect a video human pose with FHE
#  https://pyfhel.readthedocs.io/en/latest/_autosummary/Pyfhel.Pyfhel.html

limbSeq = [[3,4], [4,5], [6,7], [7,8], [9,10], [10,11], [12,13], [13,14], [1,2], [2,9], [2,12], [2,3], [2,6], \
           [3,17],[6,18],[1,16],[1,15],[16,18],[15,17]]

mapIdx = [[19,20],[21,22],[23,24],[25,26],[27,28],[29,30],[31,32],[33,34],[35,36],[37,38],[39,40], \
          [41,42],[43,44],[45,46],[47,48],[49,50],[51,52],[53,54],[55,56]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    
boxsize = 368
scale_search = [1.0]  # [0.5, 1.0, 1.5, 2.0]
stride = 8
padValue = 0.
thre_point = 0.15
thre_line = 0.05
stickwidth = 1


TOTAL_KEY_PTS = 17     # without the neck
INPUT_FRM_WIDTH = 128   # 640       # input imgage width 
INPUT_FRM_HEIGHT = 128  # 320       # input imgage height 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NetBuilderTester(unittest.TestCase):

    def test_encoded_net_builder(self):

        plain_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=4 * 4 * 8, out_features=64),
            nn.Conv2d(in_channels=1, out_channels=1, bias=False, kernel_size=5),
            nn.Conv2d(in_channels=1, out_channels=1, bias=False, kernel_size=5, stride=2)
        )

        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()

        encoded_net = build_from_pytorch(HE, plain_net)  # , [2])
        
        self.assertTrue(np.allclose(plain_net[0].weight.detach().numpy(),
                        decode_matrix(HE, encoded_net[0].weights)))
 
    
    
    
# test the encoded pose estimation inference with Fully Homomorphic Encryption
class TestEncodedPoseEstimation(unittest.TestCase):
    def __init__(self):
        self.HE = Pyfhel()
        self.HE.contextGen(p=65537, m=2048, base=3, flagBatching=True) # (65537)    # (p=65537, m=2048, base=3, flagBatching=True)
        self.HE.keyGen()
        
        self.HE.relinKeyGen(20, 100) # self.HE.relinKeyGen(20, 100)

    def encrypt_one_image(self, img_arr):
        encrypted_image = cr.encrypt_matrix(self.HE, img_arr)
        
        return encrypted_image
        
        
    def construct_model(self,args):
    
        model = pose_estimation_up_to_model22.PoseModel(num_point=19, num_vector=19)
        state_dict = torch.load(args.model, map_location=device).state_dict()  # .state_dict() # ['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = model.state_dict()
        state_dict.update(new_state_dict)
        model.load_state_dict(state_dict)
        
        model.to(device)     # use GPU
        model.eval()
    
        return model
    
    def normalize(self, origin_img):
    
        origin_img = np.array(origin_img, dtype=np.float32)
        origin_img -= 128.0 # origin_img -= np.mean(origin_img)  # origin_img -= 128.0
        origin_img /= 256.0 # origin_img /= np.std(origin_img)   # origin_img /= 256.0
    
        return origin_img


    def padRightDownCorner(self,img, stride, padValue):
    
        h = img.shape[0]
        w = img.shape[1]
    
        pad = 4 * [None]
        pad[0] = 0 # up
        pad[1] = 0 # left
        pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
        pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right
    
        img_padded = img
        pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
        img_padded = np.concatenate((pad_up, img_padded), axis=0)
        pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
        img_padded = np.concatenate((pad_left, img_padded), axis=1)
        pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
        img_padded = np.concatenate((img_padded, pad_down), axis=0)
        pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
        img_padded = np.concatenate((img_padded, pad_right), axis=1)
    
        return img_padded, pad


    def apply_multi_layers_from_pose_estimation_model(self, model):
        
        model0_sequential = model.model0      #length: 27
        model11_sequential = model.model1_1   #         9
        model12_sequential = model.model1_2   #         9
        model21_sequential = model.model2_1   #         13
        model22_sequential = model.model2_2   #         13
        
        """
        
        model31_sequential = model.model3_1   #         13
        model32_sequential = model.model3_2   #         13
        model41_sequential = model.model4_1   #         13
        model42_sequential = model.model4_2   #         13
        model51_sequential = model.model5_1   #         13
        model52_sequential = model.model5_2   #         13
        model61_sequential = model.model6_1   #         13
        model62_sequential = model.model6_2   #         13
        
        
        all_nn_sequntial = torch.nn.Sequential(*(list(model0_sequential)  +
                                                 list(model11_sequential) +list(model12_sequential) + 
                                                 list(model21_sequential) +list(model22_sequential) + 
                                                 list(model31_sequential) +list(model32_sequential) + 
                                                 list(model41_sequential) +list(model42_sequential) + 
                                                 list(model51_sequential) +list(model52_sequential) + 
                                                 list(model61_sequential) +list(model62_sequential)
                                                 ))
        
        """
        all_nn_sequntial = torch.nn.Sequential(*(list(model0_sequential)  +
                                                 list(model11_sequential) +list(model12_sequential) + 
                                                 list(model21_sequential) +list(model22_sequential)
                                                 ))
        
        
        encoded_net = build_from_pytorch(self.HE, all_nn_sequntial[0:45])  # 0:45 all_nn_sequntial[0:2])
        
        return encoded_net
    
    def apply_models_to_image(self, encoded_net, encrypted_image, mask):
        """
        Parameters
        ----------
        encoded_net : List of encoded layer
            DESCRIPTION.
        encrypted_image : np.array of encrypted image
            DESCRIPTION.
        mask:  np.array  (not encrypted)
        Returns
        -------
        None.  decrypted heat2 and vec2 which is 

        Note:
        out1_1 = self.model1_1(out0)
        out1_2 = self.model1_2(out0)
        out1 = torch.cat([out1_1, out1_2, out0], 1)
        out1_vec_mask = out1_1 * mask
        out1_heat_mask = out1_2 * mask
        
        out2_1 = self.model2_1(out1)
        out2_2 = self.model2_2(out1)
        out2 = torch.cat([out2_1, out2_2, out0], 1)
        out2_vec_mask = out2_1 * mask
        out2_heat_mask = out2_2 * mask
        """
        print("apply_models_to_image encoded_net type: ", type(encoded_net), encrypted_image.shape, mask.shape)
        
        model0_layer_index = 26
        model11_layers_index = 35   # 35, 44, is the up to model1_1 and model1_2 output
        model12_layers_index = 44
        model21_layers_index = 57    # up to model21  
        model22_layers_index = 70    # up to model22
        
        current_img_out = encrypted_image
        out1_1 = current_img_out
        out1_2 = current_img_out
        
        out2_1 = current_img_out
        out2_2 = current_img_out
        for i, model in enumerate(encoded_net):
            # each model is a layer
            current_img_out = model(current_img_out)
            
            if i == model0_layer_index:
                out0 = current_img_out
            elif i == model11_layers_index:
                out1_1 = current_img_out
            elif i == model12_layers_index:
                out1_2 = current_img_out
                current_img_out = np.concatenate((out1_1, out1_2), axis=1)
            elif i == model21_layers_index:
                out2_1 = current_img_out
            elif i == model22_layers_index:
                out2_2 = current_img_out
                
            print("apply_models_to_image i: ", i)
            
        """
        if out2_1 == current_img_out:
            out2_1 = out1_1
        
        if out2_2 == current_img_out:
            out2_2 = out1_2
        
        out2_1 = cr.decrypt_matrix(self.HE, out2_1)
        out2_2 = cr.decrypt_matrix(self.HE, out2_2)

        out2_vec_mask = out2_1 * mask
        out2_heat_mask = out2_2 * mask
        print("out2_vec_mask_out: ", out2_vec_mask.shape, out2_heat_mask.shape)
            
        return out2_vec_mask, out2_heat_mask
        """
        
        out1_1 = cr.decrypt_matrix(self.HE, out1_1)
        out1_2 = cr.decrypt_matrix(self.HE, out1_2)
        
        out1_vec_mask = out1_1 * mask
        out1_heat_mask = out1_2 * mask
        print("out1_vec_mask_out: ", out1_vec_mask.shape, out1_heat_mask.shape)
        return out1_vec_mask, out1_heat_mask

                                     
    def  process_encoded_one_frame(self, encoded_net, origin_img):
    
        
        # apply the encoded model to this one frame
        normed_img = self.normalize(origin_img)
    
        height, width, _ = normed_img.shape
    
        multiplier = [x * boxsize / height for x in scale_search]
    
        heatmap_avg = np.zeros((height, width, 19)) # num_point
        paf_avg = np.zeros((height, width, 38))     # num_vector
    
        for m in range(len(multiplier)):
            scale = 1.0     # multiplier[m]
    
            # preprocess
            imgToTest = cv2.resize(normed_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imgToTest_padded, pad = self.padRightDownCorner(imgToTest, stride, padValue)
    
            input_img = np.transpose(imgToTest_padded[:,:,:,np.newaxis], (3, 2, 0, 1)) # required shape (1, c, h, w)
            mask = np.ones((1, 1, input_img.shape[2] // stride, input_img.shape[3] // stride), dtype=np.float32)
    
            # encrypt image and no need to encrypt
            input_img = self.encrypt_one_image(input_img)
            
            #mask= self.encrypt_one_image(mask)  # no need to encrypt
            
            print("input_var mask_var shape 1111: ", input_img.shape, mask.shape)

            
            #input_var = torch.autograd.Variable(torch.from_numpy(input_img))  #.to(device)) # .cuda())
            # mask_var = torch.autograd.Variable(torch.from_numpy(mask))  #.to(device)) # .cuda())
            
            input_var = input_img
            mask_var = mask
            
            print("input_var mask_var shape 1111: ", input_var.shape, mask_var.shape)

      
            # need to return vec2 and heat2 
            # this is the model output, simulate sending out to the client and decrypt it  
            # so we need to get the decrypted result
            out2_vec_mask, out2_heat_mask = self.apply_models_to_image(encoded_net, input_img, mask_var)
        

            # get the features
            print("test process vec1, heat1: ", type(out2_vec_mask), type(out2_heat_mask)) # , out2_vec_mask.size())
            # get the heatmap
            heatmap = out2_heat_mask  #  out2_heat_mask.data.cpu().numpy()
            heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0)) # (h, w, c)
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
    
            # get the paf
            paf = out2_vec_mask       # out2_vec_mask.data.cpu().numpy()
            paf = np.transpose(np.squeeze(paf), (1, 2, 0)) # (h, w, c)
            paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            paf = paf[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
            paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_CUBIC)
            paf_avg = paf_avg + paf / len(multiplier)
    
        all_peaks = []   # all of the possible points by classes.
        peak_counter = 0
    
        for part in range(1, 19):
            map_ori = heatmap_avg[:, :, part]
            map = gaussian_filter(map_ori, sigma=3)
    
            map_left = np.zeros(map.shape)
            map_left[:, 1:] = map[:, :-1]
            map_right = np.zeros(map.shape)
            map_right[:, :-1] = map[:, 1:]
            map_up = np.zeros(map.shape)
            map_up[1:, :] = map[:-1, :]
            map_down = np.zeros(map.shape)
            map_down[:-1, :] = map[1:, :]
    
            # get the salient point and its score > thre_point
            peaks_binary = np.logical_and.reduce(
                    (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre_point))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # (w, h)
            
            # a point format: (w, h, score, number)
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i], ) for i in range(len(id))]
    
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
    
        connection_all = [] # save all of the possible lines by classes.
        special_k = []      # save the lines, which haven't legal points.
        mid_num = 10        # could adjust to accelerate (small) or improve accuracy(large).
    
        for k in range(len(mapIdx)):
    
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
    
            lenA = len(candA)
            lenB = len(candB)
    
            if lenA != 0 and lenB != 0:
                connection_candidate = []
                for i in range(lenA):
                    for j in range(lenB):
                        vec = np.subtract(candB[j][:2], candA[i][:2]) # the vector of BA
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        if norm == 0:
                            continue
                        vec = np.divide(vec, norm)
    
                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), np.linspace(candA[i][1], candB[j][1], num=mid_num)))
    
                        # get the vector between A and B.
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])
    
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * height / norm - 1, 0) # ???
                        criterion1 = len(np.nonzero(score_midpts > thre_line)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                # sort the possible line from large to small order.
                connection_candidate = sorted(connection_candidate, key=lambda x: x[3], reverse=True) # different from openpose, I think there should be sorted by x[3]
                connection = np.zeros((0, 5))
    
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0: 3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        # the number of A point, the number of B point, score, A point, B point
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]]) 
                        if len(connection) >= min(lenA, lenB):
                            break
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])
    
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])
    
        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1
    
                for i in range(len(connection_all[k])):
                    found = 0
                    flag = [False, False]
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        # fix the bug, found == 2 and not joint will lead someone occur more than once.
                        # if more than one, we choose the subset, which has a higher score.
                        if subset[j][indexA] == partAs[i]:
                            if flag[0] == False:
                                flag[0] = found
                                subset_idx[found] = j
                                flag[0] = True
                                found += 1
                            else:
                                ids = subset_idx[flag[0]]
                                if subset[ids][-1] < subset[j][-1]:
                                    subset_idx[flag[0]] = j
                        if subset[j][indexB] == partBs[i]:
                            if flag[1] == False:
                                flag[1] = found
                                subset_idx[found] = j
                                flag[1] = True
                                found += 1
                            else:
                                ids = subset_idx[flag[1]]
                                if subset[ids][-1] < subset[j][-1]:
                                    subset_idx[flag[1]] = j
    
                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2: # if found equals to 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0: # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else: # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        return all_peaks, candidate, subset


    def test_multiple_frames(self, agrs, video_frm_parent_dir, model, confg_est_frm_arr):
        #  detect a continous sequence of frames (video clips)
        # then calculate the accuracy and processing time per second/frame
        #  confg_est_frm_arr is the video's pose estimation array
        frame_path_list = sorted(glob.glob(video_frm_parent_dir + "*.jpg"))
        
        out_parent_dir = "/".join(video_frm_parent_dir.split('/')[:-2])
        video_name_dir = video_frm_parent_dir.split('/')[-2] + "_no_temporal_FHE_encoding_frame_out"
        
        output_video_dir = out_parent_dir + "/" + video_name_dir + "/"
        
        print("output_video_output: ", frame_path_list[0], output_video_dir)
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir)
        
        
        # get the ground truth
        pose_gt_frm_arr = confg_est_frm_arr[1]
        
        
        """
        queue = Queue()
        poll_interval = 0.1
        monitor_thread = Thread(target=memory_monitor, args=(queue, poll_interval))
        monitor_thread.start()
        """
        average_precision = 0.0

        elpased_time = 0
        processed_frame_path_list = frame_path_list[:1]
        for i, frm_path in enumerate(processed_frame_path_list):
            
            
            # generate image with body parts
            origin_img = cv2.imread(frm_path)
            origin_img = cv2.resize(origin_img, (INPUT_FRM_WIDTH, INPUT_FRM_HEIGHT))  # resize for fast test  # '320x240'	0; '480x352'	1 ; '640x480'	2; '960x720'	3; '1120x832'	4
            #normed_img = self.normalize(origin_img)
            time_start = time.time()


            encoded_net = self.apply_multi_layers_from_pose_estimation_model(model)
            
            
            """
            
            print("origin_img info: ", origin_img.shape)
            time_start = time.time()
            encrypted_image = self.encrypt_one_image(origin_img)
            print("encrypted_image info: ", type(encrypted_image), encrypted_image.shape, encrypted_image.dtype)
            
            
            time_start = time.time()
            
            
            model = self.construct_model(agrs)
            # model type: <class 'pose_estimation_up_to_model22.Pose_Estimation'>
            print("model:", type(model.model0))
            
            print("model.model0: ", model.model0)

            
            try:
                #encoded_net = build_from_pytorch(self.HE, model.model0[:1])
                encoded_net = self.apply_multi_layers_from_pose_estimation_model( model)
                print("encoded_net: ", encoded_net[0], model.model0[0])
            finally:
                queue.put('stop')
                monitor_thread.join()
            
            print("encoded_net aaaa: ", encoded_net[0])
            """
           
            all_peaks, candidate, subset = self.process_encoded_one_frame(encoded_net, origin_img)

            
            #for var_name in model.state_dict():
                #print(var_name, "\t", model.state_dict()[var_name])
            #    print(var_name) 
            #    break
            
            
            elpased_time += time.time() - time_start
            
            output_file_path =  output_video_dir + "no_temporal_" + frm_path.split("/")[-1]
            canvas = self.draw_point_into_image(all_peaks, candidate, subset, origin_img)
            dts = self.get_detection_pose_result_to_gt_format(all_peaks, candidate, subset)
            
            gts = pose_gt_frm_arr[i]  # get this frame's ground truth pose 17x3
            gts = self.get_pose_position_back(gts, INPUT_FRM_WIDTH, INPUT_FRM_HEIGHT)
            print("dts, gts shape ", dts.shape, gts.shape)
            AP = computeOKS_mat(gts, dts)   # computeOKSAP(gts, dts)
            print("dts, gts: ", dts, gts,  AP)
            print("frm_path: ", frm_path, output_file_path)
            average_precision += AP
            cv2.imwrite(output_file_path, canvas)
            
        print("processed time {} second per frame: ".format(elpased_time/len(processed_frame_path_list)))


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../output_models/openpose_coco_up_to_model22_iter_70000_up_to_model22.pth', help='path to the weights file')
    
    args = parser.parse_args()
    
    gc.collect()       # memory management

    obj_testEncodedPose = TestEncodedPoseEstimation()

    video_indx = 0     # to be detected video
    
    # get one video ground truth
    video_pose_gt_parent_dir = video_pose_gt_list[video_indx]
    video_pose_gt_path = data_dir + video_pose_gt_parent_dir + "frames_pickle_result/config_estimation_frm_more_dim.pkl"
    confg_est_frm_arr = get_one_video_pose_gt(video_pose_gt_path)

    # get the pose estimation result
    video_frm_parent_dir = data_dir + video_frame_list[video_indx]
    
    # load model
    model = obj_testEncodedPose.construct_model(args)
    
    obj_testEncodedPose.test_multiple_frames(args, video_frm_parent_dir, model, confg_est_frm_arr)
    
   
    
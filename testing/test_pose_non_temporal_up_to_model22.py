import argparse
import os
import math
import time
import numpy as np
import torch
import cv2
import glob

from scipy.ndimage.filters import gaussian_filter
from read_pose_video_data import get_one_video_pose_gt
from read_pose_video_data import data_dir, video_pose_gt_list, video_frame_list

from common_utils import computeOKSAP, computeOKS_mat

from preprocess_object_detection import preprocess_img


import sys
sys.path.append('..')
import pose_estimation_up_to_model22

# detect a video human pose without temporal shift module

limbSeq = [[3,4], [4,5], [6,7], [7,8], [9,10], [10,11], [12,13], [13,14], [1,2], [2,9], [2,12], [2,3], [2,6], \
           [3,17],[6,18],[1,16],[1,15],[16,18],[15,17]]

mapIdx = [[19,20],[21,22],[23,24],[25,26],[27,28],[29,30],[31,32],[33,34],[35,36],[37,38],[39,40], \
          [41,42],[43,44],[45,46],[47,48],[49,50],[51,52],[53,54],[55,56]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

boxsize = 368
scale_search = [1.0] # [0.5, 1.0, 1.5, 2.0]
stride = 8
padValue = 0.
thre_point = 0.15
thre_line = 0.05
stickwidth = 1


TOTAL_KEY_PTS = 17     # without the neck
INPUT_FRM_WIDTH = 128 # 256   # 640       # input imgage width 
INPUT_FRM_HEIGHT = 128  # 320       # input imgage height 


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if not torch.cuda.is_available() else "cpu")  # use cpu if on gpu server

"""
this model detect 18 points without background
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
 
 
# without considering temporal shift module
class Without_Temporal_Test(object):
    def __init__(self):
        pass
    
    def construct_model(self,args):
    
        model = pose_estimation_up_to_model22.PoseModel(num_point=19, num_vector=19)
        state_dict = torch.load(args.model).state_dict() # ['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = model.state_dict()
        state_dict.update(new_state_dict)
        model.load_state_dict(state_dict)
        
        model.to(device)
        #model = model.cuda()
        model.eval()
    
        return model
    
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
    
    def normalize(self, origin_img):
    
        origin_img = np.array(origin_img, dtype=np.float32)
        origin_img -= np.mean(origin_img)  # origin_img -= 128.0
        origin_img /= np.std(origin_img)   # origin_img /= 256.0
    
    
        return origin_img
            
  
    def process_one_frame(self, model, origin_img):

        normed_img = self.normalize(origin_img)
    
        height, width, _ = normed_img.shape
    
        multiplier = [x * boxsize / height for x in scale_search]
    
        heatmap_avg = np.zeros((height, width, 19)) # num_point
        paf_avg = np.zeros((height, width, 38))     # num_vector
    
        for m in range(len(multiplier)):
            scale = 1.0 # multiplier[m]
    
            # preprocess
            imgToTest = cv2.resize(normed_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imgToTest_padded, pad = self.padRightDownCorner(imgToTest, stride, padValue)
    
            input_img = np.transpose(imgToTest_padded[:,:,:,np.newaxis], (3, 2, 0, 1)) # required shape (1, c, h, w)
            mask = np.ones((1, 1, input_img.shape[2] // stride, input_img.shape[3] // stride), dtype=np.float32)
    
            input_var = torch.autograd.Variable(torch.from_numpy(input_img).to(device)) # .cuda())
            mask_var = torch.autograd.Variable(torch.from_numpy(mask).to(device)) # .cuda())
    
            # get the features
            vec1, heat1, vec2, heat2 = model(input_var, mask_var)
            print("test process vec1, heat1: ", vec1.size())
            # get the heatmap
            heatmap = heat2.data.cpu().numpy()
            heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0)) # (h, w, c)
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
    
            # get the paf
            paf = vec2.data.cpu().numpy()
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
    
        """
        # delete som rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)
        
        print('test all_peaks: ', all_peaks)
        # draw points
        canvas = cv2.imread(input_path)
        for i in range(18):
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    
        # draw lines
        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(limbSeq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    
        return canvas
        """
        
        return all_peaks, candidate, subset
    
    def read_all_videos_pose_gt_npy_files(self):
        
        for video_pose_gt_parent_dir in video_pose_gt_list:
            video_pose_gt_path = data_dir + video_pose_gt_parent_dir + "frames_pickle_result/config_estimation_frm_more_dim.pkl"
            
            get_one_video_pose_gt(video_pose_gt_path)
        
    
    def detect_one_image(self):
        # detect one image from scratch
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        parser = argparse.ArgumentParser()
        parser.add_argument('--image', type=str, required=True, help='input image')
        parser.add_argument('--output', type=str, default='test_result.png', help='output image')
        parser.add_argument('--model', type=str, default='../output_models/openpose_coco_iter_10.pth', help='path to the weights file')
    
        args = parser.parse_args()
        input_image = args.image
        output = args.output
    
    
        # load model
        model = self.construct_model(args)
    
        tic = time.time()
        print('start processing...')
    
        # generate image with body parts
        canvas = self.process(model, input_image)
    
        toc = time.time()
        print ('processing time is %.5f' % (toc - tic))
    
        cv2.imwrite(output, canvas)
    
    def get_detection_pose_result_to_gt_format(self, all_peaks, candidate, subset):
        # here one person from from all_peaks ?
        # read format of coco 18 pts without background
        # transfer to ground truth detected format 17 pts
        """
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
        
        transfer to 
        
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
        
        """
        dict_map = {0:0, 2:6, 3:8, 4:10, 5:5, 6:7, 7:9, 8:12, 9:14, 10:16, 11:11, 12:13, 13:15, 14:2, 15:1, 16:4, 17:3}
        
        dts = np.zeros((TOTAL_KEY_PTS, 3))
        for i in range(TOTAL_KEY_PTS):
            if i == 1:       # skip the neck
                continue
            for j in range(len(all_peaks[i])):
                dts[dict_map[i]] = [all_peaks[i][j][0], all_peaks[i][j][1], all_peaks[i][j][2]]
        #print("dts :", dts, dts.shape)
        return dts
    
        
        
    def draw_point_into_image(self, all_peaks, candidate, subset, canvas):
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)
        
        print('test all_peaks: ', all_peaks, subset)
        # draw points
        #canvas = cv2.imread(input_img_path)
        for i in range(18):
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, all_peaks[i][j][0:2], 1, colors[i], thickness=1)
    
        # draw lines
        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(limbSeq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                print("mx: ",i, n, index, mX, mY)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
                
        return canvas
    
    def get_pose_position_back(self, gts, INPUT_FRM_WIDTH, INPUT_FRM_HEIGHT):
        # no scaling
        for i in range(gts.shape[0]):
            gts[i] = [int(gts[i][0] * INPUT_FRM_WIDTH), int(gts[i][1]*INPUT_FRM_HEIGHT), gts[i][2]]
        return gts
    
    def detect_multiple_frames(self, model, video_frm_parent_dir, confg_est_frm_arr):
        #  detect a continous sequence of frames (video clips)
        # then calculate the accuracy and processing time per second/frame
        #  confg_est_frm_arr is the video's pose estimation array
        frame_path_list = sorted(glob.glob(video_frm_parent_dir + "*.jpg"))
        
        out_parent_dir = "/".join(video_frm_parent_dir.split('/')[:-2])
        video_name_dir = video_frm_parent_dir.split('/')[-2] + "_no_temporal_original_up_to_model22_frame_out"
        
        output_video_dir = out_parent_dir + "/" + video_name_dir + "/"
        
        print("output_video_output: ", frame_path_list[0], output_video_dir)
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir)
        
        
        # get the ground truth
        pose_gt_frm_arr = confg_est_frm_arr[1]
        
        average_precision = 0.0
        time_start = time.time()
        
        processed_frame_path_list = frame_path_list[:2]
        for i, frm_path in enumerate(processed_frame_path_list):
            # generate image with body parts
            origin_img = cv2.imread(frm_path)
            origin_img = cv2.resize(origin_img, (INPUT_FRM_WIDTH, INPUT_FRM_HEIGHT))  # resize for fast test  # '320x240'	0; '480x352'	1 ; '640x480'	2; '960x720'	3; '1120x832'	4
            print("ooorigin_img shape: ", origin_img.shape)
        
            origin_img = preprocess_img(origin_img)
            print("preprocessed object detection origin_img shape: ", origin_img.shape)
            print("resized origin_img shape: ", origin_img.shape)
            
            
            all_peaks, candidate, subset = self.process_one_frame(model, origin_img)
            output_file_path =  output_video_dir + "no_temporal_original_up_to_model22" + frm_path.split("/")[-1]
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
            
        elpased_time = time.time() - time_start
        
        print("processed time {} second per frame: ".format(elpased_time/len(processed_frame_path_list)))
        print("average precision {}  per frame: ".format(average_precision/len(processed_frame_path_list)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../output_models/openpose_coco_up_to_model22_iter_70000_up_to_model22.pth', help='path to the weights file')
    
    args = parser.parse_args()
    

    # get one video ground truth
    video_indx = 0     # to be detected video
    video_pose_gt_parent_dir = video_pose_gt_list[video_indx]
    video_pose_gt_path = data_dir + video_pose_gt_parent_dir + "frames_pickle_result/config_estimation_frm_more_dim.pkl"
    confg_est_frm_arr = get_one_video_pose_gt(video_pose_gt_path)
    
    obj_noTemporalTest_instance = Without_Temporal_Test()
    # load model
    model = obj_noTemporalTest_instance.construct_model(args)

    # get the pose estimation result
    video_frm_parent_dir = data_dir + video_frame_list[video_indx]
    obj_noTemporalTest_instance.detect_multiple_frames(model, video_frm_parent_dir, confg_est_frm_arr)
    
   
    

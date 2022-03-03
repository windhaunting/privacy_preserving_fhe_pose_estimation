#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 17:22:08 2022

@author: fubao
"""

# preprocess 
# detect the human object first on the client to save time for pose estimation application


# use detectron2

import glob
import torch

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)


data_dir = "/var/fubao/video_analytics/input_output/one_person_diy_video_dataset/"

video_pose_gt_list = ["output_005_dance/"]

video_frame_list = ["005_dance_frames/"]

def read_image(img_path):
    
    img_arr = cv2.imread(img_path)
    return img_arr


def object_detection(img_arr):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img_arr)
    
    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    return img_arr, outputs, cfg


def get_object_arr(img_arr, outputs):
    # outputs: <class 'detectron2.structures.boxes.Boxes'>

    output_coordinates = outputs["instances"].pred_boxes.tensor.cpu().numpy()[0]
    tl_x1 = int(output_coordinates[0])
    tl_y1 = int(output_coordinates[1])
    br_x1 = int(output_coordinates[2])
    br_y1 = int(output_coordinates[3])
    out_img = img_arr[tl_y1:br_y1, tl_x1:br_x1]
    #cv2.imwrite(out_file_path, out_img)

    return out_img
    
def visualize(im, outputs, cfg):
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
    
def preprocess_img(img_arr):
    # detect object first human first
    #img_arr = read_image(input_image_path)
    img_arr, outputs, cfg = object_detection(img_arr)
    #visualize(img_arr, outputs, cfg)
    
    out_img = get_object_arr(img_arr, outputs)
    
    return out_img


if __name__ == '__main__':

    video_indx = 0     # videos to be detected

    video_frm_parent_dir = data_dir + video_frame_list[video_indx]

    frame_path_list = sorted(glob.glob(video_frm_parent_dir + "*.jpg"))

    test_image = frame_path_list[0]
    img_arr = read_image(test_image)
    img_arr, outputs, cfg = object_detection(img_arr)
    #visualize(img_arr, outputs, cfg)
    #out_file_path = "test1.jpg"
    get_object_arr(img_arr, outputs)
    
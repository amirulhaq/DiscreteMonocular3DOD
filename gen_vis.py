from torch.nn import parameter
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np
import utils.visualize as vis

from utils.cs_dataset import cs_data
from cfg.settings import get_cfg_defaults
from typing import List

import json

from utils.visualize import visualize_2d, visualize_3d, draw_bev
from utils.evaluation import mkdir, path_leaf
from utils.cs_annotation import CsBbox3d
from utils.box3dImageTransform import (Camera, 
                                  Box3dImageTransform,
                                  CRS_V,
                                  CRS_C,
                                  CRS_S
                                )

def get_camera(json_data):
    camera = Camera(fx=json_data["sensor"]["fx"],
            fy=json_data["sensor"]["fy"],
            u0=json_data["sensor"]["u0"],
            v0=json_data["sensor"]["v0"],
            sensor_T_ISO_8855=json_data["sensor"]["sensor_T_ISO_8855"])
    return camera

def get_cs_targets_and_objects(target_class:List[str], camera, json_data):
    target_objects = []
    cs_data = json_data.get('objects', {})
    for cs_object in cs_data:
        if cs_object['label'] in target_class:
            target_objects.append(cs_object)
    # Sort objects based on the depth location
    sorted_targets = sorted(target_objects, key=lambda item: item['3d']['center'][0], reverse=True)
    sorted_objects = [] # saved in citiscape object format
    for sorting in sorted_targets:
        obj = CsBbox3d()
        obj.fromJsonText(sorting)
        box3d_annotation = Box3dImageTransform(camera=camera)
        box3d_annotation.initialize_box_from_annotation(obj, coordinate_system=CRS_V)
        sorted_objects.append(box3d_annotation)
    return sorted_targets, sorted_objects

def vis_all(cfg, bev=False):
    pred_dir = cfg.EVAL.SAVE_FOLDER
    cs_dir = cfg.DATASETS.ROOT

    top_dir = 'saves/vis/' + pred_dir.split('/')[-1]
    cs_eval_data = cs_data(cs_dir, cfg=cfg, split='val')
    img_list = cs_eval_data.images
    label_list = cs_eval_data.labels_json

    for i in tqdm(range(len(img_list))):
        
        img = cv2.imread(img_list[i]).astype(np.uint8)
        img = img[...,::-1]

        full_path = os.path.normpath(label_list[i])
        camera_label = open(label_list[i])
        camera_data = camera_label.read()
        camera_data = json.loads(camera_data)
        camera = get_camera(camera_data)

        full_path = full_path.split(os.sep)
        city = full_path[-2]
        label_name = full_path[-1]
        
        pred_label = os.path.join(pred_dir, city, label_name)
        label = open(pred_label)
        json_data = label.read()
        data = json.loads(json_data)
        if len(data['objects']) == 0:
            continue

        sorted_targets, sorted_objects = get_cs_targets_and_objects(['car'], camera, data)
        
        input_data = zip(sorted_targets, sorted_objects)

        city_dir = os.path.join(top_dir, city)
        mkdir(city_dir)

        fname = full_path[-1].split('.')[0]
        fname3d = os.path.join(city_dir, fname) + '_3D.png'
        fname2d = os.path.join(city_dir, fname) + '_2D.png'
        plt.imsave(fname3d, visualize_3d(img.copy(), input_data))
        plt.imsave(fname2d, visualize_2d(img.copy(), sorted_targets))

        if bev:
            _, gt = get_cs_targets_and_objects(['car'], camera, camera_data)
            bev_image =  draw_bev(sorted_objects, gt)
            fname_bev = os.path.join(city_dir, fname) + '_BEV.png'
            plt.imsave(fname_bev, bev_image)

            
if __name__ == '__main__':
    cfg = get_cfg_defaults()
    vis_all(cfg, bev=True)
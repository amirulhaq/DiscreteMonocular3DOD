from PIL import Image
# from kornia.geometry import depth
# from numpy.core.numeric import indices
import torch
from torchvision import transforms, datasets
from copy import deepcopy
import os
import cv2
from tqdm import tqdm
import numpy as np
import json
import math
from collections import namedtuple
from pyquaternion import Quaternion

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Iterable
from .cs_target_transform import cs_target_transform, line_intersect
from .post_processing.decode import rot_y2alpha

from .box3dImageTransform import ( Camera, 
                                Box3dImageTransform,
                                CRS_V,
                                CRS_C,
                                CRS_S
                                )
from .cs_annotation import CsBbox3d



class cs_data(datasets.vision.VisionDataset):
        # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    def __init__(
            self,
            root: str,
            split: str = "train",
            mode: str = "fine",
            target_type: str = "instance",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            cfg=None
    ) -> None:
        super().__init__(root) #root, transforms, transform, target_transform
        self.root = root
        self.transform = transform
        self.transforms = transforms
        self.target_transform = target_transform
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        self.json_dir = os.path.join(self.root, 'gtBbox3d', split)
        self.labels_json = []
        self.panoptic_dir = os.path.join(self.root, 'gtFinePanopticParts', split)
        self.panoptic = []

        self.target_class = ['car']
        self.depth_gaussian = cfg.MODEL.HEAD.DEPTH_GAUSSIAN
        self.depth_bin = cfg.MODEL.HEAD.DEPTH_BIN
        self.gauss_shape = cfg.MODEL.HEAD.DEPTH_GAUSSIAN_SHAPE
        
        self.resize = cfg.MODEL.BACKBONE.DOWN_RATIO if cfg is not None else 0
        assert isinstance(self.resize, int)

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val", "demoVideo")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        for city in tqdm(os.listdir(self.images_dir)):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                self._get_target_suffix(self.mode, self.target_type))
                target_name = os.path.join(target_dir, target_name)
                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_name)

                if self.split == "demoVideo":
                    continue
                
                # Add 3D information labels
                json_label = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                'gtBbox3d.json')
                json_file = os.path.join(self.json_dir, city, json_label)
                assert os.path.isfile(json_file), '{} not found.'.format(json_file)
                self.labels_json.append(json_file)

                # Add panoptic image labels
                panoptic_label = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                'gtFinePanopticParts.tif')
                panoptic_file = os.path.join(self.panoptic_dir, city, panoptic_label)
                assert os.path.isfile(panoptic_file), '{} not found.'.format(panoptic_file)
                self.panoptic.append(panoptic_file)
        
        # self.max_angle = -1e6
        # self.min_angle = 1e6

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, orientation_gt, depth_gt, and json_data).
            image = RGB image (in torch.tensor format).
            orientatation_gt = masks (in torch.tensor format) with each pixel contains the information of the corresponding objects' orientation.
            depth_gt = masks (in torch.tensor format) with each pixel contains the information of the corresponding objects' depth distance.
            json_data = a dictionary containing objects 2d and 3d information. 2d in xywh, 3d in xyc, and rotation.
        """
        preprocess = transforms.Compose([
            transforms.Resize((1024, 2048)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        image = Image.open(self.images[index]).convert('RGB')
        # Ucomment this line below if you are using OpenCV to read the image since OpenCV read image as BGR
        # image = image[:, :, ::-1] # DLA expect RGB input
        image = preprocess(image)
        
        target = cv2.imread(self.targets[index], cv2.IMREAD_GRAYSCALE)
        panoptic = np.array(Image.open(self.panoptic[index]), dtype=np.uint32)

        if self.resize:
            res_width = math.floor(target.shape[1]/self.resize)
            res_height = math.floor(target.shape[0]/self.resize)
            target = cv2.resize(target, (res_width, res_height), interpolation=cv2.INTER_NEAREST)
            panoptic = cv2.resize(panoptic.astype(np.float32), (res_width, res_height), interpolation=cv2.INTER_NEAREST)  

        json_data = self.labels_json[index]
        json_data = open(json_data)
        json_data = json.loads(json_data.read())
        target_class_only = self.get_specific_class(self.target_class, json_data)
        sorted_targets, sorted_cs_objects = target_class_only['targets'], target_class_only['objects']
        if target is None:
            raise ValueError('Target type is None')
        
        if panoptic is None:
            raise ValueError('Panoptic type is None')
        
        # orientation = np.zeros([80, target.shape[0], target.shape[1]], dtype=np.float32)
        # orientation = np.zeros([target.shape[0], target.shape[1], 81], dtype=np.float32)
        orientation = np.zeros([target.shape[0], target.shape[1], 6], dtype=np.float32)
        # depthGT = np.zeros([120, target.shape[0], target.shape[1]], dtype=np.float32)
        if self.depth_gaussian:
            depth = np.zeros([target.shape[0], target.shape[1], self.depth_bin], dtype=np.float32)
        else:
            depth = np.zeros([1, target.shape[0], target.shape[1]], dtype=np.float32)
        rot_y = np.zeros([target.shape[0], target.shape[1]], dtype=np.float32)
        
        pan_in = self.encode_segmap(np.zeros([120, target.shape[0], target.shape[1]], dtype=np.float32), target, panoptic, 'depth', sorted_targets, sorted_cs_objects)
        offset_in = np.argmax(pan_in, axis=0)
        off_res = cs_target_transform(offset_in)
        offset, offset_mask = off_res['offset'], off_res['offset_mask']
        
        '''
        center regression, offset, 3d-2d center disparity, and 3d dimension
        '''
        heatmap = np.zeros([1, target.shape[0], target.shape[1]], dtype=np.float32)
        disparity2d3d = np.zeros([2, target.shape[0], target.shape[1]], dtype=np.float32)
        dim = np.zeros([3, target.shape[0], target.shape[1]], dtype=np.float32)
        index_mask = np.zeros([2, target.shape[0], target.shape[1]], dtype=np.int8)

        self.get_targets(heatmap, dim, disparity2d3d, orientation, depth, index_mask, sorted_targets, sorted_cs_objects, rot_y=rot_y)
        
        
        ret = dict(
            image=image, \
            orientation=torch.from_numpy(orientation).float(), \
            depth=torch.from_numpy(depth).float(), \
            heatmap=torch.from_numpy(heatmap).float(), \
            dim=torch.from_numpy(dim).float(), \
            center_regression=offset.float(), \
            center_disparity=torch.from_numpy(disparity2d3d).float(), 
            index_mask = torch.from_numpy(index_mask),
            offset_mask = offset_mask.float(),
            proj_matrix=torch.from_numpy(self.proj_matrix),
            label_idx = self.labels_json[index],
            rot_y=torch.from_numpy(rot_y).float()
            # min_max_angle = [self.min_angle, self.max_angle]
        )
        return ret

    def __len__(self) -> int:
        return len(self.images)
    
    
    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)

    # You may need to change it into @classmethod in the future
    def encode_segmap(self, array, mask, pan, classify, json_data, cs_objects):
        '''
        Sets classes other than object of interest to zero so they won't be considered for training
        '''
        panoptic = deepcopy(pan)
        panoptic = panoptic // 100
        mask[mask != 101] = 0
        panoptic = panoptic * (mask/101)
        if len(np.unique(panoptic)) < len(json_data):
            raise ValueError(('Number of objects does not match! '
                                'Number of unique instance = {}. ' 
                                'Number of json labels = {}.').format(len(np.unique(panoptic)), 
                                                                        len(json_data)))

        instances = []
        for id, (object_3d, cs_object) in enumerate(zip(json_data, cs_objects)):
            instance = object_3d['instanceId']
            instances.append(instance)
            _, center_C, rot_C = cs_object.get_parameters(coordinate_system=CRS_C)

            if classify == 'depth':
                depth_x = center_C[0]
                discrete_class = self.discrete_depth_class(depth_x)
                index = np.where(panoptic == instance)
                array[discrete_class][index] = 1.

            else:
                raise ValueError('Unknown argument. {} is not a valid argument for classify.'.format(classify))


        return array
    
    def discrete_orientation_class(self, alpha):
        # return [alpha, 0, 0, 0, 0, 0, 0, 0]
        ret = [0, 0, 0, 1, 0, 1]
        if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
            r = alpha - (-0.5 * np.pi)
            ret[0] = 1
            ret[2], ret[3] = 0.5 * (np.sin(r) + 1), (0.5 * np.cos(r) + 1)
        if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
            r = alpha - (0.5 * np.pi)
            ret[1] = 1
            ret[4], ret[5] = 0.5 * (np.sin(r) + 1), 0.5 * (np.cos(r) + 1)
        return ret

    # You may need to change it into @classmethod in the future 
    def discrete_depth_class(self, di, depth_min=2.0, depth_max=100, num_bins=120):
        '''
        mode: LID,
        num_bins: 80,
        depth_min: 2.0,
        depth_max: 46.8
        '''
        if di > 98:
            di = 98
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * math.sqrt(1 + 8 * (di - depth_min) / bin_size)
        return int(indices)

    def gaussian_depth_class(self, center_z, num_bins=120, gauss_shape=3):
        gauss_range = gauss_shape
        gaussian = self.gaussian1D((gauss_shape*2)+1)

        discrete_class = self.discrete_depth_class(center_z,  num_bins=num_bins)
        l = max(0, discrete_class-gauss_range-1)
        r = min(num_bins-1, discrete_class+gauss_range)
        gauss_mask = np.zeros(num_bins)
        gauss_mask[l:r] = gaussian[l-r:]

        return gauss_mask
    
    # @classmethod in the future
    def get_specific_class(self, target_class:List[str], json_data):
        camera = Camera(fx=json_data["sensor"]["fx"],
                    fy=json_data["sensor"]["fy"],
                    u0=json_data["sensor"]["u0"],
                    v0=json_data["sensor"]["v0"],
                    sensor_T_ISO_8855=json_data["sensor"]["sensor_T_ISO_8855"])
        
        self.proj_matrix = self.get_projection_matrix(camera)

        target_objects = []
        cs_data = json_data.get('objects', {})
        for cs_object in cs_data:
            if cs_object['label'] in target_class:
                target_objects.append(cs_object)
        # Sort objects based on the depth location
        sorted_targets = sorted(target_objects, key=lambda item: item['3d']['center'][0], reverse=True)
        sorted_objects = [] # saved in citiscape object format
        # center3ds = []
        for sorting in sorted_targets:
            obj = CsBbox3d()
            obj.fromJsonText(sorting)
            box3d_annotation = Box3dImageTransform(camera=camera)
            box3d_annotation.initialize_box_from_annotation(obj, coordinate_system=CRS_V)
            sorted_objects.append(box3d_annotation)
            # box_vertices_I = box3d_annotation.get_vertices_2d()
            # _, center_C, _ = box3d_annotation.get_parameters(coordinate_system=CRS_C)
            # center3d = line_intersect(box_vertices_I['BRT'], box_vertices_I['FLB'], \
            #                     box_vertices_I['FLT'], box_vertices_I['BRB'])
            # center3ds.append(center3d)
            #sorted_objects.append(box_vertices_I)

        return dict(targets=sorted_targets, 
            objects=sorted_objects)
            # center3ds=center3ds)

    def gaussian_radius(self, h, w, thresh_min=0.7):
        a1 = 1
        b1 = h + w
        c1 = h * w * (1 - thresh_min) / (1 + thresh_min)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (h + w)
        c2 = (1 - thresh_min) * w * h
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * thresh_min
        b3 = -2 * thresh_min * (h + w)
        c3 = (thresh_min - 1) * w * h
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)

        return min(r1, r2, r3)

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def gaussian1D(self, shape=3, sigma=1): # default shape = 5
        m = (shape - 1.) / 2. 
        y = np.ogrid[-m:m+1]

        h = np.exp(-(y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def line_distance(self, a, b):
        x1, y1 = a[0], a[1]
        x2, y2 = b[0], b[1]
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return dist

    # def get_targets(self, heatmap, dim, center_disparity, orientation, depth, index_mask, targets, cs_objects, k=1):
    def get_targets(self, heatmap, dim, center_disparity, orientation, depth, index_mask, targets, cs_objects, k=1, rot_y=None):
        for target, obj in zip(targets, cs_objects):
            target = target['2d']['amodal']
            if self.resize:
                target = [x/self.resize for x in target]
            center = (target[0] + 0.5 * target[2], target[1] + 0.5 * target[3])
            radius = self.gaussian_radius(target[3], target[2])
            radius = max(0, int(radius))
            diameter = 2 * radius + 1
            gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
            
            # if orientation is not None:
            size_C, depth_C, rot_C = obj.get_parameters(coordinate_system=CRS_C)
            '''
            size_C[0] = Length
            size_C[1] = Width
            size_C[2] = Height 
            '''
            yaw = Quaternion(rot_C).yaw_pitch_roll[0] # (Quaternion(rot_C).yaw_pitch_roll[0] + np.pi) / (2 * np.pi) // 0.025
            # orientation_class = int(0 if orientation_class < 0 else 79 if orientation_class > 79 else orientation_class)

            x, y = int(center[0]), int(center[1])
            vertices = obj.get_vertices_2d()

            center3d = line_intersect(vertices['BRT'], vertices['FLB'], vertices['FLT'], vertices['BRB'])
            x_3d, y_3d = center3d[0], center3d[1]
            if self.resize:
                x_3d /= self.resize
                y_3d /= self.resize
            center_disparity[0, y, x] = center[1] - y_3d
            center_disparity[1, y, x] = center[0] - x_3d

            l_3d = size_C[0] # (self.line_distance(vertices['FRB'], vertices['BRB']) + self.line_distance(vertices['FLT'], vertices['BLT']))/2
            w_3d = size_C[1] # (self.line_distance(vertices['FRB'], vertices['FLB']) + self.line_distance(vertices['BLT'], vertices['BRT']))/2
            h_3d = size_C[2] # (self.line_distance(vertices['FRB'], vertices['FRT']) + self.line_distance(vertices['BLB'], vertices['BLT']))/2

            dim[0, y, x] = l_3d/10
            dim[1, y, x] = w_3d/4
            dim[2, y, x] = h_3d/4

            # alpha = rot_y2alpha(yaw, x, self.proj_matrix[0, 2], self.proj_matrix[0, 0])
            orientation[y, x] = self.discrete_orientation_class(yaw)
            if self.depth_gaussian:
                depth[y, x] = self.gaussian_depth_class(depth_C[0], self.depth_bin, self.gauss_shape)
            else:
                depth[0, y, x] = self.discrete_depth_class(depth_C[0], self.depth_bin)
            
            index_mask[:, y, x] = 1
            if rot_y is not None:
                rot_y[y, x] = yaw
            

            height, width = heatmap.shape[1:3]
                
            left, right = min(x, radius), min(width - x, radius + 1)
            top, bottom = min(y, radius), min(height - y, radius + 1)

            masked_heatmap  = heatmap[0, y - top:y + bottom, x - left:x + right]
            masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
            if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
                np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
            # heatmap[y - top:y + bottom, x - left:x + right] = masked_heatmap
        # return heatmap, dim, center_disparity, orientation, depth, index_mask

    def get_projection_matrix(self, camera):
        K_matrix = np.zeros((3, 3))
        K_matrix[0][0] = camera.fx
        K_matrix[0][2] = camera.u0
        K_matrix[1][1] = camera.fy
        K_matrix[1][2] = camera.v0
        K_matrix[2][2] = 1
        return K_matrix

def iterable_to_str(iterable: Iterable) -> str:
    return "'" + "', '".join([str(item) for item in iterable]) + "'"

T = TypeVar("T", str, bytes)

def verify_str_arg(
    value: T, arg: Optional[str] = None, valid_values: Iterable[T] = None, custom_msg: Optional[str] = None,
) -> T:
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = ("Unknown value '{value}' for argument {arg}. "
                   "Valid values are {{{valid_values}}}.")
            msg = msg.format(value=value, arg=arg,
                             valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)

    return value
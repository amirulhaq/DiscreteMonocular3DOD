import cv2 
import numpy as np
import torch
import math
from copy import deepcopy
from pyquaternion import Quaternion

def line_intersect(a1, a2, b1, b2) :
        """ 
        Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([a1,a2,b1,b2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return None, None
        return (x/z, y/z) 

def visualize_2d(img, cs_label, depth_data=None):
    if depth_data is not None:
        minimum = min(sorted, key= lambda x: x['3d']['center'][0])['3d']['center'][0]
        maximum = max(sorted, key= lambda x: x['3d']['center'][0])['3d']['center'][0]
    for box in cs_label:
        x, y, w, h = box['2d']['amodal']
        x, y, w, h = int(x), int(y), int(w), int(h)
        if depth_data is not None:
            color = box['3d']['center'][0]
            alpha = (color - minimum) / (maximum - minimum)
            beta = 1 - alpha
            line_color = 255 - (alpha * 255)
            sub_img = img[y:y+h, x:x+w]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8)
            white_rect[:,:,0] *= 255
            res = cv2.addWeighted(sub_img, beta, white_rect, alpha, 1.0)
            cv2.rectangle(img, (x, y), (x+w, y+h), (int(line_color),0,0), 2)
            img[y:y+h, x:x+w] = res
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
    return img
    # cv2.imshow('Boxes', img)
    # cv2.waitKey(0)

def visualize_3d(img, data):
    dst = deepcopy(img)
    faces = []
    for cs_target, cs_object in data:
        box_vertices_I = cs_object.get_vertices_2d()
        # print("\n     {:>8} {:>8}".format("u[px]", "v[px]")) # u = x, v = y
        for loc, coord in box_vertices_I.items():
            dst = cv2.circle(img, (math.floor(coord[0]),math.floor(coord[1])), radius=0, color=(0, 0, 255), thickness=8)
            # print("{}: {:8.2f} {:8.2f}".format(loc, coord[0], coord[1]))

        center = line_intersect(box_vertices_I['BRT'], box_vertices_I['FLB'], \
            box_vertices_I['FLT'], box_vertices_I['BRB'])
        # print("CTR: {:8.2f} {:8.2f}".format(center[0], center[1]))
        dst = cv2.circle(dst, (math.floor(center[0]),math.floor(center[1])), radius=0, color=(255, 0, 0), thickness=8)

        # draw 2D center
        center2d_x = cs_target['2d']['amodal'][0] + 0.5*cs_target['2d']['amodal'][2]
        center2d_y = cs_target['2d']['amodal'][1] + 0.5*cs_target['2d']['amodal'][3]
        dst = cv2.circle(dst, (math.floor(center2d_x),math.floor(center2d_y)), radius=0, color=(0, 0, 255), thickness=8)
        # draw 3Dbounding box
        dst = cv2.line(dst, tuple(np.floor(box_vertices_I['FLB']).astype(int)), tuple(np.floor(box_vertices_I['FRB']).astype(int)), color=(0, 255, 0), thickness=2)
        dst = cv2.line(dst, tuple(np.floor(box_vertices_I['FLB']).astype(int)), tuple(np.floor(box_vertices_I['BLB']).astype(int)), color=(0, 255, 0), thickness=2)
        dst = cv2.line(dst, tuple(np.floor(box_vertices_I['FLT']).astype(int)), tuple(np.floor(box_vertices_I['FLB']).astype(int)), color=(0, 255, 0), thickness=2)
        dst = cv2.line(dst, tuple(np.floor(box_vertices_I['FRT']).astype(int)), tuple(np.floor(box_vertices_I['FRB']).astype(int)), color=(0, 255, 0), thickness=2)
        dst = cv2.line(dst, tuple(np.floor(box_vertices_I['FLT']).astype(int)), tuple(np.floor(box_vertices_I['FRT']).astype(int)), color=(0, 255, 0), thickness=2)
        dst = cv2.line(dst, tuple(np.floor(box_vertices_I['FLT']).astype(int)), tuple(np.floor(box_vertices_I['BLT']).astype(int)), color=(0, 255, 0), thickness=2)
        dst = cv2.line(dst, tuple(np.floor(box_vertices_I['FRT']).astype(int)), tuple(np.floor(box_vertices_I['BRT']).astype(int)), color=(0, 255, 0), thickness=2)

        dst = cv2.line(dst, tuple(np.floor(box_vertices_I['BLT']).astype(int)), tuple(np.floor(box_vertices_I['BRT']).astype(int)), color=(0, 255, 0), thickness=2)
        dst = cv2.line(dst, tuple(np.floor(box_vertices_I['BLT']).astype(int)), tuple(np.floor(box_vertices_I['BLB']).astype(int)), color=(0, 255, 0), thickness=2)
        dst = cv2.line(dst, tuple(np.floor(box_vertices_I['BRT']).astype(int)), tuple(np.floor(box_vertices_I['BRB']).astype(int)), color=(0, 255, 0), thickness=2)
        dst = cv2.line(dst, tuple(np.floor(box_vertices_I['BLB']).astype(int)), tuple(np.floor(box_vertices_I['BRB']).astype(int)), color=(0, 255, 0), thickness=2)
        dst = cv2.line(dst, tuple(np.floor(box_vertices_I['BRB']).astype(int)), tuple(np.floor(box_vertices_I['FRB']).astype(int)), color=(0, 255, 0), thickness=2)

        face = np.array([ 
                list(np.floor(box_vertices_I['BLT'])), 
                list(np.floor(box_vertices_I['BRT'])),
                list(np.floor(box_vertices_I['BRB'])),
                list(np.floor(box_vertices_I['BLB']))                
        ], np.int32)
        faces.append(face)
    zeros = np.zeros_like(dst)
    mask = cv2.fillPoly(zeros, faces, (0, 200, 0))
    alpha = 1.0
    beta = 0.3
    dst = np.clip((alpha * dst + beta * mask ), a_min=0, a_max=255).astype(np.uint8)

    return dst

def mask2bbox(img):
    '''
    Show bounding box around masks.
    Code obtained from:
    https://stackoverflow.com/questions/58885816/how-to-obtain-boundary-coordinates-of-binary-mask-with-holes
    '''
    if torch.is_tensor(img):
        img = img.cpu().numpy() if img.is_cuda else img.numpy()
    
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = (255, 0, 255)

    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        cv2.drawContours(out, [contour], -1, color, 2)

    return out

def show_center_regression_mask(panGT, src):
    assert len(panGT.shape) == 3, print('Invalid dimension! Input shape should be 2xWxH, but got {} dimension instead.'.format(panGT.shape))
    assert panGT.shape[0] == 2, print('Invalid dimension! Input shape should be 2xWxH, but got {} dimension instead.'.format(panGT.shape))
    offsets = panGT.numpy()

    rgb = []
    zeros = np.zeros_like(offsets[0])
    rgb.append(zeros)
    for offset in offsets: # normalize values in offsety and then offsetx
        offset = np.abs(offset)
        denom = np.max(offset) - np.min(offset)
        offset -= np.min(offset)
        offset /= denom
        offset *= (255.0)
        rgb.append(offset)
            
    rgb = cv2.merge(rgb)
    rgb *= ( 255.0 / np.max(rgb))

    alpha = 1.0
    beta = 0.3
    dst = np.uint8(alpha*(src)+beta*(rgb))
    return dst.astype('uint8')

def draw_bev(pred_data, gt_data=None):
    canvas_bev = np.zeros((600, 1200, 3), dtype=np.uint8)
    scale=5

    mid_ctr = (int(canvas_bev.shape[1]/2), int(canvas_bev.shape[0]))
    bin_size = 5
    for i in range(int(canvas_bev.shape[0]//bin_size//scale)):
        radius = i * scale * bin_size
        cv2.circle(canvas_bev, mid_ctr, radius, color=(50, 50, 50), thickness=2)

    for cs_object in pred_data:
        S, C, R = cs_object.get_parameters(coordinate_system=0)
        z, x = C[0], -C[1]
        l, w = S[0], S[1]
        yaw = Quaternion(R).yaw_pitch_roll[0]
        draw_single_object_bev(canvas_bev, z, l, w, x, yaw, color=(0, 200, 200), scale=scale, thickness=2)
    
    if gt_data is not None:
        for cs_object in gt_data:
            S, C, R = cs_object.get_parameters(coordinate_system=0)
            z, x = C[0], -C[1]
            l, w = S[0], S[1]
            yaw = Quaternion(R).yaw_pitch_roll[0]
            draw_single_object_bev(canvas_bev, z, l, w, x, yaw, color=(200, 0, 0), scale=scale, thickness=2)
    
    return canvas_bev



def draw_single_object_bev(canvas_bev, z3d, l3d, w3d, x3d, ry3d, color=(0, 200, 200), scale=1, thickness=2):
    '''
    Function taken from M3D-RPN
    '''
    w = l3d * scale
    l = w3d * scale
    x = x3d * scale 
    z = canvas_bev.shape[0] - z3d * scale
    r = -ry3d-math.radians(-90) #ry3d*-1

    corners1 = np.array([
        [-w / 2, -l / 2, 1],
        [+w / 2, -l / 2, 1],
        [+w / 2, +l / 2, 1],
        [-w / 2, +l / 2, 1]
    ])

    ry = np.array([
        [+math.cos(r), -math.sin(r), 0],
        [+math.sin(r), math.cos(r), 0],
        [0, 0, 1],
    ])

    corners2 = ry.dot(corners1.T).T

    corners2[:, 0] += w/2 + x + canvas_bev.shape[1] / 2
    corners2[:, 1] += l/2 + z

    draw_line(canvas_bev, corners2[0], corners2[1], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[1], corners2[2], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[2], corners2[3], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[3], corners2[0], color=color, thickness=thickness)

def draw_line(im, v1, v2, color=(0, 200, 200), thickness=1):

    cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)
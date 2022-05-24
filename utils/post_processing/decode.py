from copy import deepcopy
import json
import torch
import numpy as np
import torch.nn as nn
from functools import reduce
from pyquaternion import Quaternion
from skimage import transform as trans
from scipy.spatial.transform import Rotation as R

def nms(x, nms_th):
    def IoU(box0, box1):
        # box0: [x, y, z, d]
        r0 = box0[3] / 2
        s0 = box0[:3] - r0
        e0 = box0[:3] + r0
        r1 = box1[3] / 2
        s1 = box1[:3] - r1
        e1 = box1[:3] + r1
        
        overlap = [max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])) for i in range(3)]
        intersection = reduce(lambda x,y:x*y, overlap)
        union = pow(box0[3], 3) + pow(box1[3], 3) - intersection
        return intersection / union
    # x:[p, z, w, h, d]
    if len(x) == 0:
        return x
    x = x[np.argsort(-x[:, 0])]
    bboxes = [x[0]]
    for i in np.arange(1, len(x)):
        bbox = x[i]
        flag = 1
        for j in range(len(bboxes)):
            if IoU(bbox[1:5], bboxes[j][1:5]) > nms_th:
                flag = -1
                break
            if flag == 1:
                bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def _nms(heat, kernel=7):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def get_transfrom_matrix(center_scale, output_size):
    def get_3rd_point(point_a, point_b):
        d = point_a - point_b
        point_c = point_b + np.array([-d[1], d[0]])
        return point_c

    center, scale = center_scale[0], center_scale[1]
    # todo: further add rot and shift here.
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    src_dir = np.array([0, src_w * -0.5])
    dst_dir = np.array([0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    get_matrix = trans.estimate_transform("affine", src, dst)
    matrix = get_matrix.params

    return matrix.astype(np.float32)

def decode_location(points,
                    points_offset,
                    depths,
                    Ks,
                    trans_mats):
        '''
        retrieve objects location in camera coordinate based on projected points
        Args:
            points: projected points on feature map in (x, y)
            points_offset: project points offset in (delta_x, delta_y)
            depths: object depth z
            Ks: camera intrinsic matrix, shape = [N, 3, 3]
            trans_mats: transformation matrix from image to feature map, shape = [N, 3, 3]

        Returns:
            locations: objects location, shape = [N, 3]
        '''
        device = points.device

        Ks = Ks.to(device=device)
        trans_mats = trans_mats.to(device=device)

        # number of points
        N = points_offset.shape[0]
        # batch size
        # N_batch = Ks.shape[0]
        # batch_id = torch.arange(N_batch).unsqueeze(1)
        # obj_id = batch_id.repeat(1, N // N_batch).flatten()

        trans_mats_inv = trans_mats.inverse()#[obj_id]
        trans_mats_inv = torch.stack([trans_mats_inv] * N)
        Ks_inv = Ks.inverse()#[obj_id]
        Ks_inv = torch.stack([Ks_inv] * N)
        # print(Ks_inv.shape)

        points = points.view(-1, 2)
        assert points.shape[0] == N
        proj_points = points + points_offset
        # transform project points in homogeneous form.
        proj_points_extend = torch.cat(
            (proj_points, torch.ones(N, 1).to(device=device)), dim=1)
        # expand project points as [N, 3, 1]
        proj_points_extend = proj_points_extend.unsqueeze(-1)
        # transform project points back on image
        proj_points_img = torch.matmul(trans_mats_inv, proj_points_extend)
        # with depth
        proj_points_img = proj_points_img * depths.view(N, -1, 1)
        proj_points_img = proj_points_img.double()
        # transform image coordinates back to object locations
        locations = torch.matmul(Ks_inv, proj_points_img)
        locations = locations.squeeze(2)

        # ret = torch.zeros_like(locations)
        # ret[:, 0] = locations[:, 2]
        # ret[:, 1] = locations[:, 0]
        # ret[:, 2] = locations[:, 1]

        return locations

def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + torch.atan2(x - cx, fx)
    pi = torch.ones_like(rot_y) * np.pi
    idx = (rot_y > pi).float()
    # if rot_y > pi:
    #   rot_y -= 2 * pi
    # if rot_y < -pi:
    #   rot_y += 2 * pi
    return (idx * (rot_y - 2 * pi)) + ((rot_y + 2 * pi) * (1 - idx))

def rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha

def decode_rot(rot):
    idx = (rot[:, 0] > rot[:, 1]).float()
    rot[:, 2:] = (rot[:, 2:] * 2) - 1
    rot1 = torch.atan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    rot2 = torch.atan2(rot[:, 4], rot[:, 5]) + ( 0.5 * np.pi)
    return (rot1 * idx) + (rot2 * (1 - idx))
    

def encode_location(K, ry, dims, locs): 
    # 3D to projected 2D location in System Coordinate of CityScapes
    l, h, w = dims[0], dims[1], dims[2]
    x, y, z = locs[0], locs[1], locs[2]

    x_corners = [0, l, l, l, l, 0, 0, 0]
    y_corners = [0, 0, h, h, 0, 0, h, h]
    z_corners = [0, 0, 0, w, w, w, w, 0]

    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    z_corners += - np.float32(w) / 2

    corners_3d = np.array([x_corners, y_corners, z_corners])
    rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = np.matmul(rot_mat, corners_3d)
    corners_3d += np.array([x, y, z]).reshape([3, 1])

    loc_center = np.array([x, y - h / 2, z])
    proj_point = np.matmul(K, loc_center)
    proj_point = proj_point[:2] / proj_point[2]

    corners_2d = np.matmul(K, corners_3d)
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array([min(corners_2d[0]), min(corners_2d[1]),
                      max(corners_2d[0]), max(corners_2d[1])])

    return proj_point, box2d, corners_3d

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def ddd_decode(prediction, target, cfg, K=40, conf_thres=0.25):
    heat, rot, depth, dim, dis = prediction['heatmap'], prediction['orientation'], prediction['depth'], prediction['dim'], prediction['c_dis']
    proj_matrix = target['proj_matrix']

    batch, _, h, w = heat.shape
    
    size = np.array([2048, 1024], dtype=np.float32)
    center = np.array([i / 2 for i in size], dtype=np.float32)
    center_size = [center, size]
    trans_mat = get_transfrom_matrix(
            center_size,
            [w, h]
        )
    trans_mat = torch.from_numpy(trans_mat)
    heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat, kernel=7)
    if cfg.MODEL.HEAD.DEPTH_GAUSSIAN:
        depth = torch.argmax(depth, dim=1)
      
    scores, _, _, ys, xs = _topk(heat, K=K)

    locs, dims, rots, quats, ret_scores =[], [], [], [], []
    for b in range(batch):
        id = torch.where(scores[b] > conf_thres)
        sb = scores[b]
        sb = sb[id]
        xb, yb = xs[b], ys[b]
        xb, yb = xb[id], yb[id]

        points = torch.cat([xb.view(-1,1), yb.view(-1,1)], dim=1)
        calib = proj_matrix[b]

        if points.shape[0] > 0:
            int_point = points.short()
            if cfg.MODEL.HEAD.DEPTH_GAUSSIAN:
                dec_depths = [depth[b, y, x] for x, y in int_point]
                dec_depths = decode_depth(dec_depths)
                dec_depths = torch.stack(dec_depths)
            else:
                dec_depths = [depth[b, 0, y, x] / 10 for x, y in int_point]
                dec_depths = torch.stack(dec_depths)

            rot_list = [rot[b, :, y, x] for x, y in int_point]
            rot_cp = deepcopy(rot_list)
            rot_ = torch.stack(rot_cp)
            dec_rot = decode_rot(rot_)
            quat = ypr2quaternion(dec_rot)

            dim_list = [dim[b, :, y, x] for x, y in int_point]
            dec_dim = torch.stack(dim_list)
            dec_dim = decode_dim(dec_dim)

            point_offset = [dis[b, :, y, x] for x, y in int_point]
            point_offset = torch.stack(point_offset)
            point_offset = torch.zeros_like(point_offset)
            loc = decode_location(points, point_offset, dec_depths, calib, trans_mat) # location in 3d spaces
        else:
            loc, dec_rot, dec_dim, sb, quat = [], [], [], [], []
        locs.append(loc)
        dims.append(dec_dim)
        rots.append(dec_rot)
        quats.append(quat)
        ret_scores.append(sb)
        # tojson(locs, dec_rot, dec_dim, filename)
    
    # xs = xs.view(batch, K, 1) + 0.5
    # ys = ys.view(batch, K, 1) + 0.5
    
    # scores = scores.view(batch, K, 1)
    # xs = xs.view(batch, K, 1)
    # ys = ys.view(batch, K, 1)
      
    # detections = torch.cat(
    #     [xs, ys, scores, rot, depth, dim, _], dim=2)
  
    return locs, rots, dims, quats, ret_scores

def ypr2quaternion(yaw, pitch=None, roll=None):
    N = yaw.shape[0]
    ypr = np.zeros((N, 3), dtype=np.float32)
    ret = torch.zeros(N, 4, device=yaw.device)
    if pitch is None and roll is None:
        ypr[:, 0] = 0.
        ypr[:, 1] = - yaw.cpu()
        ypr[:, 2] = 0.
    else:
        ypr[:, 0] = pitch.cpu()
        ypr[:, 1] = - yaw.cpu()
        ypr[:, 2] = roll.cpu()
    for i in range(ret.shape[0]):
        try:
            q = Quaternion(axis=ypr[i], radians=3.14159265)
        except: print(ypr[i])
        ret[i, 0], ret[i, 1], ret[i, 2], ret[i, 3] = q[0], q[1], q[2], q[3]
    return ret

def quaternion2ypr(q):
    angles = [0, 0, 0]
    sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
    angles[2] = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q[0] * q[2] - q[3] * q[1])
    if (abs(sinp) >= 1):
        angles[1] = np.copysign(np.pi / 2, sinp)
    else:
        angles[1] = np.arcsin(sinp)

    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
    angles[0] = torch.atan2(siny_cosp, cosy_cosp)

    return angles

def decode_depth(ids, dmin=2.0, dmax=100., num_bin=120):
    bin_size = 2 * (dmax - dmin) / (num_bin * (1 + num_bin))
    ret = []
    for idx in ids:
        idx = dmin + (bin_size * ((2 * (idx + 0.5)) ** 2 - 1) / 8 )
        ret.append(idx)
    return ret

def decode_dim(dims):
    dims[:, 0] *= 10
    dims[:, 1] *= 4
    dims[:, 2] *= 4
    return dims
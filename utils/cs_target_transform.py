import numpy as np
import torch
from skimage import transform as trans

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


def affine_transform(point, matrix):
    point_exd = np.array([point[0], point[1], 1.])
    new_point = np.matmul(matrix, point_exd)

    return new_point[:2]

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

def cs_target_transform(panoptic):
    """Generates the training target.
    reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py
    reference: https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_panoptic_fpn.py#L18
    Args:
        panoptic: numpy.array, colored image encoding panoptic label.
        segments: List, a list of dictionary containing information of every segment, it has fields:
            - id: panoptic id, after decoding `panoptic`.
            - category_id: semantic class id.
            - area: segment area.
            - bbox: segment bounding box.
            - iscrowd: crowd region.
    Returns:
        A dictionary with fields:
            - semantic: Tensor, semantic label, shape=(H, W).
            - foreground: Tensor, foreground mask label, shape=(H, W).
            - center: Tensor, center heatmap, shape=(1, H, W).
            - center_points: List, center coordinates, with tuple (y-coord, x-coord).
            - offset: Tensor, offset, shape=(2, H, W), first dim is (offset_y, offset_x).
            - semantic_weights: Tensor, loss weight for semantic prediction, shape=(H, W).
            - center_weights: Tensor, ignore region of center prediction, shape=(H, W), used as weights for center
                regression 0 is ignore, 1 is has instance. Multiply this mask to loss.
            - offset_weights: Tensor, ignore region of offset prediction, shape=(H, W), used as weights for offset
                regression 0 is ignore, 1 is has instance. Multiply this mask to loss.
    """
    height, width = panoptic.shape[0], panoptic.shape[1]
    semantic = np.zeros_like(panoptic, dtype=np.uint8)
    center = np.zeros((1, height, width), dtype=np.float32)
    center_pts = []
    offset = np.zeros((2, height, width), dtype=np.float32)
    offset_mask = np.zeros_like(offset, dtype=np.uint8)
    y_coord = np.ones_like(panoptic, dtype=np.float32)
    x_coord = np.ones_like(panoptic, dtype=np.float32)
    y_coord = np.cumsum(y_coord, axis=0) - 1
    x_coord = np.cumsum(x_coord, axis=1) - 1
    # Generate pixel-wise loss weights
    semantic_weights = np.ones_like(panoptic, dtype=np.uint8)
    # 0: ignore, 1: has instance
    # three conditions for a region to be ignored for instance branches:
    # (1) It is labeled as `ignore_label`
    # (2) It is crowd region (iscrowd=1)
    # (3) (Optional) It is stuff region (for offset branch)
    center_weights = np.zeros_like(panoptic, dtype=np.uint8)
    offset_weights = np.zeros_like(panoptic, dtype=np.uint8)

    sigma = 8
    size = 6 * sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # print('Panoptic inside function = ', np.unique(panoptic))
    # print('G = ', g.shape)
    for seg in np.unique(panoptic):
        if seg == 0:
            continue
        center_weights[panoptic == seg] = 1
        offset_weights[panoptic == seg] = 1
        mask_index = np.where(panoptic == seg)
        # print(len(mask_index[0]))
        if len(mask_index[0]) == 0:
            # the instance is completely cropped
            continue
        # Find instance area
        ins_area = len(mask_index[0])
        if ins_area < 0:
            semantic_weights[panoptic == seg] = 1

        center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
        center_pts.append([center_y, center_x])

        # generate center heatmap
        y, x = int(center_y), int(center_x)
        # outside image boundary
        if x < 0 or y < 0 or \
                x >= width or y >= height:
            continue
        # upper left
        ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
        # bottom right
        br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

        c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
        a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

        cc, dd = max(0, ul[0]), min(br[0], width)
        aa, bb = max(0, ul[1]), min(br[1], height)
        center[0, aa:bb, cc:dd] = np.maximum(
            center[0, aa:bb, cc:dd], g[a:b, c:d])

        # generate offset (2, h, w) -> (y-dir, x-dir)
        offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
        offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
        offset[offset_y_index] = center_y - y_coord[mask_index]
        offset[offset_x_index] = center_x - x_coord[mask_index]
        offset_mask[offset_y_index], offset_mask[offset_x_index] = 1, 1

    return dict(
        semantic=torch.as_tensor(semantic.astype('long')),
        center=torch.as_tensor(center.astype(np.float32)), #heatmap
        center_points=center_pts,
        offset=torch.as_tensor(offset.astype(np.float32)), # non-normalized offset based on pixel distance
        semantic_weights=torch.as_tensor(semantic_weights.astype(np.float32)),
        center_weights=torch.as_tensor(center_weights.astype(np.float32)),
        offset_weights=torch.as_tensor(offset_weights),
        offset_mask=torch.as_tensor(offset_mask)
    )


# General util function to get the boundary of a binary mask.
# 该函数用于获取二进制 mask 的边界
import cv2
import numpy as np
from PIL import Image


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)  # 计算图像对角线长度
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    mask = mask.astype(np.uint8)
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)

    # 因为之前向四周填充了0, 故而这里不再需要四周
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]

    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, ignore_index, dilation_ratio=0.015, cls_num=2,
                        gt_map=dict(),
                        reduce_zero_gt=False):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    # 注意 gt 和 dt 的 shape 不一样
    # gt = gt[0, 0]
    # dt = dt[0]
    # 这里为了让 gt 和 dt 变为 (h, w) 而不是 (1, h, w) 或者 (1, 1, h, w)


    if isinstance(dt, str):
        dt = np.load(dt)

    if isinstance(gt, str):
        gt = Image.open(gt)
    # modify if custom classes
    if gt_map is not None:
        for old_id, new_id in gt_map.items():
            gt[gt == old_id] = new_id
    if reduce_zero_gt:
        # avoid using underflow conversion
        gt[gt == 0] = 255
        gt = gt - 1
        gt[gt == 254] = 255

    # mask = (gt != ignore_index)
    # dt = dt[mask]
    # gt = gt[mask]

    # 注意这里的类别转换主要是为了后边计算边界
    gt = gt.astype(np.uint8)
    dt = dt.astype(np.uint8)

    boundary_iou_list = np.zeros((cls_num, ), dtype=np.float)
    for i in range(cls_num):
        if i==1:
            gt_i = (gt == 255)
            dt_i = (dt == 255)
        else:
            gt_i = (gt == i)
            dt_i = (dt == i)
        gt_boundary = mask_to_boundary(gt_i, dilation_ratio)
        dt_boundary = mask_to_boundary(dt_i, dilation_ratio)
        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()
        if union < 1:
            boundary_iou = 0
            continue
        boundary_iou = intersection / union
        boundary_iou_list[i] = boundary_iou

    return boundary_iou_list
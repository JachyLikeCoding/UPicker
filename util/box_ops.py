# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import math
from torchvision.ops.boxes import box_area

def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.

    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer

    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out

def crop_bbox(feats, bbox, HH, WW=None):
    """
    Take differentiable crops of feats specified by bbox.

    Inputs:
    - feats: Tensor of shape (N, C, H, W)
    - bbox: Bounding box coordinates of shape (N, 4) in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space.
    - HH, WW: Size of the output crops.

    Returns:
    - crops: Tensor of shape (N, C, HH, WW) where crops[i] is the portion of
      feats[i] specified by bbox[i], reshaped to (HH, WW) using bilinear sampling.
    """
    N = feats.size(0)
    assert bbox.size(0) == N
    assert bbox.size(1) == 4
    if WW is None: WW = HH
    bbox = box_cxcywh_to_xyxy(bbox)
    bbox = torch.clamp(bbox, 0.01, 0.99)
    bbox = 2 * bbox - 1
    x0, y0, x1, y1 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    X = tensor_linspace(x0, x1, steps=WW).view(N, 1, WW).expand(N, HH, WW)
    Y = tensor_linspace(y0, y1, steps=HH).view(N, HH, 1).expand(N, HH, WW)
    grid = torch.stack([X, Y], dim=3)
    res = torch.nn.functional.grid_sample(feats, grid, padding_mode='border', align_corners=False)
    return res


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


def complete_box_iou(boxes1, boxes2):
    """
    Complete IoU

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    b1_xy = boxes1[:, None, :2]
    b1_wh = boxes1[:, None, 2:] - b1_xy
    b1_wh_half = b1_wh / 2
    b1_mins = b1_xy
    b1_maxs = b1_xy + b1_wh

    b2_xy = boxes2[:, :2]
    b2_wh = boxes2[:, 2:]- b2_xy
    b2_wh_half = b2_wh / 2
    b2_mins = b2_xy
    b2_maxs = b2_xy + b2_wh

    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxs = torch.min(b1_maxs, b2_maxs)
    intersect_wh = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_maxs))
    # intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    # b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou, union = box_iou(boxes1, boxes2)
    # print('iou2:', iou)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxs = torch.max(b1_maxs, b2_maxs)
    enclose_wh = torch.max(enclose_maxs - enclose_mins, torch.zeros_like(intersect_maxs))

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclose_wh = torch.max(rb - lt, torch.zeros_like(intersect_maxs))

    # 计算对角线距离
    enclose_diagomal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
    ciou = iou - 1.0 * center_distance / (enclose_diagomal + 1e-7)
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(b1_wh[..., 0] / b1_wh[..., 1]) - torch.atan(b2_wh[..., 0] / b2_wh[..., 1]), 2)

    alpha = v / ((1 - iou) + v)
    ciou = ciou - alpha * v
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)
    return ciou


def efficient_box_iou(boxes1, boxes2, eps=1e-7):
    """
    Efficient IoU from paper: Focal and efficient IOU loss for Accurate Bounding Box Regression.
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    # 计算各个包围盒的中心点坐标和宽高信息
    b1_x1 = boxes1[:, 0]
    b1_y1 = boxes1[:, 1]
    b1_x2 = boxes1[:, 2]
    b1_y2 = boxes1[:, 3]
    center_x1 = (b1_x2 + b1_x1) / 2.
    center_y1 = (b1_y2 + b1_y1) / 2.
    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    
    b2_x1 = boxes2[:, 0]
    b2_y1 = boxes2[:, 1]
    b2_x2 = boxes2[:, 2]
    b2_y2 = boxes2[:, 3]
    center_x2 = (boxes2[:, 2] + boxes2[:, 0]) / 2.
    center_y2 = (boxes2[:, 3] + boxes2[:, 1]) / 2.
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 1]

    # 将 x,y,w,h 拼成一个 Nx4 和 Mx4 矩阵
    b1 = torch.stack((b1_x1, b1_y1, w1, h1, center_x1, center_y1), dim=1)
    b2 = torch.stack((b2_x1, b2_y1, w2, h2, center_x2, center_y2), dim=1)

    # 扩展维度，使得 b1 和 b2 都有形状 (N, M, 4)
    b1 = b1[:, None, :]
    b2 = b2[None, :, :]

    cw = torch.max(b1[..., 0]+b1[...,2], b2[..., 0]+b2[...,2]) - torch.min(b1[..., 0], b2[..., 0])
    ch = torch.max(b1[..., 1]+b1[...,3], b2[..., 1]+b2[...,3]) - torch.min(b1[..., 1], b2[..., 1])
    cw2 = cw ** 2 + eps
    ch2 = ch ** 2 + eps

    rho_w2 = (b2[..., 2] - b1[..., 2]) ** 2
    rho_h2 = (b2[..., 3] - b1[..., 3]) ** 2
    rho2 = (b2[..., 4] - b1[..., 4])**2  + (b2[..., 5] - b1[..., 5])**2

    # 得到一个 N x M 的矩阵
    eiou = iou - (rho2 / (cw2 + ch2) + rho_w2 / cw + rho_h2 / ch)

    return eiou


def focal_eiou_loss(boxes1, boxes2, eps=1e-7, gamma=0.05):
    """
    Efficient IoU from paper: Focal and efficient IOU loss for Accurate Bounding Box Regression.
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    # 计算各个包围盒的中心点坐标和宽高信息
    b1_x1 = boxes1[:, 0]
    b1_y1 = boxes1[:, 1]
    b1_x2 = boxes1[:, 2]
    b1_y2 = boxes1[:, 3]
    center_x1 = (b1_x2 + b1_x1) / 2.
    center_y1 = (b1_y2 + b1_y1) / 2.
    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    
    b2_x1 = boxes2[:, 0]
    b2_y1 = boxes2[:, 1]
    b2_x2 = boxes2[:, 2]
    b2_y2 = boxes2[:, 3]
    center_x2 = (boxes2[:, 2] + boxes2[:, 0]) / 2.
    center_y2 = (boxes2[:, 3] + boxes2[:, 1]) / 2.
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 1]

    # 将 x,y,w,h 拼成一个 Nx4 和 Mx4 矩阵
    b1 = torch.stack((b1_x1, b1_y1, w1, h1, center_x1, center_y1), dim=1)
    b2 = torch.stack((b2_x1, b2_y1, w2, h2, center_x2, center_y2), dim=1)

    # 扩展维度，使得 b1 和 b2 都有形状 (N, M, 4)
    b1 = b1[:, None, :]
    b2 = b2[None, :, :]

    cw = torch.max(b1[..., 0]+b1[...,2], b2[..., 0]+b2[...,2]) - torch.min(b1[..., 0], b2[..., 0])
    ch = torch.max(b1[..., 1]+b1[...,3], b2[..., 1]+b2[...,3]) - torch.min(b1[..., 1], b2[..., 1])
    cw2 = cw * cw + eps
    ch2 = ch * ch + eps

    rho_w2 = (b2[..., 2] - b1[..., 2]) * (b2[..., 2] - b1[..., 2])
    rho_h2 = (b2[..., 3] - b1[..., 3]) * (b2[..., 3] - b1[..., 3])
    rho2 = (b2[..., 4] - b1[..., 4])**2  + (b2[..., 5] - b1[..., 5])**2

    # 得到一个 N x M 的矩阵
    eiou = iou - (rho2 / (cw2 + ch2) + rho_w2 / cw2 + rho_h2 / ch2)
    eiou_loss = 1 - eiou
    focal_eiou_loss = (iou**gamma) * eiou_loss
    
    return focal_eiou_loss


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

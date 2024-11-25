# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import time
import cv2
import numpy as np
from typing import Iterable
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision.ops.boxes import batched_nms, nms
from torch.utils.tensorboard import SummaryWriter

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.data_prefetcher import data_prefetcher
from datasets.panoptic_eval import PanopticEvaluator
from datasets.selfdet import selective_search
# from datasets.coco_eval_proposals import edge_boxes
from datasets.voc_eval import VocEvaluator
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from util.plot_utils import plot_prediction

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5   # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def dict_test(x):
    return x['pred_logits']


def train_one_epoch(model: torch.nn.Module, swav_model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, writer, 
                    device: torch.device, epoch: int, max_norm: float = 0,
                    args=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False
    
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    #for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # print('targets:', targets)
        if need_tgt_for_training:
            outputs = model(samples, targets)
        else:
            outputs = model(samples)

        if swav_model is not None:
            with torch.no_grad():
                for elem in targets:
                    elem['patches'] = swav_model(elem['patches'])

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        # total loss
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        writer.add_scalar('Loss/train',float(loss_value),epoch)
        writer.add_scalar('Learning rate', optimizer.param_groups[0]["lr"], epoch)
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()

        losses.backward()
        
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)


        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args=None, writer=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0.5]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if 'loaddet' in output_dir:
            outputs = dict(pred_logits = [], pred_boxes = [])
            for i, target in enumerate(targets):
                image_id = target['image_id'].item()
                loaded = torch.load(str(image_id) + '.pt', map_location = device)
                outputs['pred_logits'].append(loaded['pred_logits'])
                outputs['pred_boxes'].append(loaded['pred_boxes'])
            model = lambda *ignored: outputs

        model_time = time.time()
        
        if need_tgt_for_training:
            outputs = model(samples, targets)
        else:
            outputs = model(samples)
            
        model_time = time.time() - model_time
        
        # calculate the loss
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        print('valid loss: ', sum(loss_dict_reduced_scaled.values()))
        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        for i, result in enumerate(results):
            image_id = targets[i]['image_id'].item()
            keep = nms(result['boxes'].cpu(), result['scores'].cpu(), args.iou_threshold)
            result['boxes'], result['scores'], result['labels'] = result['boxes'][keep].cuda(), result['scores'][keep].cuda(), result['labels'][keep].cuda()

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time = model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    return stats, coco_evaluator


@torch.no_grad()
def viz(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    print('******visualize pretrain results and prediction*******')
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    print('\ndata_loader dataset: ', data_loader.dataset)

    for samples, targets in data_loader:
        print('samples, targets: ', samples, targets)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        top_k = len(targets[0]['boxes'])

        outputs = model(samples)
        indices = outputs['pred_logits'][0].softmax(-1)[..., 1].sort(descending=True)[1][:top_k]
        predictied_boxes = torch.stack([outputs['pred_boxes'][0][i] for i in indices]).unsqueeze(0)
        logits = torch.stack([outputs['pred_logits'][0][i] for i in indices]).unsqueeze(0)
        fig, ax = plt.subplots(1, 3, figsize=(10,3), dpi=200)

        img = samples.tensors[0].cpu().permute(1,2,0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255)
        img = img.astype('uint8')
        h, w = img.shape[:-1]

        # SS results: selective search results
        boxes_ss = get_ss_res(img, h, w, top_k)
        plot_prediction(samples.tensors[0:1], boxes_ss, torch.zeros(1, boxes_ss.shape[1], 4).to(logits), ax[0], plot_prob=False)
        ax[0].set_title('Selective Search')

        # Pred results
        plot_prediction(samples.tensors[0:1], predictied_boxes, logits, ax[1], plot_prob=False)
        ax[1].set_title('Prediction (Ours)')

        # GT Results
        plot_prediction(samples.tensors[0:1], targets[0]['boxes'].unsqueeze(0), torch.zeros(1, targets[0]['boxes'].shape[0], 4).to(logits), ax[2], plot_prob=False)
        ax[2].set_title('GT')

        for i in range(3):
            ax[i].set_aspect('equal')
            ax[i].set_axis_off()

        plt.savefig(os.path.join(output_dir, f'img_{int(targets[0]["image_id"][0])}.jpg'))
        plt.close()


def get_ss_res(img, h, w, top_k):
    print('get_ss_res-----',img.size, h, w, top_k)
    boxes = selective_search(img, h, w)[:top_k]
    boxes = torch.tensor(boxes).unsqueeze(0)
    boxes = box_xyxy_to_cxcywh(boxes)/torch.tensor([w, h, w, h])
    return boxes


# def get_edge_boxes(img, bgr2rgb = (2, 1, 0), algo_edgedet = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz') if os.path.exists('model.yml.gz') else None):
#     edges = algo_edgedet.detectEdges(img[..., bgr2rgb].astype(np.float32) / 255.0)
#     orimap = algo_edgedet.computeOrientation(edges)
#     edges = algo_edgedet.edgesNms(edges, orimap)
#     algo_edgeboxes = cv2.ximgproc.createEdgeBoxes()
#     algo_edgeboxes.setMaxBoxes(50)
#     boxes_xywh, scores = algo_edgeboxes.getBoundingBoxes(edges, orimap)

#     if scores is None:
#         boxes_xywh, scores = np.array([[0, 0.0, img.shape[1], img.shape[0]]]), np.ones((1, ))

#     return boxes_xywh, scores.squeeze()
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    if training:
        samples, targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries 正负样本数量相同
        # bs大小的list，item是全是1的tensor，size是各个image gt的数量
        dn_number = dn_number * 2
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        # 各个image gt的数量
        known_num = [sum(k) for k in known]
        # print('\n[debug]: known_num = ', known_num)

        # 没有gt
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        
        # 所有的1 cat到一起
        unmask_bbox = unmask_label = torch.cat(known)
        # 取出所有gt的label， box
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        img_ids = []
        sizes = []
        for t in targets:
            sizes.append([t['size']])
            try:
                img_ids.append([t['image_id']])
            except:
                print('no image_id')


        # 标识属于哪个图片 like tensor
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        # 返回一个二维张量，其中每一行都是非零值的索引
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        # 拉平
        known_indice = known_indice.view(-1)
        # 并没有使用
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        # （all gt*2*dn_number） 这里的N是bs中所有gt的数量总和
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)

        known_labels_expand = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        # noise on the label
        if label_noise_ratio > 0:
            # print('label_noise_ratio: ', label_noise_ratio)
            p = torch.rand_like(known_labels_expand.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.4)).view(-1)  # half of bbox prob
            # print('chose indice for label noise: ', chosen_indice)
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expand.scatter_(0, chosen_indice, new_label)

        single_pad = int(max(known_num))
        pad_size = int(single_pad * 2 * dn_number)

        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        
        # noise on the box
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            # 中心坐标减去宽高的一半，左上的边界点
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            # 中心坐标加上宽高的一半，右下的边界点
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2
            # torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) 选出的是0 1这两种值
            # torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32)*2-1 选出的是-1或1
            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            # rand_part 0-1 [2*gt count*dn number]
            rand_part = torch.rand_like(known_bboxs)
            
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            # 加上随机的偏移，左上、右下的点随机的进行了偏移
            # print('\n[debug]: rand part = ', rand_part)
            # print('box_noise_scale: ', box_noise_scale)
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            # 裁剪，防止溢出
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            # 左上和右下点的和除以2就是中心点坐标
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            # 右下减去左上的差值，就是高宽
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        # 2*gt count*dn_number
        m = known_labels_expand.long().to('cuda')
        # print('known_labels_expand shape: ', known_labels_expand.shape)
        # print('known bbox expand shape: ', known_bbox_expand.shape)
        
        import numpy as np
        # visualize POS and NEG
        # Extract GT, Positive, and Negative BBoxes
        gt_bboxes = [t['boxes'].cpu().numpy() for t in targets]
        gt_bboxes_flat = np.concatenate(gt_bboxes)
        bbox_counts = [len(bbox) for bbox in gt_bboxes]
        bbox_start_indices = np.cumsum([0] + bbox_counts[:-1])

        def get_bboxes_for_image(bbox_array, start_indices, image_index):
            start_idx = start_indices[image_index]
            end_idx = start_indices[image_index + 1] if image_index + 1 < len(start_indices) else len(bbox_array)
            return bbox_array[start_idx:end_idx]


        pos_bboxes = known_bbox_expand[positive_idx].cpu().numpy()
        neg_bboxes = known_bbox_expand[negative_idx].cpu().numpy()
        # print('gt_bboxes_flat shape: ', gt_bboxes_flat.shape)
        # print('pos bboxes shape: ', pos_bboxes.shape)
        # print('neg bboxes shape: ', neg_bboxes.shape)
        images = samples.tensors

        sizes_flattened = [tensor[0] for tensor in sizes]
        sizes_tensor = torch.stack(sizes_flattened)
        sizes_tensor_cpu = sizes_tensor.cpu()
        sizes = sizes_tensor_cpu.numpy()


        img_ids_flattened = [tensor[0] for tensor in img_ids]
        img_ids_tensor = torch.stack(img_ids_flattened)
        img_ids_tensor_cpu = img_ids_tensor.cpu()
        img_ids = img_ids_tensor_cpu.numpy()
        # print(type(img_ids))
        # print(img_ids)

        # Visualize
        # for i in range(len(gt_bboxes)):  # Assuming samples is a list of images
        #     image = images[i]
        #     gt_bboxes_i = gt_bboxes[i]
        #     pos_bboxes_i = get_bboxes_for_image(pos_bboxes, bbox_start_indices, i)  # Adjust the number based on the GT bboxes count
        #     neg_bboxes_i = get_bboxes_for_image(neg_bboxes, bbox_start_indices, i)
        #     size_i = sizes[i]
        #     id = img_ids[i]
        #     # print(size_i)
        #     # print('gt_bboxes_i: ', gt_bboxes_i)
        #     # print('pos_bboxes_i: ', pos_bboxes_i)
        #     # print('neg_bboxes_i: ', neg_bboxes_i)
        #     visualize_bboxes(image, gt_bboxes_i, pos_bboxes_i, neg_bboxes_i, size_i, id)

        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
        # print('[dn_compoments] input query label:', len(input_query_label), len(input_query_label[0]), len(input_query_label[1]))
        # print('[dn_compoments] input quert bbox: ', len(input_query_bbox), len(input_query_bbox[0]), len(input_query_bbox[1]))
        
        # map in order
        map_known_indice = torch.tensor([]).to('cuda')
        
        if len(known_num):
            # like tensor([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,0])
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }
    else:
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord




import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import cv2

def visualize_bboxes(image, gt_bboxes, pos_bboxes, neg_bboxes, image_size, img_id):
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())  # 归一化
    height, width = image_size

    vis_image = image.copy()

    def draw_boxes(image, boxes, color, label):
        for box in boxes:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            x1, y1, x2, y2 = (x1 * width, y1 * height, x2 * width, y2 * height)
            w = int(w * width)
            h = int(h * height)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
  
            cv2.rectangle(image, (x1-w//2, y1-h//2), (x2-w//2, y2-h//2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    draw_boxes(vis_image, gt_bboxes, (0, 255, 0), 'GT')
    draw_boxes(vis_image, pos_bboxes, (255, 0, 0), 'POS')
    draw_boxes(vis_image, neg_bboxes, (0, 0, 255), 'NEG')
    plt.imshow(vis_image)

    plt.savefig(f'dn_display_{img_id}.png', dpi=400)

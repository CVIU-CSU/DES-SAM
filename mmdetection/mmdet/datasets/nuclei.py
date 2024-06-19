# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp


import numpy as np


from mmdet.registry import DATASETS

import pycocotools.mask as maskUtils
from skimage.draw import polygon2mask

from .metrics import compute_metrics
from .coco import CocoDataset

import cv2
import scipy.io as sio


@DATASETS.register_module()
class NucleiDataset(CocoDataset):

    def evaluate(self,
                 results,
                 metric='bbox',
                 ):
        thr = threshold # The threshold value
        metric_dict = {'img_name': []}
        metric = ['F1', 'aji', 'pq']
        for m in metric:
            if m == 'pq':
                metric_dict['dq'] = []
                metric_dict['sq'] = []
                metric_dict['pq'] = []
                metric_dict['dice'] = []
                metric_dict['iou'] = []
                metric_dict['Adice'] = []
            else:
                metric_dict[m] = []

        for id in range(0, len(results)):
            _, inst_gt, inst_pred = self.gen_mask(results, self.coco, id, thr)
            metrics = compute_metrics(inst_pred, inst_gt, metric)
            for k in metrics.keys():
                metric_dict[k].append(metrics[k])
        del metric_dict['img_name']
        for k in metric_dict.keys():
            metric_dict[k] = np.mean(metric_dict[k])
        return metric_dict
    def gen_mask(self, preds, gts, id, thr):
        img_name = gts.loadImgs(id)[0]['file_name']
        h, w = gts.loadImgs(id)[0]['height'], gts.loadImgs(id)[0]['width']
        inst_id = gts.get_ann_ids(img_ids=id)
        gts_per_img = [gts.load_anns(i) for i in inst_id]
        results_per_img = preds[id]
        pred_iid = 1
        inst_pred = np.zeros((h, w))
        bboxes = np.concatenate(results_per_img[0])
        masks = np.concatenate(results_per_img[1])

        for bbox, mask in zip(bboxes, masks[0]):

            score = bbox[4]
            if score > thr:
                mask = np.ascontiguousarray(maskUtils.decode(mask))
                inst_pred[mask == 1] = pred_iid
                pred_iid += 1
        gt_iid = 1
        inst_gt = np.zeros((h, w))
        for gt in gts_per_img:
            mask = gt[0]['segmentation']
            mask = gts.annToMask(gt[0])
            inst_gt[mask == 1] = gt_iid
            gt_iid += 1
        return img_name, inst_gt, inst_pred
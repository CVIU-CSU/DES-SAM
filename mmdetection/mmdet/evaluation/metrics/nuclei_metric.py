# Copyright (c) OpenMMLab. All rights reserved.

from collections import OrderedDict
import os.path as osp


import numpy as np
from mmdet.datasets.metrics import compute_metrics
from mmdet.evaluation.metrics.coco_metric import CocoMetric


from mmdet.registry import DATASETS, METRICS

import pycocotools.mask as maskUtils
from skimage.draw import polygon2mask


import cv2
import scipy.io as sio
from mmengine.logging import MMLogger
from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
import tempfile
from typing import Any, Dict, Generator, ItemsView, List, Tuple

@METRICS.register_module()
class NucleiMetric(CocoMetric):

    def compute_metrics(self, results):
        logger: MMLogger = MMLogger.get_current_instance()

        thr = Threshold
        metric_dict = {'img_name': []}
        metric = ['F1', 'aji', 'pq','haus']
        for m in metric:
            if m == 'pq':
                metric_dict['dq'] = []
                metric_dict['sq'] = []
                metric_dict['pq'] = []
                metric_dict['dice'] = []
                metric_dict['iou'] = []
                metric_dict['haus'] = []
                metric_dict['Adice'] = []

            else:
                metric_dict[m] = []
        gts, preds = zip(*results)
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix 
        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        for id in range(0, len(results)):
            img_name, inst_gt, inst_pred = self.gen_mask(preds, self._coco_api, id, thr)
            metrics = compute_metrics(inst_pred, inst_gt, metric)
            for k in metrics.keys():
                metric_dict[k].append(metrics[k])
        del metric_dict['img_name']
        eval_results = OrderedDict()
        for k in metric_dict.keys():
            eval_results['m'+k] = np.mean(metric_dict[k])
        return eval_results
    def gen_mask(self, preds, gts, id, thr):
        img_name = gts.load_imgs(id)[0]['file_name']
        h, w = gts.load_imgs(id)[0]['height'], gts.load_imgs(id)[0]['width']
        inst_id = gts.get_ann_ids(img_ids=id)
        gts_per_img = [gts.load_anns(i) for i in inst_id]
        results_per_img = preds[id]
        pred_iid = 1
        inst_pred = np.zeros((h, w))
        scores = results_per_img["scores"]
        masks = results_per_img["masks"]

        for score, mask in zip(scores, masks):
            if score > thr:
                mask = np.ascontiguousarray(maskUtils.decode(mask))
                min_area =70
                mask, _ = remove_small_regions(mask, min_area, mode="holes")
                unchanged = not changed
                mask, _ = remove_small_regions(mask, min_area, mode="islands")
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
        
def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True
        
def poly2mask(points, width, height):
    mask = np.zeros((width, height), dtype=np.int32)
    obj = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, obj, 1)
    return mask

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
from mmdet.structures import SampleList

import copy
import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor
import cv2
from segment_anything import sam_model_registry, SamPredictor
from mmengine.structures import InstanceData

ori_level = 4
@MODELS.register_module()
class FasterRCNNSimple(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

    def set_epoch(self,epoch):
        self.backbone.epoch = epoch

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        
        x = self.extract_feat(batch_inputs)

        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x[:ori_level], rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]      

        rpn_results = []
       
        rpn_results = rpn_results_list
        roi_losses = self.roi_head.loss(x, rpn_results,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses

    def mask_pred(self,predictor,results_list,batch_data_samples):
        #masks
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        for i,batch_img_meta in enumerate(batch_img_metas):
            image = cv2.imread(batch_img_meta['img_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            input_boxes = results_list[i].bboxes
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            results_list[i].masks = masks[:,0,:,:]
        return results_list

    def mask_save(self,results_list,batch_data_samples):
        #masks
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        l = len(batch_img_metas[0]['img_path'].split("/")[-1]) + len(batch_img_metas[0]['img_path'].split("/")[-2]) + 1
        root_path = batch_img_metas[0]['img_path'][:-l]
        
        pred_path = root_path + "pred_result"
        import os
        import numpy as np
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
        for i,batch_img_meta in enumerate(batch_img_metas):
            mask_name = batch_img_meta['img_path'].split("/")[-1].replace(".png",".npy")
            masks = results_list[i].masks 
            save_path = os.path.join(pred_path,mask_name)
            np.save(save_path,masks.cpu().numpy())
        return results_list

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        
        # sam
        #sam_checkpoint = "weight/sam_vit_b_01ec64.pth"
        #model_type = "vit_b"

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x[:ori_level], batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)
        

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
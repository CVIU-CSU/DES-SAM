# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from mmdet.models.utils.misc import multi_apply

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.bbox.transforms import get_box_tensor
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig

@MODELS.register_module()
class StandardRoIMaskHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 prompt_encoder: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 mask_label_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 ori_level: int = 0) -> None:
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            self.shared_head = MODELS.build(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(prompt_encoder, mask_head, mask_label_head)
                
        self.ori_level = ori_level
        self.init_assigner_sampler()

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            self.bbox_sampler = TASK_UTILS.build(
                self.train_cfg.sampler, default_args=dict(context=self))

    def init_bbox_head(self, bbox_roi_extractor: ConfigType,
                       bbox_head: ConfigType) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def init_mask_head(self,prompt_encoder: ConfigType,mask_head: ConfigType,mask_label_head: ConfigType) -> None:
        self.prompt_encoder = MODELS.build(prompt_encoder)
        self.mask_head = MODELS.build(mask_head)
        self.mask_label_head = MODELS.build(mask_label_head)

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')
            
            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x[:4]])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results,pos_inds = self.bbox_loss(x[:4], sampling_results)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x,sampling_results,
                                          batch_gt_instances,
                                          batch_img_metas)
            losses.update(mask_results['loss_mask'])
        return losses

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois) # [512,256,7,7]
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results,bbox_loss_and_target['pos_inds']
    
    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  batch_gt_instances: InstanceList,
                  batch_img_metas: list) -> dict:
        # during trainning rois can be proposals
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results]) # 256,5
        # posroi box
        gt_target_box = []
        for bgi,res in zip(batch_gt_instances,sampling_results):
            gt_target_box.append(bgi.bboxes[res.pos_assigned_gt_inds])
        gt_target_box = torch.cat(gt_target_box)
        pos_rois = pos_rois.split([len(res.pos_priors) for res in sampling_results],0)
        gt_target_box = gt_target_box.split([len(res.pos_priors) for res in sampling_results],0)
        low_res_masks,mask_tokens_preds,low_res_mask_labels,mask_tokens_pred_labels = multi_apply(self._mask_forward,x[self.ori_level],pos_rois,batch_img_metas,gt_target_box)
        mask_results = dict(mask_preds=low_res_masks,mask_tokens_preds=mask_tokens_preds,gt_mask=low_res_mask_labels,gt_token=mask_tokens_pred_labels)
        # prompt tuning
        mask_loss_and_target = self.mask_head.loss_and_target(
            batch_img_metas = batch_img_metas,
            mask_result=mask_results,
            rois = pos_rois,
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)
        mask_results.update(loss_mask=mask_loss_and_target)
        return mask_results

    def _mask_forward(self, x , rois,batch_img_meta,gt_target_box=None) -> dict:
        assert rois is not None

        num_forward = 40
        part_roises = rois[:,1:].split(num_forward)
        gt_target_boxes = gt_target_box.split(num_forward)
        low_res_masks = []
        mask_tokens_preds = []
        low_res_mask_labels = []
        mask_tokens_pred_labels = []
        for part_rois,target_boxes in zip(part_roises,gt_target_boxes):
            sparse_embeddings , dense_embeddings = self.prompt_encoder(
                    points = None,
                    boxes = part_rois,
                    masks = None,
            )
            low_res_mask,  mask_tokens_pred = self.mask_head(
                image_embeddings=x.unsqueeze(0),#  [1,256,64,64]
                image_pe=self.prompt_encoder.get_dense_pe(), # [1,256,64,64]
                sparse_prompt_embeddings=sparse_embeddings, # [1,2,256]
                dense_prompt_embeddings=dense_embeddings,# [1,256,64,64]
                multimask_output=False,# true
            )
            target_sparse_embeddings , target_dense_embeddings = self.prompt_encoder(
                points = None,
                boxes = target_boxes,
                masks = None,
            )
            low_res_mask_label,  mask_tokens_pred_label= self.mask_label_head(
                image_embeddings=x.unsqueeze(0),#  [1,256,64,64]
                image_pe=self.prompt_encoder.get_dense_pe(), # [1,256,64,64]
                sparse_prompt_embeddings=target_sparse_embeddings, # [1,2,256]
                dense_prompt_embeddings=target_dense_embeddings,# [1,256,64,64]
                multimask_output=False,# true
            )
            low_res_masks.append(low_res_mask)
            mask_tokens_preds.append(mask_tokens_pred)
            low_res_mask_labels.append(low_res_mask_label)
            mask_tokens_pred_labels.append(mask_tokens_pred_label)
           
            
        low_res_masks = torch.cat(low_res_masks)
        mask_tokens_preds = torch.cat(mask_tokens_preds)
        low_res_mask_labels = torch.cat(low_res_mask_labels)
        mask_tokens_pred_labels = torch.cat(mask_tokens_pred_labels)
        
        return low_res_masks,mask_tokens_preds,low_res_mask_labels,mask_tokens_pred_labels
    
    def _mask_pred(self, x , rois) -> dict:
        assert rois is not None

        num_forward = 50
        part_roises = rois[:,1:].split(num_forward)
        low_res_masks = []
        for part_rois in part_roises:
            sparse_embeddings , dense_embeddings = self.prompt_encoder(
                    points = None,
                    boxes = part_rois,
                    masks = None,
            )
            low_res_mask  = self.mask_head.predict(
                image_embeddings=x,#  [bs,256,64,64]
                image_pe=self.prompt_encoder.get_dense_pe(), # [bs,256,64,64]
                sparse_prompt_embeddings=sparse_embeddings, # [bs,2,256]
                dense_prompt_embeddings=dense_embeddings,# [bs,256,64,64]
            )
            low_res_masks.append(low_res_mask)
        low_res_masks = torch.cat(low_res_masks)
        return low_res_masks 
    
    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results = self._bbox_forward(x[:4], rois)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list

    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list
        mask_preds  = self._mask_pred(x[self.ori_level], mask_rois)
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
        # TODO: Handle the case where rescale is false
        # post process (same with SAM)
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
       
        return results_list
    

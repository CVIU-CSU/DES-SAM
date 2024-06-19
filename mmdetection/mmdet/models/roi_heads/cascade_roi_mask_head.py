# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import ModuleList
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.test_time_augs import merge_aug_masks
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptMultiConfig)
from ..utils.misc import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead
from mmdet.models.utils.misc import multi_apply

ori_level = 4
@MODELS.register_module()
class CascadeMASKRoIHead(BaseRoIHead):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages: int,
                 stage_loss_weights: Union[List[float], Tuple[float]],
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 prompt_encoder: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 mask_label_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 ori_level: int = 0) -> None:
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super().__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=None,
            mask_head=None,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
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

    def init_bbox_head(self, bbox_roi_extractor: MultiConfig,
                       bbox_head: MultiConfig) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (:obj:`ConfigDict`, dict or list):
                Config of box roi extractor.
            bbox_head (:obj:`ConfigDict`, dict or list): Config
                of box in box head.
        """
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(MODELS.build(roi_extractor))
            self.bbox_head.append(MODELS.build(head))

    def init_mask_head(self,prompt_encoder: ConfigType,mask_head: ConfigType,mask_label_head: ConfigType) -> None:
        self.prompt_encoder = MODELS.build(prompt_encoder)
        self.mask_head = MODELS.build(mask_head)
        self.mask_label_head = MODELS.build(mask_label_head)


    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    TASK_UTILS.build(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    TASK_UTILS.build(
                        rcnn_train_cfg.sampler,
                        default_args=dict(context=self)))

    def _bbox_forward(self, stage: int, x: Tuple[Tensor],
                      rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def bbox_loss(self, stage: int, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Run forward function and calculate loss for box head in training.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
                - `rois` (Tensor): RoIs with the shape (n, 5) where the first
                  column indicates batch id of each RoI.
                - `bbox_targets` (tuple):  Ground truth for proposals in a
                  single image. Containing the following list of Tensors:
                  (labels, label_weights, bbox_targets, bbox_weights)
        """
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_results.update(rois=rois)

        bbox_loss_and_target = bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg[stage])
        bbox_results.update(bbox_loss_and_target)

        return bbox_results

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  batch_gt_instances: InstanceList,
                  batch_img_metas: list) -> dict:
        # 预测的box获取到mask √
        # TODO 对mask进行boxinst的损失 不再利用sam进行伪标签的生成  loss的实现
        # 对sam的微调分为两部分：prompt tunning（kl) mask decoder (kl)
        # during trainning rois can be proposals
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results]) # 256,5
        # posroi box
        gt_target_box = []
        for bgi,res in zip(batch_gt_instances,sampling_results):
            gt_target_box.append(bgi.bboxes[res.pos_assigned_gt_inds])
        gt_target_box = torch.cat(gt_target_box)
        #pos_rois[:, 1:] = get_box_tensor(self.bbox_head.bbox_coder.decode(pos_rois[:, 1:], bbox_pred[pos_inds]))
        pos_rois = pos_rois.split([len(res.pos_priors) for res in sampling_results],0)
        gt_target_box = gt_target_box.split([len(res.pos_priors) for res in sampling_results],0)
        low_res_masks,mask_tokens_preds,low_res_mask_labels,mask_tokens_pred_labels,iou_preds= multi_apply(self._mask_forward,x[self.ori_level],pos_rois,gt_target_box)
        mask_results = dict(mask_preds=low_res_masks,mask_tokens_preds=mask_tokens_preds,gt_mask=low_res_mask_labels,gt_token=mask_tokens_pred_labels,iou_preds = iou_preds) # 得到正的pos的框 xyxy
        # 计算loss 做预测进行微调，调整个maskdecoder 不用伪标签的方式
        mask_loss_and_target = self.mask_head.loss_and_target(
            batch_img_metas = batch_img_metas,
            mask_result=mask_results,
            rois = pos_rois,#rois = torch.cat(pos_rois),
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)
        mask_results.update(loss_mask=mask_loss_and_target)
        #mask_results['loss_mask'].update(loss_kl)
        return mask_results

    def _mask_forward(self, x , rois,gt_target_box=None) -> dict:
        assert rois is not None

        num_forward = 40
        part_roises = rois[:,1:].split(num_forward)
        gt_target_boxes = gt_target_box.split(num_forward)
        low_res_masks = []
        mask_tokens_preds = []
        low_res_mask_labels = []
        mask_tokens_pred_labels = []
        iou_preds = []
        for part_rois,target_boxes in zip(part_roises,gt_target_boxes):
            sparse_embeddings , dense_embeddings = self.prompt_encoder(
                    points = None,
                    boxes = part_rois,
                    masks = None,
            )
            low_res_mask,  mask_tokens_pred ,iou_pred= self.mask_head(
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
            iou_preds.append(iou_pred)
            
        low_res_masks = torch.cat(low_res_masks)
        mask_tokens_preds = torch.cat(mask_tokens_preds)
        low_res_mask_labels = torch.cat(low_res_mask_labels)
        mask_tokens_pred_labels = torch.cat(mask_tokens_pred_labels)
        iou_preds = torch.cat(iou_preds)


        #mask_results = dict(mask_preds=low_res_masks,mask_tokens_pred=mask_tokens_pred,mask_tokens_frozen=mask_tokens_frozen) # 得到正的pos的框 xyxy
        # 计算loss
        
        return low_res_masks,mask_tokens_preds,low_res_mask_labels,mask_tokens_pred_labels,iou_preds

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
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
        # TODO: May add a new function in baseroihead
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        num_imgs = len(batch_data_samples)
        losses = dict()
        results_list = rpn_results_list
        for stage in range(self.num_stages):
            self.current_stage = stage

            stage_loss_weight = self.stage_loss_weights[stage]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[stage]
                bbox_sampler = self.bbox_sampler[stage]

                for i in range(num_imgs):
                    results = results_list[i]
                    # rename rpn_results.bboxes to rpn_results.priors
                    results.priors = results.pop('bboxes')

                    assign_result = bbox_assigner.assign(
                        results, batch_gt_instances[i],
                        batch_gt_instances_ignore[i])

                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        results,
                        batch_gt_instances[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x[:ori_level]])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self.bbox_loss(stage, x[:ori_level], sampling_results)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            # mask head forward and loss
            if stage == self.num_stages - 1: # 只在最后一层训练一个mask head
                if self.with_mask:
                    mask_results = self.mask_loss(x, sampling_results,
                                                batch_gt_instances,batch_img_metas)
                    '''for name, value in mask_results['loss_mask'].items():
                        losses[f's{stage}.{name}'] = (
                            value * stage_loss_weight if 'loss' in name else value)'''
                    losses.update(mask_results['loss_mask'])

            # refine bboxes
            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                with torch.no_grad():
                    results_list = bbox_head.refine_bboxes(
                        sampling_results, bbox_results, batch_img_metas)
                    # Empty proposal
                    if results_list is None:
                        break
        return losses

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False,
                     **kwargs) -> InstanceList:
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
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head[-1].predict_box_type,
                num_classes=self.bbox_head[-1].num_classes,
                score_per_cls=rcnn_test_cfg is None)

        rois, cls_scores, bbox_preds = self._refine_roi(
            x=x[:ori_level],
            rois=rois,
            batch_img_metas=batch_img_metas,
            num_proposals_per_img=num_proposals_per_img,
            **kwargs)

        results_list = self.bbox_head[-1].predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            rcnn_test_cfg=rcnn_test_cfg)
        return results_list

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
        # 分组 一张一张的推理的时候
        #mask_rois = mask_rois.split([len(res.bboxes) for res in results_list],0)
        mask_preds,iou_pred  = self._mask_pred(x[self.ori_level], mask_rois)
        #import numpy as np
        #np.save('mask_preds',mask_preds.cpu().numpy())
        #mask_results = dict(mask_preds=mask_preds, mask_iou=mask_iou)
        #mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        #mask_preds = torch.cat(mask_preds)
        #mask_iou = torch.cat(mask_iou)
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
        #mask_iou = mask_iou.split(num_mask_rois_per_img, 0)

        # TODO: Handle the case where rescale is false
        # 后处理
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        #results_list = self.mask_head.predict_by_iou(
        #    mask_iou_preds=mask_iou, results_list=results_list)
        return results_list
    
    def _mask_pred(self, x , rois) -> dict:
        assert rois is not None

        num_forward = 50
        part_roises = rois[:,1:].split(num_forward)
        low_res_masks = []
        iou_preds = []
        for part_rois in part_roises:
            sparse_embeddings , dense_embeddings = self.prompt_encoder(
                    points = None,
                    boxes = part_rois,
                    masks = None,
            )
            low_res_mask,iou_pred  = self.mask_head.predict(
                image_embeddings=x,#  [1,256,64,64]
                image_pe=self.prompt_encoder.get_dense_pe(), # [1,256,64,64]
                sparse_prompt_embeddings=sparse_embeddings, # [1,2,256]
                dense_prompt_embeddings=dense_embeddings,# [1,256,64,64]
                #multimask_output=False,# true
            )
            low_res_masks.append(low_res_mask)
            iou_preds.append(iou_pred)
        low_res_masks = torch.cat(low_res_masks)
        iou_preds = torch.cat(iou_preds)
        #mask_results = dict(mask_preds=low_res_masks,mask_tokens_pred=mask_tokens_pred,mask_tokens_frozen=mask_tokens_frozen) # 得到正的pos的框 xyxy
        # 计算loss
        
        return low_res_masks,iou_preds
    
    
    def _refine_roi(self, x: Tuple[Tensor], rois: Tensor,
                    batch_img_metas: List[dict],
                    num_proposals_per_img: Sequence[int], **kwargs) -> tuple:
        """Multi-stage refinement of RoI.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): shape (n, 5), [batch_ind, x1, y1, x2, y2]
            batch_img_metas (list[dict]): List of image information.
            num_proposals_per_img (sequence[int]): number of proposals
                in each image.

        Returns:
            tuple:

               - rois (Tensor): Refined RoI.
               - cls_scores (list[Tensor]): Average predicted
                   cls score per image.
               - bbox_preds (list[Tensor]): Bbox branch predictions
                   for the last stage of per image.
        """
        # "ms" in variable names means multi-stage
        ms_scores = []
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(
                stage=stage, x=x, rois=rois, **kwargs)

            # split batch bbox prediction back to each image
            cls_scores = bbox_results['cls_score']
            bbox_preds = bbox_results['bbox_pred']

            rois = rois.split(num_proposals_per_img, 0)
            cls_scores = cls_scores.split(num_proposals_per_img, 0)
            ms_scores.append(cls_scores)

            # some detector with_reg is False, bbox_preds will be None
            if bbox_preds is not None:
                # TODO move this to a sabl_roi_head
                # the bbox prediction of some detectors like SABL is not Tensor
                if isinstance(bbox_preds, torch.Tensor):
                    bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
                else:
                    bbox_preds = self.bbox_head[stage].bbox_pred_split(
                        bbox_preds, num_proposals_per_img)
            else:
                bbox_preds = (None, ) * len(batch_img_metas)

            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                if bbox_head.custom_activation:
                    cls_scores = [
                        bbox_head.loss_cls.get_activation(s)
                        for s in cls_scores
                    ]
                refine_rois_list = []
                for i in range(len(batch_img_metas)):
                    if rois[i].shape[0] > 0:
                        bbox_label = cls_scores[i][:, :-1].argmax(dim=1)
                        # Refactor `bbox_head.regress_by_class` to only accept
                        # box tensor without img_idx concatenated.
                        refined_bboxes = bbox_head.regress_by_class(
                            rois[i][:, 1:], bbox_label, bbox_preds[i],
                            batch_img_metas[i])
                        refined_bboxes = get_box_tensor(refined_bboxes)
                        refined_rois = torch.cat(
                            [rois[i][:, [0]], refined_bboxes], dim=1)
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_scores = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(len(batch_img_metas))
        ]
        return rois, cls_scores, bbox_preds

    def forward(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
                batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        results = ()
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            rois, cls_scores, bbox_preds = self._refine_roi(
                x, rois, batch_img_metas, num_proposals_per_img)
            results = results + (cls_scores, bbox_preds)
        # mask head
        if self.with_mask:
            aug_masks = []
            rois = torch.cat(rois)
            for stage in range(self.num_stages):
                mask_results = self._mask_forward(stage, x, rois)
                mask_preds = mask_results['mask_preds']
                mask_preds = mask_preds.split(num_proposals_per_img, 0)
                aug_masks.append([m.sigmoid().detach() for m in mask_preds])

            merged_masks = []
            for i in range(len(batch_img_metas)):
                aug_mask = [mask[i] for mask in aug_masks]
                merged_mask = merge_aug_masks(aug_mask, batch_img_metas[i])
                merged_masks.append(merged_mask)
            results = results + (merged_masks, )
        return results

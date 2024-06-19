# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
from mmdet.structures import SampleList
from segment_anything import sam_model_registry, SamPredictor
import cv2


@MODELS.register_module()
class FasterRCNN(TwoStageDetector):
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

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        
        # sam
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        #device = "cuda:0"  # or  "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=batch_inputs.device)
        predictor = SamPredictor(sam)

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)
        #ori_level = len(x) - 1
        ori_level = len(x)
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
        #self.mask_save(results_list,batch_data_samples)
        results_list = self.mask_pred(
            predictor = predictor,
            results_list=results_list,
            batch_data_samples=batch_data_samples)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
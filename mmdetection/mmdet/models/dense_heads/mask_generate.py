import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from typing import List, Tuple, Type
from mmengine.model import BaseModule
from mmdet.models.dense_heads.sam import FrozenSAM

from mmdet.registry import MODELS
from segment_anything.modeling import TwoWayTransformer
from mmdet.structures.mask import mask_target
from mmdet.models.task_modules.samplers import SamplingResult
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor
from mmdet.utils import InstanceList
from mmdet.utils.typing_utils import ConfigType
from mmdet.models.utils import empty_instances
from mmengine.structures import InstanceData
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from mmcv.ops.roi_align import roi_align as ROIAlign
from mmdet.models.utils import unfold_wo_center


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss

def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
    
@MODELS.register_module()
class MaskDecoder(BaseModule):
    mask_threshold: float = 0.0

    def __init__(
        self,
        *,
        transformer_dim=256,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        class_agnostic: int = False,
        loss_mask: ConfigType = dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
        clipped_size: tuple = (72,72),
        pairwise_weights: dict = { 
            "loss_local_pairwise": 1.0,
            "loss_global_pairwise": 0.001,},
        mask_bce_weights: float = 0.5,
        mask_dice_weights: float = 0.01,        
    ) -> None:
        
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            )

        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens =  num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim) # [4,256]

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.class_agnostic = class_agnostic
        self.loss_mask = MODELS.build(loss_mask)
        # learned_prompt
        self.learned_prompt = nn.Embedding(1, transformer_dim)
        self.clipped_size = clipped_size
        # pair_wise loss weights
        self.loss_weights = pairwise_weights
        self.mask_bce_loss = mask_bce_weights
        self.mask_dice_loss = mask_dice_weights
        self.init_weights()
   
    def init_weights(self):
        for params in self.parameters():
            params.requires_grad=False
        self.iou_token.weight.requires_grad = False
        self.mask_tokens.weight.requires_grad = False
        self.learned_prompt.weight.requires_grad = True

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        masks, mask_tokens_pred  = self.predict_masks(
            image_embeddings=image_embeddings,# [1,256,64,64]
            image_pe=image_pe,# [1,256,64,64]
            sparse_prompt_embeddings=sparse_prompt_embeddings,# [1,2,256]
            dense_prompt_embeddings=dense_prompt_embeddings,# [1,256,64,64]
        )

        masks = masks[:, 0, :, :]

        # Prepare output
        return masks, mask_tokens_pred 

    def predict(self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens cat [1,256] [4,256] -> [5,256] 
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight,self.learned_prompt.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) 
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) 

        # Expand per-image data in batch direction to be per-mask image_embeddings 
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) 
        b, c, h, w = src.shape

        # Run the transformer 
        # src:[bs,256,64,64]+dense_prompt_embeddings pos_src:[bs,256,64,64] tokens:output_tokens+sparse_prompt_embeddings
        hs, src = self.transformer(src, pos_src, tokens) 
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        #iou_pred = self.iou_prediction_head(iou_token_out)
        return masks[:, 0, :, :] #,iou_pred
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens cat [1,256] [4,256] -> [5,256] 
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight,self.learned_prompt.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) 
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) 

        # Expand per-image data in batch direction to be per-mask image_embeddings 
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings 
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) 
        b, c, h, w = src.shape 

        # Run the transformer src:[1,256,64,64]+dense_prompt_embeddings pos_src:[1,256,64,64] tokens:output_tokens+sparse_prompt_embeddings
        hs, src = self.transformer(src, pos_src, tokens) 
        iou_token_out = hs[:, 0, :] 
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :] 

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w) 
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return masks,mask_tokens_out
    
    def get_targets(self, sampling_results: List[SamplingResult],
                    batch_gt_instances: InstanceList,
                    rcnn_train_cfg: ConfigDict) -> Tensor:
        
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks = [res.masks for res in batch_gt_instances]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def loss_and_target(self, batch_img_metas,
                        mask_result:dict,
                        rois:Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        
        # ROIALIGN
        pred_masks = []
        for i in range(len(mask_result['mask_preds'])):
            im_mask = postprocess_trainning_masks(
                mask_result['mask_preds'][i], batch_img_metas[i]['ori_shape'][:2])
            pred_mask = torch.cat([ROIAlign(im_mask[j][None,None],rois[i][j][None],self.clipped_size,1.0, 0) for j in range(rois[i].shape[0])])
            pred_masks.append(pred_mask)
        # mask loss
        mask_bce_loss = 0
        mask_dice_loss = 0   
        for batch_idx in range(len(mask_result['mask_preds'])):
            gt_mask = (mask_result['gt_mask'][batch_idx] > 0.0).float()
            pred_mask = mask_result['mask_preds'][batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )  
        mask_preds = torch.cat(pred_masks).sigmoid()
        num_masks = mask_preds.shape[0]
        # pairwise loss  
        # get ready for target
        gt_imgs_sim = []
        gt_imgs = []
        gt_boxmasks = []
        gt_bboxes = []
        for i in range(len(sampling_results)):
            gt_imgs_sim.append(batch_gt_instances[i].gt_imgs_sim[sampling_results[i].pos_assigned_gt_inds])
            gt_imgs.append(batch_gt_instances[i].gt_imgs[sampling_results[i].pos_assigned_gt_inds])
            gt_boxmasks.append(batch_gt_instances[i].gt_boxmasks[sampling_results[i].pos_assigned_gt_inds])
            gt_bboxes.append(batch_gt_instances[i].bboxes[sampling_results[i].pos_assigned_gt_inds]) 
        gt_imgs_sim = torch.cat(gt_imgs_sim)
        gt_imgs = torch.cat(gt_imgs)
        gt_boxmasks = torch.cat(gt_boxmasks)
        gt_bboxes = torch.cat(gt_bboxes)
        targets = {"mask": gt_boxmasks, "imgs_sim": gt_imgs_sim, "imgs": gt_imgs, "gt_boxes": gt_bboxes}
        kwargs = {'pred_polys': ""}
        losses = {}
        loss_map = {
            "global_pairwise":self.loss_global_pairwise,
            "local_pairwise":self.loss_local_pairwise}
        box_snake_losses = ['local_pairwise','global_pairwise']
        self.loss_map = {k: loss_map[k] for k in box_snake_losses}
        # set pairwise window size
        self.local_pairwise_kernel_size = 3
        self.local_pairwise_dilation = 1
        self.local_pairwise_color_threshold = 0.1
        
        for loss in box_snake_losses:
            losses.update(self.get_loss(loss, mask_preds, targets, num_masks, **kwargs))

        # total loss
        losses.update({"mask_bce_loss":mask_bce_loss*self.mask_bce_loss,"mask_dice_loss":mask_dice_loss*self.mask_dice_loss})
        
        return losses
    
    def get_loss(self, loss, pred_masks, target_masks, num_masks, **kwargs):
        assert loss in self.loss_map, f"do you really want to compute {loss} loss?"
        return self.loss_map[loss](pred_masks, target_masks, num_masks, **kwargs)
    
    # change from boxsnake   
    # calculate local_pairwise 
    def loss_local_pairwise(self, pred_masks, targets, num_masks, **kwargs):
        target_masks, imgs_sim = targets["mask"], targets["imgs_sim"]
        # fg_prob = torch.sigmoid(pred_masks) if self.is_logits else pred_masks
        fg_prob = pred_masks # [N_boxes,1,72,72]
        fg_prob_unfold = unfold_wo_center(
            fg_prob, kernel_size=self.local_pairwise_kernel_size,
            dilation=self.local_pairwise_dilation)
        pairwise_term = torch.abs(fg_prob[:, :, None] - fg_prob_unfold)[:, 0]
        weights = imgs_sim * target_masks.float() # limit to the box
        loss_local_pairwise = (weights * pairwise_term).sum() / weights.sum().clamp(min=1.0)
        # TODO: which one ?
        # loss_local_pairwise = (weights * pairwise_term).flatten(1).sum(-1) / weights.flatten(1).sum(-1).clamp(min=1.0)
        # loss_local_pairwise = loss_local_pairwise.sum() / num_masks
        loss = {"loss_local_pairwise": loss_local_pairwise * self.loss_weights.get("loss_local_pairwise", 0.)}
        # TODO: add different pairwise format
        del target_masks
        del imgs_sim
        return loss
    
    # change from boxsnake
    # calculate global_pairwise
    def loss_global_pairwise(self, pred_masks, targets, num_masks, **kwargs):
        """
        ref: https://www.math.ucla.edu/~lvese/PAPERS/JVCIR2000.pdf and boxlevelset method
        pred_masks: shpae=(N, 1, H, W)
        limit the mask into the box mask, imgs is cropped, it does't need mask gated
        """
        target_masks, imgs = targets["mask"], targets["imgs"] # shape=(N, 1, H, W), (N, 3, H, W)
        # prepare pred_masks
        pred_masks_back = 1.0 - pred_masks
        C_, H_, W_ = imgs.shape[1:]
        # imgs_wbox = imgs * target_masks # TODO, dose this matter in the cropped way?  
        level_set_energy = get_region_level_energy(imgs, pred_masks, C_) + \
                           get_region_level_energy(imgs, pred_masks_back, C_)

        pixel_num = float(H_ * W_)

        level_set_losses = torch.mean((level_set_energy) / pixel_num) # HW weights
        losses = {"loss_global_pairwise": level_set_losses * self.loss_weights.get('loss_global_pairwise', 0.)} # instances weights

        del target_masks
        del imgs
        return losses

    def predict_by_feat(self,
                        mask_preds: Tuple[Tensor],
                        results_list: List[InstanceData],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: ConfigDict,
                        rescale: bool = False,
                        activate_map: bool = False,) -> InstanceList:
        
        assert len(mask_preds) == len(results_list) == len(batch_img_metas)

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = results_list[img_id]
            bboxes = results.bboxes
            if bboxes.shape[0] == 0:
                results_list[img_id] = empty_instances(
                    [img_meta],
                    bboxes.device,
                    task_type='mask',
                    instance_results=[results],
                    mask_thr_binary=rcnn_test_cfg.mask_thr_binary)[0]
            else:
                im_mask = self._predict_by_feat_single(
                    bboxes = bboxes,
                    mask_preds=mask_preds[img_id],
                    img_meta=img_meta,
                    rescale = rescale,
                    mask_threshold=rcnn_test_cfg.mask_thr_binary)
                results.masks = im_mask
        return results_list

    def _predict_by_feat_single(self,
                                bboxes,
                                mask_preds: Tensor,
                                img_meta: dict,
                                rescale,
                                mask_threshold=None) -> Tensor:
        # Upscale the masks to the original image resolution
        # image_shape batch_input_shape 
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        img_h, img_w = img_meta['ori_shape'][:2]

        if rescale:  # in-placed rescale the bboxes
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)

        im_mask = postprocess_masks(
            mask_preds, img_meta['ori_shape'][:2]
        )
        #im_mask = im_mask.sigmoid()
        if not mask_threshold:
            im_mask = im_mask > self.mask_threshold
        else:
            im_mask = im_mask > mask_threshold
        
        return im_mask

def postprocess_trainning_masks(
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        #np.save('masks',masks.cpu().numpy())
        masks = masks.unsqueeze(1)
        masks = F.interpolate(
            masks,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        input_size = get_preprocess_shape(original_size[0],original_size[1])
        masks[..., input_size[0]:,input_size[1]:] = 0
        masks = masks.squeeze(1)
        return masks
    
def postprocess_masks(
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        #np.save('masks',masks.cpu().numpy())
        masks = masks.unsqueeze(1)
        masks = F.interpolate(
            masks,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        input_size = get_preprocess_shape(original_size[0],original_size[1])
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        masks = masks.squeeze(1)
        return masks

def get_preprocess_shape(oldh: int, oldw: int, long_side_length=1024) -> Tuple[int, int]:
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def get_region_level_energy(imgs, masks, num_channel):
    avg_sim = torch.sum(imgs * masks, dim=(2, 3), keepdim=True) / torch.sum(masks, dim=(2, 3), keepdim=True).clamp(min=1e-5)
    region_level = torch.pow(imgs - avg_sim, 2) * masks  
    return torch.sum(region_level, dim=(1, 2, 3)) / num_channel


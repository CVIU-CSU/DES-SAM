import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from typing import List, Tuple, Type
from mmengine.model import BaseModule

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

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
#  determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

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
class FrozenSAM(BaseModule):
    mask_threshold: float = 0.0
    sam_checkpoint = '/root/userfolder/hln/sam_cls/sam_cls/weights/sam_vit_b_01ec64.pth'
    model_type = "vit_b"

    def __init__(
        self,
        *,
        transformer_dim=256,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        class_agnostic: int = False,
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
        #[1,256]
        self.iou_token = nn.Embedding(1, transformer_dim)
        #self.num_mask_tokens = num_multimask_outputs + 1
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
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            param.requires_grad = False
        self.iou_token.weight.requires_grad = False
        self.mask_tokens.weight.requires_grad = False

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        masks,mask_tokens_out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        # Prepare output
        masks = masks[:, 0, :, :]
        return masks,mask_tokens_out

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens cat [1,256] [4,256] -> [5,256]
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) 
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) 

        # Expand per-image data in batch direction to be per-mask image_embeddings
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings 
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) 
        b, c, h, w = src.shape 

        hs, src = self.transformer(src, pos_src, tokens) 
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :] #一个点预测4个mask 
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w) 
        # 预测头
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        #iou_pred = self.iou_prediction_head(iou_token_out)

        return masks,mask_tokens_out
    

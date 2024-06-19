import math
import torch.nn as nn
import torch
from mmdet.registry import MODELS
import torch.nn.functional as F

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
        
class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]

@MODELS.register_module()
class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        out_channels,
        scale_factors,
        #in_feature = None,
        top_block=None,
        square_pad=0,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 256,
        with_mask: bool=False,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
            patch_size: 16 
            in_chans:
        """
        super(SimpleFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors
        # C H W 256 64 64
        input_shape = [in_chans,img_size//patch_size,img_size//patch_size]
        
        strides = [int(patch_size / scale) for scale in scale_factors] 
        #_assert_strides_are_log2_contiguous(strides)
        # embed dim
        dim = input_shape[0]
        self.stages = nn.ModuleList()
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm2d(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
            
            if scale != 0.0: 
                layers.extend(
                    [
                        nn.Conv2d(
                            out_dim,
                            out_channels,
                            kernel_size=1,
                            bias=False
                        ),
                        LayerNorm2d(out_channels),
                        nn.Conv2d(
                            out_channels,
                            out_channels,
                            padding=1,
                            kernel_size=3,
                            bias=False
                        ),
                        LayerNorm2d(out_channels),
                    ]
                )
                layers = nn.Sequential(*layers)
            else:
                layers = nn.Identity()

            stage = int(math.log2(strides[idx]))
            #self.add_sublayer(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.layers_ori = nn.Identity()
        #self.stages.append(layers_ori)

        #self.in_feature = in_feature
        #self.top_block = LastLevelMaxPool()
        self.top_block = None
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        num_levels = 1
        if self.top_block is not None:
            for s in range(stage, stage + num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad


    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x: torch.Tensor)  -> tuple:
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
         #这里backbone部分，也就是ViT
        results = []

        for stage in self.stages:
            output_features = stage(x)
            results.append(output_features) # 这里就是反卷积部分
        in_feature = "p5"
        if self.top_block is not None:   # 这里就是论文中池化部分
            top_block_in_feature = results[self._out_features.index(in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        results.append(self.layers_ori(x))
        return tuple(results)


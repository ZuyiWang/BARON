# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList, BaseModule
from torch import Tensor, nn
from mmdet.utils import ConfigType, OptConfigType
from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock

from .detr_layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DetrTransformerEncoder, DetrTransformerEncoderLayer)

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

###### used for bbox_head, MultiheadAttention ######
class ImageTextCommunicationTransformerEncoder(DetrTransformerEncoder):
    """Transformer encoder of ITC."""

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            ITCTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims

    def forward(self, image_query: Tensor, text_query: Tensor, 
                image_query_pos: Tensor = None,
                image_key_padding_mask: Tensor = None, 
                text_query_pos: Tensor = None,
                text_key_padding_mask: Tensor = None,
                text_self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        for layer in self.layers:
            image_query = layer(
                image_query=image_query,
                image_query_pos=image_query_pos,
                image_key_padding_mask=image_key_padding_mask,
                text_query=text_query,
                text_query_pos=text_query_pos,
                text_key_padding_mask=text_key_padding_mask,
                text_self_attn_mask=text_self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                **kwargs)
        return image_query

class ITCTransformerEncoderLayer(BaseModule):
    """Encoder layer of Deformable DETR."""

    def __init__(self,
                 image_self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 text_self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.image_self_attn_cfg = image_self_attn_cfg
        self.text_self_attn_cfg = text_self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg
        if 'batch_first' not in self.image_self_attn_cfg:
            self.image_self_attn_cfg['batch_first'] = True
        else:
            assert self.image_self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'
        
        if 'batch_first' not in self.text_self_attn_cfg:
            self.text_self_attn_cfg['batch_first'] = True
        else:
            assert self.text_self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'
        
        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()


    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_image_attn = MultiheadAttention(**self.image_self_attn_cfg)
        self.self_text_attn = MultiheadAttention(**self.text_self_attn_cfg)
        self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_image_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(4)
        ]
        self.norms = ModuleList(norms_list)
    
    def forward(self, text_query: Tensor, text_query_pos: Tensor, 
                image_query: Tensor, image_query_pos: Tensor,
                image_key_padding_mask: Tensor = None, 
                text_key_padding_mask: Tensor = None, 
                text_self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,**kwargs) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        image_query = self.self_image_attn(
            query=image_query,
            key=image_query,
            value=image_query,
            query_pos=image_query_pos,
            key_pos=image_query_pos,
            key_padding_mask=image_key_padding_mask,
            **kwargs)
        image_query = self.norms[0](image_query)

        text_query = self.self_text_attn(
            query=text_query,
            key=text_query,
            value=text_query,
            query_pos=text_query_pos,
            key_pos=text_query_pos,
            attn_mask=text_self_attn_mask,
            **kwargs)
        text_query = self.norms[1](text_query)

        image_query = self.cross_attn(
            query=image_query,
            key=text_query,
            value=text_query,
            query_pos=image_query_pos,
            key_pos=text_query_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=text_key_padding_mask,
            **kwargs)
        image_query = self.norms[2](image_query)
        image_query = self.ffn(image_query)
        image_query = self.norms[3](image_query)

        return image_query


# ###### used for RPN, MultiScaleDeformableAttention ######
# class ImageTextCommunicationTransformerEncoder(DetrTransformerEncoder):
#     """Transformer encoder of ITC."""

#     def _init_layers(self) -> None:
#         """Initialize encoder layers."""
#         self.layers = ModuleList([
#             ITCTransformerEncoderLayer(**self.layer_cfg)
#             for _ in range(self.num_layers)
#         ])

#         if self.num_cp > 0:
#             if checkpoint_wrapper is None:
#                 raise NotImplementedError(
#                     'If you want to reduce GPU memory usage, \
#                     please install fairscale by executing the \
#                     following command: pip install fairscale.')
#             for i in range(self.num_cp):
#                 self.layers[i] = checkpoint_wrapper(self.layers[i])

#         self.embed_dims = self.layers[0].embed_dims

#     def forward(self, image_query: Tensor, image_query_pos: Tensor,
#                 image_key_padding_mask: Tensor, spatial_shapes: Tensor,
#                 level_start_index: Tensor, valid_ratios: Tensor,
#                 text_query: Tensor, text_query_pos: Tensor,
#                 text_key_padding_mask: Tensor,
#                 text_self_attn_mask: Tensor,
#                 cross_attn_mask: Tensor,
#                 **kwargs) -> Tensor:
#         """Forward function of Transformer encoder.

#         Args:
#             query (Tensor): The input query, has shape (bs, num_queries, dim).
#             query_pos (Tensor): The positional encoding for query, has shape
#                 (bs, num_queries, dim).
#             key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
#                 input. ByteTensor, has shape (bs, num_queries).
#             spatial_shapes (Tensor): Spatial shapes of features in all levels,
#                 has shape (num_levels, 2), last dimension represents (h, w).
#             level_start_index (Tensor): The start index of each level.
#                 A tensor has shape (num_levels, ) and can be represented
#                 as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
#             valid_ratios (Tensor): The ratios of the valid width and the valid
#                 height relative to the width and the height of features in all
#                 levels, has shape (bs, num_levels, 2).

#         Returns:
#             Tensor: Output queries of Transformer encoder, which is also
#             called 'encoder output embeddings' or 'memory', has shape
#             (bs, num_queries, dim)
#         """
#         reference_points = self.get_encoder_reference_points(
#             spatial_shapes, valid_ratios, device=image_query.device)
#         for layer in self.layers:
#             image_query = layer(
#                 image_query=image_query,
#                 image_query_pos=image_query_pos,
#                 image_key_padding_mask=image_key_padding_mask,
#                 text_query=text_query,
#                 text_query_pos=text_query_pos,
#                 text_key_padding_mask=text_key_padding_mask,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 valid_ratios=valid_ratios,
#                 reference_points=reference_points,
#                 text_self_attn_mask=text_self_attn_mask,
#                 cross_attn_mask=cross_attn_mask,
#                 **kwargs)
#         return image_query

#     @staticmethod
#     def get_encoder_reference_points(
#             spatial_shapes: Tensor, valid_ratios: Tensor,
#             device: Union[torch.device, str]) -> Tensor:
#         """Get the reference points used in encoder.

#         Args:
#             spatial_shapes (Tensor): Spatial shapes of features in all levels,
#                 has shape (num_levels, 2), last dimension represents (h, w).
#             valid_ratios (Tensor): The ratios of the valid width and the valid
#                 height relative to the width and the height of features in all
#                 levels, has shape (bs, num_levels, 2).
#             device (obj:`device` or str): The device acquired by the
#                 `reference_points`.

#         Returns:
#             Tensor: Reference points used in decoder, has shape (bs, length,
#             num_levels, 2).
#         """

#         reference_points_list = []
#         for lvl, (H, W) in enumerate(spatial_shapes):
#             ref_y, ref_x = torch.meshgrid(
#                 torch.linspace(
#                     0.5, H - 0.5, H, dtype=torch.float32, device=device),
#                 torch.linspace(
#                     0.5, W - 0.5, W, dtype=torch.float32, device=device))
#             ref_y = ref_y.reshape(-1)[None] / (
#                 valid_ratios[:, None, lvl, 1] * H)
#             ref_x = ref_x.reshape(-1)[None] / (
#                 valid_ratios[:, None, lvl, 0] * W)
#             ref = torch.stack((ref_x, ref_y), -1)
#             reference_points_list.append(ref)
#         reference_points = torch.cat(reference_points_list, 1)
#         # [bs, sum(hw), num_level, 2]
#         reference_points = reference_points[:, :, None] * valid_ratios[:, None]
#         return reference_points

# class ITCTransformerEncoderLayer(BaseModule):
#     """Encoder layer of Deformable DETR."""

#     def __init__(self,
#                  image_self_attn_cfg: OptConfigType = dict(
#                      embed_dims=256, num_heads=8, dropout=0.0),
#                  text_self_attn_cfg: OptConfigType = dict(
#                      embed_dims=256, num_heads=8, dropout=0.0),
#                  cross_attn_cfg: OptConfigType = dict(
#                      embed_dims=256,
#                      num_heads=8,
#                      dropout=0.0,
#                      batch_first=True),
#                  ffn_cfg: OptConfigType = dict(
#                      embed_dims=256,
#                      feedforward_channels=1024,
#                      num_fcs=2,
#                      ffn_drop=0.,
#                      act_cfg=dict(type='ReLU', inplace=True)),
#                  norm_cfg: OptConfigType = dict(type='LN'),
#                  init_cfg: OptConfigType = None) -> None:

#         super().__init__(init_cfg=init_cfg)

#         self.image_self_attn_cfg = image_self_attn_cfg
#         self.text_self_attn_cfg = text_self_attn_cfg
#         self.cross_attn_cfg = cross_attn_cfg
#         if 'batch_first' not in self.image_self_attn_cfg:
#             self.image_self_attn_cfg['batch_first'] = True
#         else:
#             assert self.image_self_attn_cfg['batch_first'] is True, 'First \
#             dimension of all DETRs in mmdet is `batch`, \
#             please set `batch_first` flag.'
        
#         if 'batch_first' not in self.text_self_attn_cfg:
#             self.text_self_attn_cfg['batch_first'] = True
#         else:
#             assert self.text_self_attn_cfg['batch_first'] is True, 'First \
#             dimension of all DETRs in mmdet is `batch`, \
#             please set `batch_first` flag.'
        
#         # if 'batch_first' not in self.cross_attn_cfg:
#         #     self.cross_attn_cfg['batch_first'] = True
#         # else:
#         #     assert self.cross_attn_cfg['batch_first'] is True, 'First \
#         #     dimension of all DETRs in mmdet is `batch`, \
#         #     please set `batch_first` flag.'

#         self.ffn_cfg = ffn_cfg
#         self.norm_cfg = norm_cfg
#         self._init_layers()


#     def _init_layers(self) -> None:
#         """Initialize self_attn, ffn, and norms."""
#         self.self_image_attn = MultiScaleDeformableAttention(**self.image_self_attn_cfg)
#         self.self_text_attn = MultiheadAttention(**self.text_self_attn_cfg)
#         # self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
#         self.cross_attn = SingleScaleBiAttentionBlock(**self.cross_attn_cfg)
#         self.embed_dims = self.self_image_attn.embed_dims
#         self.ffn = FFN(**self.ffn_cfg)
#         norms_list = [
#             build_norm_layer(self.norm_cfg, self.embed_dims)[1]
#             for _ in range(4)
#         ]
#         self.norms = ModuleList(norms_list)
    
#     def forward(self, text_query: Tensor, text_query_pos: Tensor, 
#                 image_query: Tensor, image_query_pos: Tensor,
#                 image_key_padding_mask: Tensor, 
#                 text_key_padding_mask: Tensor, 
#                 text_self_attn_mask: Tensor = None,
#                 cross_attn_mask: Tensor = None,**kwargs) -> Tensor:
#         """Forward function of an encoder layer.

#         Args:
#             query (Tensor): The input query, has shape (bs, num_queries, dim).
#             query_pos (Tensor): The positional encoding for query, with
#                 the same shape as `query`.
#             key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
#                 input. ByteTensor. has shape (bs, num_queries).
#         Returns:
#             Tensor: forwarded results, has shape (bs, num_queries, dim).
#         """
#         image_query = self.self_image_attn(
#             query=image_query,
#             key=image_query,
#             value=image_query,
#             query_pos=image_query_pos,
#             key_pos=image_query_pos,
#             key_padding_mask=image_key_padding_mask,
#             **kwargs)
#         image_query = self.norms[0](image_query)

#         text_query = self.self_text_attn(
#             query=text_query,
#             key=text_query,
#             value=text_query,
#             query_pos=text_query_pos,
#             key_pos=text_query_pos,
#             attn_mask=text_self_attn_mask,
#             **kwargs)
#         text_query = self.norms[1](text_query)

#         # image_query = self.cross_attn(
#         #     query=image_query,
#         #     key=text_query,
#         #     value=text_query,
#         #     query_pos=image_query_pos,
#         #     key_pos=text_query_pos,
#         #     attn_mask=cross_attn_mask,
#         #     key_padding_mask=text_key_padding_mask,
#         #     **kwargs)
#         image_query, output_text_query = self.cross_attn(
#                 visual_feature=image_query,
#                 lang_feature=text_query,
#                 attention_mask_v=image_key_padding_mask,
#                 attention_mask_l=None,
#                 )
#         image_query = self.norms[2](image_query)
#         image_query = self.ffn(image_query)
#         image_query = self.norms[3](image_query)

#         return image_query

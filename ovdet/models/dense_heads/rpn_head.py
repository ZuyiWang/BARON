# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.models import RPNHead
from mmdet.structures.bbox import (empty_box_as, get_box_tensor,
                                   get_box_wh, scale_boxes)

import numpy as np
from typing import List, Optional, Tuple, Union
from ovdet.utils import multi_apply, unpack_gt_instances
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList
from mmdet.models.layers import SinePositionalEncoding
from ..layers import ImageTextCommunicationTransformerEncoder
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from torch.nn.init import normal_

@MODELS.register_module()
class CustomRPNHead(RPNHead):
    # The official code of MMDet3.x in this part has bug when using AMP
    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert with_nms, '`with_nms` must be True in RPNHead'
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)

            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores.float(),
                                                results.level_ids, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
            # TODO: This would unreasonably show the 0th class label
            #  in visualization
            results.labels = results.scores.new_zeros(
                len(results), dtype=torch.long)
            del results.level_ids
        else:
            # To avoid some potential error
            results_ = InstanceData()
            results_.bboxes = empty_box_as(results.bboxes)
            results_.scores = results.scores.new_zeros(0)
            results_.labels = results.scores.new_zeros(0)
            results = results_
        return results


@MODELS.register_module()
class DetachRPNHead(CustomRPNHead):
    def _init_layers(self):
        super()._init_layers()
        self.rpn_cls = nn.Sequential(nn.Conv2d(in_channels=self.feat_channels,
                                               out_channels=self.feat_channels,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(self.feat_channels,
                                               self.num_base_priors * self.cls_out_channels,
                                               1)
                                     )

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        x = self.rpn_conv(x)
        x = F.relu(x)
        # In Baron, this is used to avoid suppression on novel categories
        rpn_cls_score = self.rpn_cls(x.detach())
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

@MODELS.register_module()
class ITCRPNHead(CustomRPNHead):
    def __init__(self,
                 cls_embeddings_path='/home/wangzy/ObjectDetectionModel/BARON/ovdet/data/metadata/coco_clip_hand_craft_attn12.npy',
                 base_cls_ids = None,
                 text_embedding=512,
                #  positional_encoding=dict(num_feats=256, normalize=True, offset=-0.5),
                #  ItcEncoder=None,
                #  num_feature_levels=5,
                 **kwargs) -> None:
        self.text_embedding = text_embedding
        # self.num_feature_levels = num_feature_levels
        # self.positional_encoding = positional_encoding
        super().__init__(**kwargs)
        cls_embeddings = torch.from_numpy(
                np.load(cls_embeddings_path)).float()
        self.register_buffer('cls_embeddings', cls_embeddings)
        self.base_cls_ids = torch.tensor(base_cls_ids, device=self.cls_embeddings.device)
        # self.ItcEncoder = ImageTextCommunicationTransformerEncoder(**ItcEncoder)
        # self.level_embed = nn.Parameter(
        #     torch.Tensor(self.num_feature_levels, self.text_embedding))
        


    def _init_layers(self):
        super()._init_layers()
        self.rpn_cls = nn.Sequential(nn.Conv2d(in_channels=self.feat_channels,
                                               out_channels=self.feat_channels,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                                     nn.ReLU(),
                                    #  nn.Conv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 1)
                                     nn.Conv2d(self.feat_channels,
                                               self.num_base_priors * self.text_embedding,
                                               1)
                                     )
        # self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
    
    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        # for p in self.ItcEncoder.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        # for m in self.modules():
        #     if isinstance(m, MultiScaleDeformableAttention):
        #         m.init_weights()
        # normal_(self.level_embed)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        x = self.rpn_conv(x)
        x = F.relu(x)
        # In Baron, this is used to avoid suppression on novel categories
        rpn_cls_score = self.rpn_cls(x.detach())
        bs, _, H, W = x.shape
        if self.training:
            text_embedding = self.cls_embeddings[self.base_cls_ids==1]
        else:
            text_embedding = self.cls_embeddings
        rpn_cls_score = (rpn_cls_score.permute(0, 2, 3, 1).reshape(bs, H, W, self.num_base_priors, -1) \
                         @ text_embedding.T).max(dim=-1)[0]
        rpn_cls_score = rpn_cls_score.permute(0, 3, 1, 2)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred
    
    ###### copy from anchor_head ######
    def forward(self, x: Tuple[Tensor],
                 batch_gt_instances: InstanceList = None,
                 batch_img_metas: List[dict] = None,
                 batch_gt_instances_ignore: OptInstanceList = None) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        return multi_apply(self.forward_single, x)
    
        # rpn_cls_score_list, rpn_bbox_pred_list = multi_apply(self.forward_single, x)
        # ###### pre ITC encoder ######
        # batch_size = rpn_cls_score_list[0].size(0)
        # # construct binary masks for the transformer.
        # assert batch_img_metas is not None
        # batch_input_shape = batch_img_metas[0]['batch_input_shape']
        # input_img_h, input_img_w = batch_input_shape
        # img_shape_list = [sample['img_shape'] for sample in batch_img_metas]
        # same_shape_flag = all([
        #     s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        # ])
        # # support torch2onnx without feeding masks
        # if torch.onnx.is_in_onnx_export() or same_shape_flag:
        #     mlvl_masks = []
        #     mlvl_pos_embeds = []
        #     for feat in rpn_cls_score_list:
        #         mlvl_masks.append(None)
        #         mlvl_pos_embeds.append(
        #             self.positional_encoding(None, input=feat))
        # else:
        #     masks = rpn_cls_score_list[0].new_ones(
        #         (batch_size, input_img_h, input_img_w))
        #     for img_id in range(batch_size):
        #         img_h, img_w = img_shape_list[img_id]
        #         masks[img_id, :img_h, :img_w] = 0
        #     # NOTE following the official DETR repo, non-zero
        #     # values representing ignored positions, while
        #     # zero values means valid positions.

        #     mlvl_masks = []
        #     mlvl_pos_embeds = []
        #     for feat in rpn_cls_score_list:
        #         mlvl_masks.append(
        #             F.interpolate(masks[None], size=feat.shape[-2:]).to(
        #                 torch.bool).squeeze(0))
        #         mlvl_pos_embeds.append(
        #             self.positional_encoding(mlvl_masks[-1]))

        # feat_flatten = []
        # lvl_pos_embed_flatten = []
        # mask_flatten = []
        # spatial_shapes = []
        # for lvl, (feat, mask, pos_embed) in enumerate(
        #         zip(rpn_cls_score_list, mlvl_masks, mlvl_pos_embeds)):
        #     batch_size, c, h, w = feat.shape
        #     spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
        #     # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
        #     feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
        #     pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
        #     lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
        #     # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
        #     if mask is not None:
        #         mask = mask.flatten(1)

        #     feat_flatten.append(feat)
        #     lvl_pos_embed_flatten.append(lvl_pos_embed)
        #     mask_flatten.append(mask)
        #     spatial_shapes.append(spatial_shape)

        # # (bs, num_feat_points, dim)
        # feat_flatten = torch.cat(feat_flatten, 1)
        # lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        # if mask_flatten[0] is not None:
        #     mask_flatten = torch.cat(mask_flatten, 1)
        # else:
        #     mask_flatten = None

        # # (num_level, 2)
        # spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        # level_start_index = torch.cat((
        #     spatial_shapes.new_zeros((1, )),  # (num_level)
        #     spatial_shapes.prod(1).cumsum(0)[:-1]))
        # if mlvl_masks[0] is not None:
        #     valid_ratios = torch.stack(  # (bs, num_level, 2)
        #         [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        # else:
        #     valid_ratios = rpn_cls_score_list[0].new_ones(batch_size, len(rpn_cls_score_list),
        #                                           2)
        
        # ###### forward ITC encoder ######
        # output_img_feat = self.ItcEncoder(
        #     image_query=feat_flatten,
        #     image_key_padding_mask=mask_flatten,
        #     image_query_pos=lvl_pos_embed_flatten,
        #     text_query=self.cls_embeddings[None].repeat(batch_size, 1, 1), 
        #     text_query_pos=None, 
        #     text_key_padding_mask=None, 
        #     text_self_attn_mask=None,
        #     cross_attn_mask=None,
        #     spatial_shapes=spatial_shapes,
        #     level_start_index=level_start_index,
        #     valid_ratios=valid_ratios
        # )

    
    ###### copy from base_dense_head ######
    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        # outs = self(x)
        outs = self(x, batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg)
        return losses, predictions
    
    ###### copy from deformable_detr ######
    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        """Get the valid radios of feature map in a level.

        .. code:: text

                    |---> valid_W <---|
                 ---+-----------------+-----+---
                  A |                 |     | A
                  | |                 |     | |
                  | |                 |     | |
            valid_H |                 |     | |
                  | |                 |     | H
                  | |                 |     | |
                  V |                 |     | |
                 ---+-----------------+     | |
                    |                       | V
                    +-----------------------+---
                    |---------> W <---------|

          The valid_ratios are defined as:
                r_h = valid_H / H,  r_w = valid_W / W
          They are the factors to re-normalize the relative coordinates of the
          image to the relative coordinates of the current level feature map.

        Args:
            mask (Tensor): Binary mask of a feature map, has shape (bs, H, W).

        Returns:
            Tensor: valid ratios [r_w, r_h] of a feature map, has shape (1, 2).
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

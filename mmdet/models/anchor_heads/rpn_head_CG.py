import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox,multi_apply
from mmdet.ops import nms
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class RPNHead_CG(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead_CG, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(self.in_channels * 4, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

        self.scale_conv_k_2 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.scale_conv_v_2 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.scale_conv_k_3 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.scale_conv_v_3 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.scale_conv_k_4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.scale_conv_v_4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.scale_conv_k_5 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.scale_conv_v_5 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.scale_conv_k_6 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.scale_conv_v_6 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.batch_ = nn.BatchNorm2d(5)
        self.batch_1 = nn.BatchNorm2d(self.in_channels * 2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

        normal_init(self.scale_conv_k_2, std=0.01)
        normal_init(self.scale_conv_v_2, std=0.01)

        normal_init(self.scale_conv_k_3, std=0.01)
        normal_init(self.scale_conv_v_3, std=0.01)

        normal_init(self.scale_conv_k_4, std=0.01)
        normal_init(self.scale_conv_v_4, std=0.01)

        normal_init(self.scale_conv_k_5, std=0.01)
        normal_init(self.scale_conv_v_5, std=0.01)

        normal_init(self.scale_conv_k_6, std=0.01)
        normal_init(self.scale_conv_v_6, std=0.01)

        normal_init(self.batch_, std=0.01)
        normal_init(self.batch_1, std=0.01)

    def forward_single(self, x):
        batch, C, height, width = x.size()
        query = x.view(batch, C, -1)
        key = x.view(batch, C, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key)
        value = x.view(batch, C, -1)
        atten = F.softmax(energy)

        out = torch.bmm(atten, value)
        out = out.view(batch, C, height, width)

        out = self.batch_1(out)
        x = self.rpn_conv(torch.cat([x,out],dim=1))

        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def forward(self, feats):
        torch.cuda.empty_cache()

        p2 = feats[0]
        scale_k = self.scale_conv_k_2(p2)
        scale_q = scale_k
        scale_v = self.scale_conv_v_2(p2)
        m_batchsize, C, height, width = scale_k.size()
        scale_q = scale_q.view(m_batchsize, C, -1)
        scale_k = scale_k.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(scale_q, scale_k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new)
        proj_value = scale_v.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        p2_out = out.view(m_batchsize, C, height, width)

        p3 = feats[1]
        scale_k = self.scale_conv_k_3(p3)
        scale_q = scale_k
        scale_v = self.scale_conv_v_3(p3)
        m_batchsize, C, height, width = scale_k.size()
        scale_q = scale_q.view(m_batchsize, C, -1)
        scale_k = scale_k.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(scale_q, scale_k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new)
        proj_value = scale_v.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        p3_out = out.view(m_batchsize, C, height, width)
        p3_out_interpolate = F.interpolate(p3_out, scale_factor=2, mode='nearest')

        p4 = feats[2]
        scale_k = self.scale_conv_k_4(p4)
        scale_q = scale_k
        scale_v = self.scale_conv_v_4(p4)
        m_batchsize, C, height, width = scale_k.size()
        scale_q = scale_q.view(m_batchsize, C, -1)
        scale_k = scale_k.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(scale_q, scale_k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new)
        proj_value = scale_v.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        p4_out = out.view(m_batchsize, C, height, width)
        p4_out_interpolate = F.interpolate(p4_out, scale_factor=4, mode='nearest')

        p5 = feats[3]
        scale_k = self.scale_conv_k_5(p5)
        scale_q = scale_k
        scale_v = self.scale_conv_v_5(p5)
        m_batchsize, C, height, width = scale_k.size()
        scale_q = scale_q.view(m_batchsize, C, -1)
        scale_k = scale_k.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(scale_q, scale_k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new)
        proj_value = scale_v.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        p5_out = out.view(m_batchsize, C, height, width)
        p5_out_interpolate = F.interpolate(p5_out, scale_factor=8, mode='nearest')

        p6 = feats[4]
        scale_k = self.scale_conv_k_6(p6)
        scale_q = scale_k
        scale_v = self.scale_conv_v_6(p6)
        m_batchsize, C, height, width = scale_k.size()
        scale_q = scale_q.view(m_batchsize, C, -1)
        scale_k = scale_k.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(scale_q, scale_k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new)
        proj_value = scale_v.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        p6_out = out.view(m_batchsize, C, height, width)
        p6_out_interpolate = F.interpolate(p6_out, scale_factor=16, mode='nearest')

        p_merge = torch.cat([p2_out,p3_out_interpolate,p4_out_interpolate,p5_out_interpolate,p6_out_interpolate],dim=1)

        m_batchsize, C, height, width = p_merge.size()
        proj_query = p_merge.view(m_batchsize, C, -1)
        proj_key = p_merge.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new)
        proj_value = p_merge.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = out.view(m_batchsize, C, -1)
        out_mean = torch.mean(out, dim=2)

        atten_p2 = out_mean[:, 0].unsqueeze(1).expand(m_batchsize, 256) \
            .unsqueeze(2).expand(m_batchsize, 256, 256) \
            .unsqueeze(3).expand(m_batchsize, 256, 256, 256)
        atten_p3 = out_mean[:, 1].unsqueeze(1).expand(m_batchsize, 256) \
            .unsqueeze(2).expand(m_batchsize, 256, 128) \
            .unsqueeze(3).expand(m_batchsize, 256, 128, 128)
        atten_p4 = out_mean[:, 2].unsqueeze(1).expand(m_batchsize, 256) \
            .unsqueeze(2).expand(m_batchsize, 256, 64) \
            .unsqueeze(3).expand(m_batchsize, 256, 64, 64)
        atten_p5 = out_mean[:, 3].unsqueeze(1).expand(m_batchsize, 256) \
            .unsqueeze(2).expand(m_batchsize, 256, 32) \
            .unsqueeze(3).expand(m_batchsize, 256, 32, 32)
        atten_p6 = out_mean[:, 4].unsqueeze(1).expand(m_batchsize, 256) \
            .unsqueeze(2).expand(m_batchsize, 256, 16) \
            .unsqueeze(3).expand(m_batchsize, 256, 16, 16)

        p2_new = torch.cat([atten_p2 * p2,p2],dim=1)
        p3_new = torch.cat([atten_p3 * p3,p3],dim=1)
        p4_new = torch.cat([atten_p4 * p4,p4],dim=1)
        p5_new = torch.cat([atten_p5 * p5,p5],dim=1)
        p6_new = torch.cat([atten_p6 * p6,p6],dim=1)

        feats_new = (p2_new, p3_new, p4_new, p5_new, p6_new)
        rpn_outs = multi_apply(self.forward_single, feats_new)

        return rpn_outs

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(RPNHead_CG, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals

from mmcv.cnn import xavier_init

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.core import multi_apply

from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class FPNAtten(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(FPNAtten, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        channels_ = 256
        self.r_conv = nn.Conv2d(channels_ * 4, channels_, 3, padding=1)

        # self.scale_conv_k_2 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        # # self.scale_conv_q_2 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        # self.scale_conv_v_2 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        #
        # self.scale_conv_k_3 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        # # self.scale_conv_q_3 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        # self.scale_conv_v_3 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        #
        # self.scale_conv_k_4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        # # self.scale_conv_q_4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        # self.scale_conv_v_4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        #
        # self.scale_conv_k_5 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        # # self.scale_conv_q_5 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        # self.scale_conv_v_5 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        #
        # self.scale_conv_k_6 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        # # self.scale_conv_q_6 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        # self.scale_conv_v_6 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.batch_ = nn.BatchNorm2d(5)
        self.batch_11 = nn.BatchNorm2d(channels_ * 2)
        self.gamma = nn.Parameter(torch.zeros(1))


    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        normal_init(self.r_conv, std=0.01)

        # normal_init(self.scale_conv_k_2, std=0.01)
        # # normal_init(self.scale_conv_q_2, std=0.01)
        # normal_init(self.scale_conv_v_2, std=0.01)
        #
        # normal_init(self.scale_conv_k_3, std=0.01)
        # # normal_init(self.scale_conv_q_3, std=0.01)
        # normal_init(self.scale_conv_v_3, std=0.01)
        #
        # normal_init(self.scale_conv_k_4, std=0.01)
        # # normal_init(self.scale_conv_q_4, std=0.01)
        # normal_init(self.scale_conv_v_4, std=0.01)
        #
        # normal_init(self.scale_conv_k_5, std=0.01)
        # # normal_init(self.scale_conv_q_5, std=0.01)
        # normal_init(self.scale_conv_v_5, std=0.01)
        #
        # normal_init(self.scale_conv_k_6, std=0.01)
        # # normal_init(self.scale_conv_q_6, std=0.01)
        # normal_init(self.scale_conv_v_6, std=0.01)

        normal_init(self.batch_, std=0.01)
        normal_init(self.batch_11, std=0.01)

    def forward_single(self, x):

        batch, C, height, width = x.size()
        query = x.view(batch, C, -1)
        key = x.view(batch, C, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # atten = F.softmax(energy_new)
        value = x.view(batch, C, -1)
        atten = F.softmax(energy)

        out = torch.bmm(atten, value)
        out = out.view(batch, C, height, width)
        out = self.batch_11(out)
        x = self.r_conv(torch.cat([x,out],dim=1))

        # x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        # rpn_cls_score = self.rpn_cls(x)
        # rpn_bbox_pred = self.rpn_reg(x)
        # return x.clone().detach()
        return x

    def forward(self, inputs):

        torch.cuda.empty_cache()


        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        #----------------------------------


        feats = tuple(outs)

        p2 = feats[0]
        # print(p2)
        # scale_k = self.scale_conv_k_2(p2)
        # scale_q = scale_k
        # scale_v = self.scale_conv_v_2(p2)
        # m_batchsize, C, height, width = scale_k.size()
        # scale_q = scale_q.view(m_batchsize, C, -1)
        # scale_k = scale_k.view(m_batchsize, C, -1).permute(0, 2, 1)
        # energy = torch.bmm(scale_q, scale_k)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # attention = F.softmax(energy_new)
        # proj_value = scale_v.view(m_batchsize, C, -1)
        #
        # out = torch.bmm(attention, proj_value)
        # p2_out = out.view(m_batchsize, C, height, width)
        p2_out = p2

        p3 = feats[1]
        # scale_k = self.scale_conv_k_3(p3)
        # scale_q = scale_k
        # scale_v = self.scale_conv_v_3(p3)
        # m_batchsize, C, height, width = scale_k.size()
        # scale_q = scale_q.view(m_batchsize, C, -1)
        # scale_k = scale_k.view(m_batchsize, C, -1).permute(0, 2, 1)
        # energy = torch.bmm(scale_q, scale_k)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # attention = F.softmax(energy_new)
        # proj_value = scale_v.view(m_batchsize, C, -1)
        #
        # out = torch.bmm(attention, proj_value)
        # p3_out = out.view(m_batchsize, C, height, width)
        p3_out = p3
        p3_out_interpolate = F.interpolate(p3_out, scale_factor=2, mode='nearest')

        p4 = feats[2]
        # scale_k = self.scale_conv_k_4(p4)
        # scale_q = scale_k
        # scale_v = self.scale_conv_v_4(p4)
        # m_batchsize, C, height, width = scale_k.size()
        # scale_q = scale_q.view(m_batchsize, C, -1)
        # scale_k = scale_k.view(m_batchsize, C, -1).permute(0, 2, 1)
        # energy = torch.bmm(scale_q, scale_k)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # attention = F.softmax(energy_new)
        # proj_value = scale_v.view(m_batchsize, C, -1)
        #
        # out = torch.bmm(attention, proj_value)
        # p4_out = out.view(m_batchsize, C, height, width)
        p4_out = p4
        p4_out_interpolate = F.interpolate(p4_out, scale_factor=4, mode='nearest')

        p5 = feats[3]
        # scale_k = self.scale_conv_k_5(p5)
        # scale_q = scale_k
        # scale_v = self.scale_conv_v_5(p5)
        # m_batchsize, C, height, width = scale_k.size()
        # scale_q = scale_q.view(m_batchsize, C, -1)
        # scale_k = scale_k.view(m_batchsize, C, -1).permute(0, 2, 1)
        # energy = torch.bmm(scale_q, scale_k)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # attention = F.softmax(energy_new)
        # proj_value = scale_v.view(m_batchsize, C, -1)
        #
        # out = torch.bmm(attention, proj_value)
        # p5_out = out.view(m_batchsize, C, height, width)
        p5_out = p5
        p5_out_interpolate = F.interpolate(p5_out, scale_factor=8, mode='nearest')

        p6 = feats[4]
        # scale_k = self.scale_conv_k_6(p6)
        # scale_q = scale_k
        # scale_v = self.scale_conv_v_6(p6)
        # m_batchsize, C, height, width = scale_k.size()
        # scale_q = scale_q.view(m_batchsize, C, -1)
        # scale_k = scale_k.view(m_batchsize, C, -1).permute(0, 2, 1)
        # energy = torch.bmm(scale_q, scale_k)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # attention = F.softmax(energy_new)
        # proj_value = scale_v.view(m_batchsize, C, -1)
        #
        # out = torch.bmm(attention, proj_value)
        # p6_out = out.view(m_batchsize, C, height, width)
        p6_out = p6
        p6_out_interpolate = F.interpolate(p6_out, scale_factor=16, mode='nearest')

        p_merge = torch.cat([p2_out, p3_out_interpolate, p4_out_interpolate, p5_out_interpolate, p6_out_interpolate],
                            dim=1)

        m_batchsize, C, height, width = p_merge.size()
        proj_query = p_merge.view(m_batchsize, C, -1)
        proj_key = p_merge.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new)
        proj_value = p_merge.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        # out = self.batch_(out)
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

        p2_new = torch.cat([atten_p2 * p2, p2], dim=1)
        p3_new = torch.cat([atten_p3 * p3, p3], dim=1)
        p4_new = torch.cat([atten_p4 * p4, p4], dim=1)
        p5_new = torch.cat([atten_p5 * p5, p5], dim=1)
        p6_new = torch.cat([atten_p6 * p6, p6], dim=1)

        # feats_new = (p2_new, p3_new, p4_new, p5_new)
        feats_new = (p2_new, p3_new, p4_new, p5_new, p6_new)
        # out_mean[:, 4].unsqueeze(1).expand(2, 256).unsqueeze(2).expand(2, 256, 16).unsqueeze(3).expand(2, 256, 16, 16)

        outs = multi_apply(self.forward_single, feats_new)
        # outs = (outs[0][0],outs[0][1], outs[0][2],outs[0][3],outs[0][4])
        outs = (torch.unsqueeze(outs[0][0],0),torch.unsqueeze(outs[0][1],0), torch.unsqueeze(outs[0][2],0),torch.unsqueeze(outs[0][3],0),torch.unsqueeze(outs[0][4],0))


        return outs
        # return tuple(outs)

from __future__ import division

import torch
import torch.nn as nn

from .base_new import BaseDetectorNew
from .test_mixins import RPNTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import (build_assigner, bbox2roi, dbbox2roi, bbox2result, build_sampler,
                        dbbox2result, merge_aug_masks, roi2droi, mask2poly,
                        get_best_begin_point, polygonToRotRectangle_batch,
                        gt_mask_bp_obbs_list, choose_best_match_batch,
                        choose_best_Rroi_batch, dbbox_rotate_mapping, bbox_rotate_mapping)
from mmdet.core import (bbox_mapping, merge_aug_proposals, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms, merge_rotate_aug_proposals,
                        merge_rotate_aug_bboxes, multiclass_nms_rbbox)
import copy
from mmdet.core import RotBox2Polys, polygonToRotRectangle_batch
import cv2 as cv
import mmcv
import numpy as np
import random



@DETECTORS.register_module
class RoITransformerFcosSample(BaseDetectorNew, RPNTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 shared_head_rbbox=None,
                 rpn_head=None,
                 fcos_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 rbbox_roi_extractor=None,
                 rbbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None

        assert rbbox_roi_extractor is not None
        assert rbbox_head is not None
        super(RoITransformerFcosSample, self).__init__()

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if fcos_head is not None:
            self.fcos_head = builder.build_head(fcos_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if shared_head_rbbox is not None:
            self.shared_head_rbbox = builder.build_shared_head(shared_head_rbbox)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
        # import pdb
        # pdb.set_trace()
        if rbbox_head is not None:
            self.rbbox_roi_extractor = builder.build_roi_extractor(
                rbbox_roi_extractor)
            self.rbbox_head = builder.build_head(rbbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.rbbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_fcos(self):
        return hasattr(self, 'fcos_head') and self.fcos_head is not None

    def init_weights(self, pretrained=None):
        super(RoITransformerFcosSample, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_fcos:
            self.fcos_head.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_shared_head_rbbox:
            self.shared_head_rbbox.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_rbbox:
            self.rbbox_roi_extractor.init_weights()
            self.rbbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)      #true:backbone+fpn
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()

        # trans gt_masks to gt_obbs
        gt_obbs = gt_mask_bp_obbs_list(gt_masks)

        #FCOS forward and loss
        if self.with_fcos:
            fcos_outs=self.fcos_head(x)
            fcos_loss_inputs = fcos_outs + (gt_bboxes, gt_labels, img_meta, self.train_cfg.fcos)
            fcos_losses = self.fcos_head.loss(
                *fcos_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(fcos_losses)

            bboxes_pred_inputs = fcos_outs + (img_meta, self.test_cfg.fcos)
            fcos_bboxes_pred = self.fcos_head.get_bboxes(*bboxes_pred_inputs)

            fcos_proposal = [0] * len(fcos_bboxes_pred)
            for i in range(len(fcos_bboxes_pred)):
                fcos_bboxes_pred_all=fcos_bboxes_pred[i][0]
                # if len(fcos_bboxes_pred_all)>=100:
                #     values,topk_inds=fcos_bboxes_pred_all[:,4].topk(100)
                #     fcos_all_proposal_tenor=fcos_bboxes_pred_all[topk_inds, :]
                #     # fcos_all_proposal_tenor = torch.squeeze(fcos_bboxes_pred_all[topk_inds, :])
                # else:
                #     fcos_all_proposal_tenor = fcos_bboxes_pred_all
                fcos_all_proposal_tenor = fcos_bboxes_pred_all
                fcos_proposal_tensor = fcos_all_proposal_tenor
                if len(fcos_proposal_tensor.shape) >= 2:
                    fcos_proposal[i] = fcos_proposal_tensor
                else:
                    fcos_proposal[i] = fcos_proposal_tensor.unsqueeze(0)


            # fcos_proposal = [0]*len(fcos_bboxes_pred)
            # for i in range(len(fcos_bboxes_pred)):
            #     fcos_bboxes_pred_p2=fcos_bboxes_pred[i][2]     #output of p2
            #     fcos_bboxes_pred_p3 = fcos_bboxes_pred[i][4]   #output of p3
            #     fcos_bboxes_pred_p4 = fcos_bboxes_pred[i][6]  # output of p4
            #     fcos_bboxes_pred_p5 = fcos_bboxes_pred[i][8]  # output of p5
            #     fcos_bboxes_pred_p6 = fcos_bboxes_pred[i][10]  # output of p6
            #
            #     fcos_p2_proposal_tensor = torch.squeeze(fcos_bboxes_pred_p2)
            #     fcos_p3_proposal_tensor = torch.squeeze(fcos_bboxes_pred_p3)
            #     fcos_p4_proposal_tensor = torch.squeeze(fcos_bboxes_pred_p4)
            #     fcos_p5_proposal_tensor = torch.squeeze(fcos_bboxes_pred_p5)
            #     fcos_p6_proposal_tensor = torch.squeeze(fcos_bboxes_pred_p6)
            #
            #     if len(fcos_p2_proposal_tensor.shape) < 2:
            #         fcos_p2_proposal_tensor = fcos_p2_proposal_tensor.unsqueeze(0)
            #     if len(fcos_p3_proposal_tensor.shape) < 2:
            #         fcos_p3_proposal_tensor = fcos_p3_proposal_tensor.unsqueeze(0)
            #     if len(fcos_p4_proposal_tensor.shape) < 2:
            #         fcos_p4_proposal_tensor = fcos_p4_proposal_tensor.unsqueeze(0)
            #     if len(fcos_p5_proposal_tensor.shape) < 2:
            #         fcos_p5_proposal_tensor = fcos_p5_proposal_tensor.unsqueeze(0)
            #     if len(fcos_p6_proposal_tensor.shape) < 2:
            #         fcos_p6_proposal_tensor = fcos_p6_proposal_tensor.unsqueeze(0)
            #
            #     fcos_proposal_tensor = torch.cat([fcos_p2_proposal_tensor, fcos_p3_proposal_tensor])
            #     fcos_proposal_tensor = torch.cat([fcos_proposal_tensor, fcos_p4_proposal_tensor])
            #     fcos_proposal_tensor = torch.cat([fcos_proposal_tensor, fcos_p5_proposal_tensor])
            #     fcos_proposal_tensor = torch.cat([fcos_proposal_tensor, fcos_p6_proposal_tensor])
            #
            #     if len(fcos_proposal_tensor.shape) >= 2:
            #         fcos_proposal[i] = fcos_proposal_tensor
            #     else:
            #         fcos_proposal[i] = fcos_proposal_tensor.unsqueeze(0)

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)


            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

            ######## add fcos_proposal
            proposal_list1 = [0]*len(proposal_list)
            for j in range(len(proposal_list)):
                try:
                    proposal_list1[j] = torch.cat([fcos_proposal[j], proposal_list[j]])
                    # proposal_list1[j] = proposal_list1[j][0:2000]

                except:
                    print(fcos_proposal[0].shape, proposal_list[0].shape)
                    exit()

        else:
            proposal_list = proposals

        # ###### show proposal##########
        # for b in range(len(img)):
        #     image_0 = img[b].permute(1, 2, 0).cpu().numpy()
        #     _range = np.max(image_0) - np.min(image_0)
        #     image = ((image_0 - np.min(image_0)) / _range) * 255
        #     # cv.imwrite('/home/cver/data/GQX/AerialDetection/show_anchor/1.png',image)
        #     gt_box = gt_bboxes[b].cpu().numpy()
        #     fcos_anchor = fcos_proposal[b].cpu().numpy()
        #     RPN_anchor = proposal_list[b].cpu().numpy()
        #     for k in range(len(gt_box)):
        #         xxx1 = gt_box[k, 0]
        #         yyy1 = gt_box[k, 1]
        #         xxx2 = gt_box[k, 2]
        #         yyy2 = gt_box[k, 3]
        #         image = cv.rectangle(image, (xxx1, yyy1), (xxx2, yyy2), (0, 255, 0), 1)
        #     for i in range(len(fcos_anchor)):
        #         x1 = fcos_anchor[i, 0]
        #         y1 = fcos_anchor[i, 1]
        #         x2 = fcos_anchor[i, 2]
        #         y2 = fcos_anchor[i, 3]
        #         image = cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        #     # # for j in range(len(RPN_anchor)):
        #     # for j in range(500):
        #     #     xx1 = RPN_anchor[j, 0]
        #     #     yy1 = RPN_anchor[j, 1]
        #     #     xx2 = RPN_anchor[j, 2]
        #     #     yy2 = RPN_anchor[j, 3]
        #     #     image = cv.rectangle(image, (xx1, yy1), (xx2, yy2), (0, 0, 255), 1)
        #
        #     random_index = str(random.random() * 10000)
        #     cv.imwrite(
        #         '/home/cver/data/GQX/AerialDetection/Proposal_show/epoch4/FCOS_all_top100_gt/' + random_index + '.png',
        #         image)
        #########show proposal##########

        # assign gts and sample proposals (hbb assign)
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn[0].assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn[0].sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]    #gt_bboxes_ignore = None
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            ###### fcos_proposal assign gts  and sample proposals
            for i in range(len(fcos_proposal)):
                if len(fcos_proposal[i]) > 0:
                    fcos_bbox_assigner = build_assigner(self.train_cfg.rcnn[0].assigner)
                    fcos_bbox_sampler = build_sampler(
                        self.train_cfg.rcnn[0].sampler, context=self)
                    num_imgs = img.size(0)
                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]  # gt_bboxes_ignore = None
                    fcos_sampling_results = []
                    for i in range(num_imgs):
                        fcos_assign_result = fcos_bbox_assigner.assign(
                            fcos_proposal[i], gt_bboxes[i], gt_bboxes_ignore[i],
                            gt_labels[i])
                        fcos_sampling_result = fcos_bbox_sampler.sample(
                            fcos_assign_result,
                            fcos_proposal[i],
                            gt_bboxes[i],
                            gt_labels[i],
                            feats=[lvl_feat[i][None] for lvl_feat in x])
                        fcos_sampling_results.append(fcos_sampling_result)
            # for i in range(num_imgs):
            #     pos_bboxes_ = sampling_results[i].pos_bboxes
            #     fcos_proposal_i = fcos_proposal[i]
            #     if len(pos_bboxes_) + len(fcos_proposal_i) > 128:
            #         pos_bboxes_i = torch.cat([pos_bboxes_,fcos_proposal_i[:,:4]])[:128]
            #     else:
            #         pos_bboxes_i = torch.cat([pos_bboxes_, fcos_proposal_i[:, :4]])
            #     pos_inds_i = torch.tensor([x for x in range(len(pos_bboxes_i))])
            #     pos_is_gt_i_add = torch.tensor([0 for x in range((len(pos_bboxes_i) - len(pos_bboxes_)))])
            #     pos_is_gt_i = torch.cat([sampling_results[i].pos_is_gt ,pos_is_gt_i_add.cuda().type(torch.uint8)])
            #
            #     sampling_results[i].pos_bboxes = pos_bboxes_i
            #     sampling_results[i].pos_inds = pos_inds_i
            #     sampling_results[i].pos_is_gt = pos_is_gt_i
            #
            #     neg_bboxes_ = sampling_results[i].neg_bboxes
            #     sampling_results[i].neg_bboxes = neg_bboxes_[:int(len(neg_bboxes_) - (len(pos_bboxes_i) - len(pos_bboxes_)))]
            #     sampling_results[i].neg_inds = torch.tensor([x + len(pos_bboxes_i) for x in range(len(neg_bboxes_))])



        # RoI Transformer or RoI pooling?

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            ## rbbox
            rbbox_targets = self.bbox_head.get_target(sampling_results, gt_masks, gt_labels, self.train_cfg.rcnn[0])

            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *rbbox_targets)
            # losses.update(loss_bbox)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(0, name)] = (value)

            #fcos proposal to rcnn bbox_head
            for i in range(len(fcos_proposal)):
                if len(fcos_proposal[i]) > 0:
                    fcos_rois = bbox2roi([res.bboxes for res in fcos_sampling_results])
                    fcos_bbox_feats = self.bbox_roi_extractor(
                        x[:self.bbox_roi_extractor.num_inputs], fcos_rois)
                    if self.with_shared_head:
                        fcos_bbox_feats = self.shared_head(fcos_bbox_feats)
                    fcos_cls_score, fcos_bbox_pred = self.bbox_head(fcos_bbox_feats)
                    ##fcos_rbbox
                    fcos_rbbox_targets = self.bbox_head.get_target(fcos_sampling_results, gt_masks, gt_labels, self.train_cfg.rcnn[0])
                    fcos_loss_bbox = self.bbox_head.loss(fcos_cls_score, fcos_bbox_pred,
                                                    *fcos_rbbox_targets)
                    # losses.update(fcos_loss_bbox)
                    for name, value in fcos_loss_bbox.items():
                        losses['s{}.{}'.format(00, name)] = (value)


        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        roi_labels = rbbox_targets[0]
        with torch.no_grad():
            # import pdb
            # pdb.set_trace()
            ### ROI Transformer
            rotated_proposal_list = self.bbox_head.refine_rbboxes(
                roi2droi(rois), roi_labels, bbox_pred, pos_is_gts, img_meta
            )

        for i in range(len(fcos_proposal)):
            if len(fcos_proposal[i]) > 0:
                fcos_pos_is_gts = [res.pos_is_gt for res in fcos_sampling_results]
                fcos_roi_labels = fcos_rbbox_targets[0]
                with torch.no_grad():
                    # import pdb
                    # pdb.set_trace()
                    ### ROI Transformer
                    fcos_rotated_proposal_list = self.bbox_head.refine_rbboxes(
                        roi2droi(fcos_rois), fcos_roi_labels, fcos_bbox_pred, fcos_pos_is_gts, img_meta
                    )

        # import pdb
        # pdb.set_trace()
        # assign gts and sample proposals (rbb assign)
        if self.with_rbbox:
            bbox_assigner = build_assigner(self.train_cfg.rcnn[1].assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn[1].sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                gt_obbs_best_roi = choose_best_Rroi_batch(gt_obbs[i])
                assign_result = bbox_assigner.assign(
                    rotated_proposal_list[i], gt_obbs_best_roi, gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    rotated_proposal_list[i],
                    torch.from_numpy(gt_obbs_best_roi).float().to(rotated_proposal_list[i].device),
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            ##fcos rbbox proposal assign and sample
            for i in range(len(fcos_proposal)):
                if len(fcos_proposal[i]) > 0:
                    fcos_bbox_assigner = build_assigner(self.train_cfg.rcnn[1].assigner)
                    fcos_bbox_sampler = build_sampler(
                        self.train_cfg.rcnn[1].sampler, context=self)
                    num_imgs = img.size(0)
                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    fcos_sampling_results = []
                    for i in range(num_imgs):
                        fcos_gt_obbs_best_roi = choose_best_Rroi_batch(gt_obbs[i])
                        fcos_assign_result = fcos_bbox_assigner.assign(
                            fcos_rotated_proposal_list[i], fcos_gt_obbs_best_roi, gt_bboxes_ignore[i],
                            gt_labels[i])
                        fcos_sampling_result = fcos_bbox_sampler.sample(
                            fcos_assign_result,
                            fcos_rotated_proposal_list[i],
                            torch.from_numpy(fcos_gt_obbs_best_roi).float().to(fcos_rotated_proposal_list[i].device),
                            gt_labels[i],
                            feats=[lvl_feat[i][None] for lvl_feat in x])
                        fcos_sampling_results.append(fcos_sampling_result)

        if self.with_rbbox:
            # (batch_ind, x_ctr, y_ctr, w, h, angle)
            rrois = dbbox2roi([res.bboxes for res in sampling_results])
            # feat enlarge
            # rrois[:, 3] = rrois[:, 3] * 1.2
            # rrois[:, 4] = rrois[:, 4] * 1.4
            rrois[:, 3] = rrois[:, 3] * self.rbbox_roi_extractor.w_enlarge
            rrois[:, 4] = rrois[:, 4] * self.rbbox_roi_extractor.h_enlarge
            rbbox_feats = self.rbbox_roi_extractor(x[:self.rbbox_roi_extractor.num_inputs],
                                                   rrois)
            if self.with_shared_head_rbbox:
                rbbox_feats = self.shared_head_rbbox(rbbox_feats)
            cls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
            rbbox_targets = self.rbbox_head.get_target_rbbox(sampling_results, gt_obbs,
                                                        gt_labels, self.train_cfg.rcnn[1])
            loss_rbbox = self.rbbox_head.loss(cls_score, rbbox_pred, *rbbox_targets)
            for name, value in loss_rbbox.items():
                losses['s{}.{}'.format(1, name)] = (value)

            ##fcos rbbox loss
            for i in range(len(fcos_proposal)):
                if len(fcos_proposal[i]) > 0:
                    fcos_rrois = dbbox2roi([res.bboxes for res in fcos_sampling_results])
                    fcos_rrois[:, 3] = fcos_rrois[:, 3] * self.rbbox_roi_extractor.w_enlarge
                    fcos_rrois[:, 4] = fcos_rrois[:, 4] * self.rbbox_roi_extractor.h_enlarge
                    fcos_rbbox_feats = self.rbbox_roi_extractor(x[:self.rbbox_roi_extractor.num_inputs],
                                                           fcos_rrois)
                    if self.with_shared_head_rbbox:
                        fcos_rbbox_feats = self.shared_head_rbbox(fcos_rbbox_feats)
                    fcos_cls_score, fcos_rbbox_pred = self.rbbox_head(fcos_rbbox_feats)
                    fcos_rbbox_targets = self.rbbox_head.get_target_rbbox(fcos_sampling_results, gt_obbs,
                                                                gt_labels, self.train_cfg.rcnn[1])
                    fcos_loss_rbbox = self.rbbox_head.loss(fcos_cls_score, fcos_rbbox_pred, *fcos_rbbox_targets)
                    for name, value in fcos_loss_rbbox.items():
                        losses['s{}.{}'.format(11, name)] = (value)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']

        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        # ______________________________________#
        bbox_label = cls_score.argmax(dim=1)
        rrois = self.bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label, bbox_pred,
                                                      img_meta[0])

        rrois_enlarge = copy.deepcopy(rrois)
        rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor.w_enlarge
        rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor.h_enlarge
        rbbox_feats = self.rbbox_roi_extractor(
            x[:len(self.rbbox_roi_extractor.featmap_strides)], rrois_enlarge)
        if self.with_shared_head_rbbox:
            rbbox_feats = self.shared_head_rbbox(rbbox_feats)

        rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
        det_rbboxes, det_labels = self.rbbox_head.get_det_rbboxes(
            rrois,
            rcls_score,
            rbbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        rbbox_results = dbbox2result(det_rbboxes, det_labels,
                                     self.rbbox_head.num_classes)

        # FCOS simple_test
        # fcos_outs = self.fcos_head(x)
        # fcos_bbox_inputs = fcos_outs + (img_meta, self.test_cfg.fcos, rescale)
        # fcos_bbox_list = self.fcos_head.get_bboxes(*fcos_bbox_inputs)
        # fcos_bbox_results = [
        #     bbox2result(fcos_det_bboxes, fcos_det_labels, self.fcos_head.num_classes)
        #     for fcos_det_bboxes, fcos_det_labels in fcos_bbox_list
        # ]

        return rbbox_results

    def aug_test(self, imgs, img_metas, proposals=None, rescale=None):
        # raise NotImplementedError
        # import pdb; pdb.set_trace()
        proposal_list = self.aug_test_rpn_rotate(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        rcnn_test_cfg = self.test_cfg.rcnn

        aug_rbboxes = []
        aug_rscores = []
        for x, img_meta in zip(self.extract_feats(imgs), img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)

            angle = img_meta[0]['angle']
            # print('img shape: ', img_shape)
            if angle != 0:
                try:

                    proposals = bbox_rotate_mapping(proposal_list[0][:, :4], img_shape,
                                                angle)
                except:
                    import pdb; pdb.set_trace()
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)


            bbox_label = cls_score.argmax(dim=1)
            rrois = self.bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label,
                                                          bbox_pred,
                                                          img_meta[0])

            rrois_enlarge = copy.deepcopy(rrois)
            rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor.w_enlarge
            rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor.h_enlarge
            rbbox_feats = self.rbbox_roi_extractor(
                x[:len(self.rbbox_roi_extractor.featmap_strides)], rrois_enlarge)
            if self.with_shared_head_rbbox:
                rbbox_feats = self.shared_head_rbbox(rbbox_feats)

            rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
            rbboxes, rscores = self.rbbox_head.get_det_rbboxes(
                rrois,
                rcls_score,
                rbbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=None)
            aug_rbboxes.append(rbboxes)
            aug_rscores.append(rscores)

        merged_rbboxes, merged_rscores = merge_rotate_aug_bboxes(
            aug_rbboxes, aug_rscores, img_metas, rcnn_test_cfg
        )
        det_rbboxes, det_rlabels = multiclass_nms_rbbox(
                                merged_rbboxes, merged_rscores, rcnn_test_cfg.score_thr,
                                rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)

        if rescale:
            _det_rbboxes = det_rbboxes
        else:
            _det_rbboxes = det_rbboxes.clone()
            _det_rbboxes[:, :4] *= img_metas[0][0]['scale_factor']

        rbbox_results = dbbox2result(_det_rbboxes, det_rlabels,
                                     self.rbbox_head.num_classes)
        return rbbox_results










import torch
import os
import re
import argparse

import cv2
import numpy as np

from skimage import io
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt

import warnings

import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

from mmdet.core import bbox2roi

def parse_args():
    parser = argparse.ArgumentParser(description='MManGoCam demo')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--layer_name', help='layer name')
    args = parser.parse_args()
    return args

class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]
        # print('gradient:', self.gradient)

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))
#         module = self.net.rpn.head.bbox_pred
#         self.handlers.append(module.register_forward_hook(self._get_features_hook))
#         self.handlers.append(module.register_backward_hook(self._get_grads_hook))
            
                # print(self.handlers)

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        # output = self.net([inputs])
        x = self.net.extract_feat(inputs['img'][0])
        rpn_outs = self.net.rpn_head(x)
        result_list = get_bboxes_tmp(rpn_outs,inputs['img_metas'],self.net.rpn_head.test_cfg)
        #res= self.net.roi_head.simple_test_bboxes(
        #    x, inputs['img_metas'], result_list, self.net.roi_head.test_cfg, rescale=True)
        #############################################################################
        # 由於cascade rcnn的roi-head是cascade roi head 因此涉及到stage的問題，需要重寫roi-head.simple_test_bboxes
        # 其中stage的編號要對上
        ###########################################################################3
        rois = bbox2roi(result_list)
        bbox_results = self.net.roi_head._bbox_forward(2,x,rois) # stage 2
        img_metas = inputs['img_metas']
        proposals = result_list
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.net.roi_head.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.net.roi_head.bbox_head[2].get_bboxes( # stage 2
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=True,
                cfg=self.net.test_cfg['rcnn'])
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            
        res = [det_bboxes, det_labels]
        ########################################################################################
        # print(output)
        score = res[0][0][index][4]
        # proposal_idx = output[0]['labels'][index]  # box来自第几个proposal
        # print(score)
        score.backward()
        # print('gradient:', self.gradient)

        # gradient = self.gradient[proposal_idx].cpu().data.numpy()  # [C,H,W]
        gradient = self.gradient.cpu().data.numpy().squeeze()
        
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        # feature = self.feature[proposal_idx].cpu().data.numpy()  # [C,H,W]
        feature = self.feature.cpu().data.numpy().squeeze()

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        # print(cam.shape)
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        # box = output[0]['instances'].pred_boxes.tensor[index].detach().numpy().astype(np.int32)
        box = res[0][0][index][:-1].detach().numpy().astype(np.int32)
        x1, y1, x2, y2 = box
        
        # cam = cv2.resize(cam, (x2 - x1, y2 - y1))
        # cam = cv2.resize(cam, (y2 - y1, x2 - x1)).T
        # print(cam.shape)

        # class_id = output[0]['instances'].pred_classes[index].detach().numpy()
        class_id = res[1][0][index].detach().numpy()
        plt.imshow(cam)
        return cam, box, class_id

def get_bboxes_tmp(rpn_outs, img_metas, test_cfg):
    
    with_nms = False
    rescale = False
    cls_scores = rpn_outs[0]
    bbox_preds = rpn_outs[1]

    num_levels = len(cls_scores)

    device = cls_scores[0].device
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_anchors = model.rpn_head.anchor_generator.grid_anchors(
        featmap_sizes, device=device)

    result_list = []
    for img_id in range(len(img_metas)):
        cls_score_list = [
            cls_scores[i][img_id] for i in range(num_levels) # 此處不再使用detach()
        ]
        bbox_pred_list = [
            bbox_preds[i][img_id] for i in range(num_levels) # 此處不再使用detach()
        ]
        img_shape = img_metas[img_id]['img_shape']
        scale_factor = img_metas[img_id]['scale_factor']

        if with_nms:
            # some heads don't support with_nms argument
            proposals = model.rpn_head._get_bboxes_single(cls_score_list,
                                                bbox_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, test_cfg, rescale)
        else:
            proposals = model.rpn_head._get_bboxes_single(cls_score_list,
                                                bbox_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, test_cfg, rescale)
        result_list.append(proposals)
    
    return result_list

def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), heatmap

def main():
    args = parse_args()

    cfg =Config.fromfile(args.config)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    model.CLASSES = checkpoint['meta']['CLASSES']

    model.eval()

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    data = []
    for i, t in enumerate(data_loader):
        tmp = {}
        tmp['img'] = t['img']
        tmp['img_metas'] = t['img_metas'][0].data[0]
        data.append(tmp)
    test_cfg = model.rpn_head.test_cfg
    grad_cam = GradCAM(model, args.layer_name)
    mask, box, class_id = grad_cam(data[0],2)

    image_dict = {}
    im = data[0]['img'][0].squeeze(0).permute(1,2,0)
    mask = cv2.resize(mask, (im.shape[1], im.shape[0]))
    image_cam, image_dict['heatmap'] = gen_cam(im, mask)
    image_cam.save('examples/main_r101_result.jpg')
    
if __name__ == '__main__':
    main()
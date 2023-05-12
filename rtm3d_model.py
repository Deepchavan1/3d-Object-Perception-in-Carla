#!/usr/bin/env python3

import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np


colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 120, 0],
          [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]


class RTM3D:
    def __init__(self):
        # parser = argparse.ArgumentParser(description='Demonstration config for RTM3D Implementation')



        self.configs = {}
        self.configs['saved_fn'] = 'rtm3d'
        self.configs['arch'] = 'resnet_18'
        self.configs['pretrained_path'] = '/home/dorleco/RTM3D/checkpoints/rtm3d_resnet_18/Complete_model_rtm3d_resnet_18_epoch_15.pth'
        self.configs['head_conv'] = -1
        self.configs['K'] = 100


        # parser.add_argument('--saved_fn', type=str, default='rtm3d', metavar='FN',
        #                     help='The name using for saving logs, models,...')
        # parser.add_argument('-a', '--arch', type=str, default='resnet_18', metavar='ARCH',
        #                     help='The name of the model architecture')
        # parser.add_argument('--pretrained_path', type=str, default="/home/dorleco/RTM3D/checkpoints/rtm3d_resnet_18/Complete_model_rtm3d_resnet_18_epoch_15.pth", metavar='PATH',
        #                     help='the path of the pretrained checkpoint')
        # parser.add_argument('--head_conv', type=int, default=-1,
        #                     help='conv layer channels for output head'
        #                         '0 for no conv layer'
        #                         '-1 for default setting: '
        #                         '64 for resnets and 256 for dla.')
        # parser.add_argument('--K', type=int, default=100,
        #                     help='the number of top K')




        # parser.add_argument('--use_left_cam_prob', type=float, default=1.,
        #                     help='The probability of using the left camera')
        # parser.add_argument('--dynamic-sigma', action='store_true',
        #                     help='If true, compute sigma based on Amax, Amin then generate heamap'
        #                         'If false, compute radius as CenterNet did')
        # parser.add_argument('--no_cuda', action='store_true',
        #                     help='If true, cuda is not used.')
        # parser.add_argument('--gpu_idx', default=0, type=int,
        #                     help='GPU index to use.')


        self.configs['use_left_cam_prob'] = 1
        # self.configs['dynamic-sigma'] 
        self.configs['no_cuda'] = False
        self.configs['gpu_idx'] = 0




        # parser.add_argument('--num_samples', type=int, default=None,
        #                     help='Take a subset of the dataset to run and debug')
        # parser.add_argument('--num_workers', type=int, default=1,
        #                     help='Number of threads for loading data')
        # parser.add_argument('--batch_size', type=int, default=1,
        #                     help='mini-batch size (default: 4)')
        # parser.add_argument('--peak_thresh', type=float, default=0.2)
        # parser.add_argument('--show_image', action='store_true',
        #                     help='If true, show the image during demostration')


        self.configs['num_samples'] = None
        self.configs['num_workers'] = 1
        self.configs['batch_size'] = 1
        self.configs['peak_thresh'] = 0.2
        self.configs['show_image'] = False




        # parser.add_argument('--save_test_output', action='store_true', default=True,
        #                     help='If true, the output image of the testing phase will be saved')
        # parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
        #                     help='the type of the test output (support image or video)')
        # parser.add_argument('--output_video_fn', type=str, default='out_rtm3d', metavar='PATH',
        #                     help='the video filename if the output format is video')

        
        self.configs['save_test_output'] = True
        self.configs['output_format'] = 'image'
        self.configs['output_video_fn'] = 'out_rtm3d'




        


        # self.configs = edict(vars(parser.parse_args()))
        self.configs['pin_memory'] = True
        self.configs['distributed'] = False  # For testing on 1 GPU only
        self.configs['input_size'] = (384, 1280)
        self.configs['hm_size'] = (96, 320)
        self.configs['down_ratio'] = 4
        self.configs['max_objects'] = 50

        if self.configs['head_conv'] == -1:  # init default head_conv
            self.configs['head_conv = 256'] if 'dla' in self.configs['arch'] else 64

        self.configs['num_classes'] = 3
        self.configs['num_vertexes'] = 8
        self.configs['num_center_offset'] = 2
        self.configs['num_vertexes_offset'] = 2
        self.configs['num_dimension'] = 3
        self.configs['num_rot'] = 8
        self.configs['num_depth'] = 1
        self.configs['num_wh'] = 2
        self.configs['heads'] = {
            'hm_mc': self.configs['num_classes'],
            'hm_ver': self.configs['num_vertexes'],
            'vercoor': self.configs['num_vertexes'] * 2,
            'cenoff': self.configs['num_center_offset'],
            'veroff': self.configs['num_vertexes_offset'],
            'dim': self.configs['num_dimension'],
            'rot': self.configs['num_rot'],
            'depth': self.configs['num_depth'],
            'wh': self.configs['num_wh']
        }

        ####################################################################
        ##############Dataset, Checkpoints, and results dir self.configs#########
        ####################################################################
        self.configs['root_dir'] = '../'
        self.configs['dataset_dir'] = os.path.join(self.configs['root_dir'], 'dataset', 'kitti')

        # print(self.configs['save_test_output'])

        if self.configs['save_test_output']:
            self.configs['results_dir'] = os.path.join(self.configs['root_dir'], 'results', self.configs['saved_fn'])
            # print(self.configs['results_dir'])
            self.make_folder(self.configs['results_dir'])


    def make_folder(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    

    def denormalize_img(slef, img):
        mean_rgb = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        std_rgb = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        return ((img * std_rgb + mean_rgb) * 255).astype(np.uint8)


    def normalize_img(self, img):
        mean_rgb = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        std_rgb = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        return (img / 255. - mean_rgb) / std_rgb

    def detect(self, imgs):
        model = torch.load(self.configs['pretrained_path'])

        self.configs['device']= torch.device('cpu' if self.configs['no_cuda'] else 'cuda:{}'.format(self.configs['gpu_idx']))


        model = model.to(device=self.configs['device'])

        model.eval()
        
        with torch.no_grad():
            
            imgs = self.normalize_img(imgs)
            imgs = imgs.transpose(2, 0, 1)
            imgs = imgs.reshape(1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
            imgs =  torch.from_numpy(imgs)

            input_imgs = imgs.to(device=self.configs['device']).float()
            outputs = model(input_imgs)

            # print(outputs)

            outputs['hm_mc'] = _sigmoid(outputs['hm_mc'])
            outputs['hm_ver'] = _sigmoid(outputs['hm_ver'])
            outputs['depth'] = 1. / (_sigmoid(outputs['depth']) + 1e-9) - 1.
            
            detections = rtm3d_decode(outputs['hm_mc'], outputs['hm_ver'], outputs['vercoor'], outputs['cenoff'],
                                    outputs['veroff'], outputs['wh'], outputs['rot'], outputs['depth'],
                                    outputs['dim'], K=self.configs['K'])
            detections = detections.cpu().numpy()

            detections = post_processing_2d(detections, self.configs['num_classes'], self.configs['down_ratio'])
            detections = get_final_pred(detections[0], self.configs['num_classes'], self.configs['peak_thresh'])

            img = imgs.squeeze().permute(1, 2, 0).numpy()
            img = self.denormalize_img(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # calib = kitti_data_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))

            # Draw prediction in the image
            out_img = draw_predictions(img, detections, colors, self.configs['num_classes'], show_3dbox=True)
            cv2.imwrite(os.path.join(self.configs['results_dir'], 'result.jpg'), out_img)

        cv2.destroyAllWindows()
        return out_img


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def rtm3d_decode(hm_mc, hm_ver, ver_coor, cen_off, ver_off, wh, rot, depth, dim, K=40, hm_size=(96, 320)):
    device = hm_mc.device
    batch_size, num_classes, height, width = hm_mc.size()
    num_vertexes = hm_ver.size(1)

    hm_mc = _nms(hm_mc)
    scores, inds, clses, ys, xs = _topk(hm_mc, K=K)
    if cen_off is not None:
        cen_off = _transpose_and_gather_feat(cen_off, inds)
        cen_off = cen_off.view(batch_size, K, 2)
        xs = xs.view(batch_size, K, 1) + cen_off[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + cen_off[:, :, 1:2]
    else:
        xs = xs.view(batch_size, K, 1) + 0.5
        ys = ys.view(batch_size, K, 1) + 0.5

    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch_size, K, 8)
    depth = _transpose_and_gather_feat(depth, inds)
    depth = depth.view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch_size, K, 3)
    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch_size, K, 2)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)

    ver_coor = _transpose_and_gather_feat(ver_coor, inds)
    ver_coor = ver_coor.view(batch_size, K, num_vertexes * 2)
    ver_coor[..., ::2] += xs.view(batch_size, K, 1).expand(batch_size, K, num_vertexes)
    ver_coor[..., 1::2] += ys.view(batch_size, K, 1).expand(batch_size, K, num_vertexes)
    ver_coor = ver_coor.view(batch_size, K, num_vertexes, 2).permute(0, 2, 1, 3).contiguous()  # b x num_vers x K x 2
    pure_ver_pos = ver_coor.unsqueeze(3).expand(batch_size, num_vertexes, K, K, 2)

    hm_ver = _nms(hm_ver)
    thresh = 0.1
    ver_score, ver_inds, ver_ys, ver_xs = _topk_channel(hm_ver, K=K)  # b x num_vertexes x K
    if ver_off is not None:
        ver_off = _transpose_and_gather_feat(ver_off, ver_inds.view(batch_size, -1))
        ver_off = ver_off.view(batch_size, num_vertexes, K, 2)
        ver_xs = ver_xs + ver_off[:, :, :, 0]
        ver_ys = ver_ys + ver_off[:, :, :, 1]
    else:
        ver_xs = ver_xs + 0.5
        ver_ys = ver_ys + 0.5

    mask = (ver_score > thresh).float()
    ver_score = (1 - mask) * -1 + mask * ver_score
    ver_ys = (1 - mask) * (-10000) + mask * ver_ys
    ver_xs = (1 - mask) * (-10000) + mask * ver_xs
    ver_pos = torch.stack([ver_xs, ver_ys], dim=-1).unsqueeze(2).expand(batch_size, num_vertexes, K, K, 2)
    # dist size: (batch_size, num_vertexes, K, K, 2) --> (batch_size, num_vertexes, K, K)
    dist = (((pure_ver_pos - ver_pos) ** 2).sum(dim=4) ** 0.5)
    min_dist, min_ind = dist.min(dim=3)  # b x num_vertexes x K
    ver_score = ver_score.gather(2, min_ind).unsqueeze(-1)  # b x num_vertexes x K x 1
    min_dist = min_dist.unsqueeze(-1)
    min_ind = min_ind.view(batch_size, num_vertexes, K, 1, 1).expand(batch_size, num_vertexes, K, 1, 2)
    ver_pos = ver_pos.gather(3, min_ind)
    ver_pos = ver_pos.view(batch_size, num_vertexes, K, 2)

    hm_h, hm_w = hm_size
    dummy_h = torch.ones(size=(batch_size, num_vertexes, K, 1), device=device, dtype=torch.float) * hm_h
    dummy_w = torch.ones(size=(batch_size, num_vertexes, K, 1), device=device, dtype=torch.float) * hm_w

    mask = (ver_pos[..., 0:1] < 0) + (ver_pos[..., 0:1] > hm_w) + \
           (ver_pos[..., 1:2] < 0) + (ver_pos[..., 1:2] > hm_h) + \
           (ver_score < thresh) + (min_dist > (torch.max(dummy_h, dummy_w) * 0.3))
    mask = (mask > 0).float().expand(batch_size, num_vertexes, K, 2)
    ver_coor = (1 - mask) * ver_pos + mask * ver_coor
    ver_coor = ver_coor.permute(0, 2, 1, 3).contiguous().view(batch_size, K, num_vertexes * 2)

    # (scores x 1, xs x 1, ys x 1, wh x 2, bboxes x 4, ver_coor x 16, rot x 8, depth x 1, dim x 3, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, wh-3:5, bboxes-5:9, ver_coor-9:25, rot-25:33, depth-33:34, dim-34:37, clses-37:38)
    # detections: [batch_size, K, 38]
    detections = torch.cat([scores, xs, ys, wh, bboxes, ver_coor, rot, depth, dim, clses], dim=2)

    return detections


def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def get_pred_depth(depth):
    return depth


def post_processing_2d(detections, num_classes=3, down_ratio=4):
    """

    :param detections: [batch_size, K, 38]
    # (scores x 1, xs x 1, ys x 1, wh x 2, bboxes x 4, ver_coor x 16, rot x 8, depth x 1, dim x 3, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, wh-3:5, bboxes-5:9, ver_coor-9:25, rot-25:33, depth-33:34, dim-34:37, clses-37:38)
    :param conf_thresh:
    :return:
    """
    # TODO: Need to consider rescale to the original scale: bbox, xs, ys, and ver_coor - 1:25

    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j] = np.concatenate([
                detections[i, inds, :1].astype(np.float32),
                detections[i, inds, 1:25].astype(np.float32) * down_ratio,
                get_alpha(detections[i, inds, 25:33])[:, np.newaxis].astype(np.float32),
                get_pred_depth(detections[i, inds, 33:34]).astype(np.float32),
                detections[i, inds, 34:37].astype(np.float32)], axis=1)
        ret.append(top_preds)

    return ret


def get_final_pred(detections, num_classes=3, peak_thresh=0.2):
    for j in range(num_classes):
        if len(detections[j] > 0):
            keep_inds = (detections[j][:, 0] > peak_thresh)
            detections[j] = detections[j][keep_inds]

    return detections


def draw_predictions(img, detections, colors, num_classes=3, show_3dbox=False):
    for j in range(num_classes):
        if len(detections[j] > 0):
            for det in detections[j]:
                # (scores-0:1, xs-1:2, ys-2:3, wh-3:5, bboxes-5:9, ver_coor-9:25, rot-25:26, depth-26:27, dim-27:30)
                _score = det[0]
                _x, _y, _wh, _bbox, _ver_coor = int(det[1]), int(det[2]), det[3:5], det[5:9], det[9:25]
                _rot, _depth, _dim = det[25], det[26], det[27:30]
                _bbox = np.array(_bbox, dtype=int)
                if not show_3dbox:
                    img = cv2.rectangle(img, (_bbox[0], _bbox[1]), (_bbox[2], _bbox[3]), colors[-j - 1], 2)
                if show_3dbox:
                    _ver_coor = np.array(_ver_coor, dtype=int).reshape(-1, 2)
                    img = draw_box_3d(img, _ver_coor, color=colors[j])
                # print('_depth: {:.2f}n, _dim: {}, _rot: {:.2f} radian'.format(_depth, _dim, _rot))

    return img


def post_processing_3d(detections, conf_thresh=0.95):
    """

    """
    pass


def draw_box_3d(image, corners, color=(0, 0, 255)):
    ''' Draw 3d bounding box in image
        corners: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    '''

    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                     (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]), color, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                     (corners[f[2], 0], corners[f[2], 1]), color, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                     (corners[f[3], 0], corners[f[3], 1]), color, 1, lineType=cv2.LINE_AA)

    return image

def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)

if __name__ == '__main__':

    
    img_path = "/home/dorleco/RTM3D/dataset/kitti/testing/image_2/000010.png"

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    rtm3d = RTM3D()

    rtm3d.detect(img)


import math
import pdb
import torch.nn.functional as F
from torch import nn
import torch

from .mobilenetv2 import mobile_half
from .shufflenetv1 import ShuffleV1
from .shufflenetv2 import ShuffleV2
from .resnet_cifar import build_resnet_backbone, build_resnetx4_backbone
from .vgg import build_vgg_backbone
from .wide_resnet_cifar import wrn

import torch
from torch import nn
import torch.nn.functional as F

from .resnet  import *

class ABF(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=4):
        super(ABF, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//2,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel//2, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[3].weight, a=1)  # pyre-ignore

    def forward(self, x, shape=None, out_shape=None):

        for idx, x_item in enumerate(x):
            x_tmp = F.interpolate(x_item, (shape, shape), mode="nearest")
            if idx == 0:
                y = x_tmp
            else:
                y = torch.cat([y, x_tmp], dim=1)

        y_mix = self.conv1(y)
        # output
        if y_mix.shape[-1] != out_shape:
            y_mix = F.interpolate(y_mix, (out_shape, out_shape), mode="nearest")
        z = self.conv2(y_mix)
        return z

class ABF_wide(nn.Module):
    def __init__(self, in_channel, out_channel, in_channels, shrink):
        super(ABF_wide, self).__init__()

        self.conv_refine0 = nn.Sequential(nn.Conv2d(in_channels[0], in_channels[0]//shrink, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(in_channels[0]//shrink),
        )
        nn.init.kaiming_uniform_(self.conv_refine0[0].weight, a = 1)

        self.conv_refine1 = nn.Sequential(nn.Conv2d(in_channels[1], in_channels[1]//shrink, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(in_channels[1]//shrink),
                                          )
        nn.init.kaiming_uniform_(self.conv_refine1[0].weight, a=1)

        self.conv_refine2 = nn.Sequential(nn.Conv2d(in_channels[2], in_channels[2]//shrink, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(in_channels[2]//shrink),
                                          )
        nn.init.kaiming_uniform_(self.conv_refine2[0].weight, a=1)

        self.conv_refine3 = nn.Sequential(nn.Conv2d(in_channels[3], in_channels[3]//shrink, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(in_channels[3]//shrink),
                                          )
        nn.init.kaiming_uniform_(self.conv_refine3[0].weight, a=1)

        if len(in_channels) > 4:
            self.conv_refine4 = nn.Sequential(nn.Conv2d(in_channels[4], in_channels[4]//shrink, kernel_size=1, bias=False),
                                              nn.BatchNorm2d(in_channels[4]//shrink),
                                              )
            nn.init.kaiming_uniform_(self.conv_refine4[0].weight, a=1)


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, shape=None, out_shape=None):

        for idx, x_item in enumerate(x):
            if len(x) > 4:
                if idx == 4:
                    x_item = self.conv_refine0(x_item)
                elif idx == 3:
                    x_item = self.conv_refine1(x_item)
                elif idx == 2:
                    x_item = self.conv_refine2(x_item)
                elif idx == 1:
                    x_item = self.conv_refine3(x_item)
                elif idx == 0:
                    x_item = self.conv_refine4(x_item)
            else:
                if idx == 3:
                    x_item = self.conv_refine0(x_item)
                elif idx == 2:
                    x_item = self.conv_refine1(x_item)
                elif idx == 1:
                    x_item = self.conv_refine2(x_item)
                elif idx == 0:
                    x_item = self.conv_refine3(x_item)

            x_tmp = F.interpolate(x_item, (shape, shape), mode="nearest")

            if idx == 0:
                y = x_tmp
            else:
                y = torch.cat([y, x_tmp], dim=1)

        y_mix = self.conv1(y)

        # output
        if y_mix.shape[-1] != out_shape:
            y_mix = F.interpolate(y_mix, (out_shape, out_shape), mode="nearest")

        z = self.conv2(y_mix)

        return z

class PatchKD(nn.Module):
    def __init__(
        self, student, in_channels, out_channels, shapes, out_shapes, shrink,
    ):
        super(PatchKD, self).__init__()
        self.student = student
        self.shapes = shapes
        self.out_shapes = shapes if out_shapes is None else out_shapes

        abfs = nn.ModuleList()
        # in_channel = sum(in_channels[:])
        if shrink is not None:
            in_channel = 0
            for in_num in range(len(in_channels)):
                in_channel = in_channel + in_channels[in_num]//shrink
        else:
            in_channel = sum(in_channels[:])

        for idx in range(len(in_channels)):
            # if in_channels[0] <= 16 and in_channels[-1] <= 64:
            if shrink is None:
                abfs.append(ABF(in_channel, out_channels[idx]))
            else:
                abfs.append(ABF_wide(in_channel, out_channels[idx], in_channels, shrink))

        self.abfs = abfs[::-1]
        self.to('cuda')

    def forward(self, x):
        student_features = self.student(x, is_feat=True)
        logit = student_features[1]
        x = student_features[0][::-1]
        results = []
        for layer_num, abf, shape, out_shape in zip(range(len(x[0:])), self.abfs[0:], self.shapes[0:], self.out_shapes[0:]):
            out_features = abf(x[0:len(self.abfs[0:])], shape, out_shape)
            results.insert(0, out_features)

        return results, logit

def build_den_patch_kd(model, num_classes, teacher = '', shrink_input = None):
    out_shapes = None
    if 'x4' in model:
        student = build_resnetx4_backbone(depth = int(model[6:-2]), num_classes = num_classes)
        in_channels = [64,128,256,256]
        out_channels = [64,128,256,256]
        shapes = [1,8,16,32]
        shrink = None
    elif 'ResNet50' in model:
        student = ResNet50(num_classes = num_classes)
        in_channels = [16,32,64,64]
        out_channels = [16,32,64,64]
        shapes = [1,8,16,32,32]
        assert False
    elif 'resnet' in model:
        student = build_resnet_backbone(depth = int(model[6:]), num_classes = num_classes)
        in_channels = [16,32,64,64]
        out_channels = [16,32,64,64]
        shapes = [1,8,16,32,32]
        shrink = None
        # shrink = 2
    elif 'vgg' in model:
        student = build_vgg_backbone(depth = int(model[3:]), num_classes = num_classes)
        in_channels = [128,256,512,512,512]
        shapes = [1,4,4,8,16]
        if 'ResNet50' in teacher:
            out_channels = [256,512,1024,2048,2048]
            out_shapes = [1,4,8,16,32]
        else:
            out_channels = [128,256,512,512,512]
            # shrink = 4
            # TODO: add by zhanghn, 2024-5-4
            shrink = shrink_input
    elif 'mobile' in model:
        student = mobile_half(num_classes = num_classes)
        in_channels = [12,16,48,160,1280]
        shapes = [1,2,4,8,16]
        if 'ResNet50' in teacher:
            out_channels = [256,512,1024,2048,2048]
            out_shapes = [1,4,8,16,32]
            # shrink = 2
            # TODO: add by zhanghn, 2024-5-4
            shrink = shrink_input
        else:
            out_channels = [128,256,512,512,512]
            out_shapes = [1,4,4,8,16]
            # shrink = 4
            # TODO: add by zhanghn, 2024-5-4
            shrink = shrink_input
    elif 'shufflev1' in model:
        student = ShuffleV1(num_classes = num_classes)
        in_channels = [240,480,960,960]
        shapes = [1,4,8,16]
        if 'wrn' in teacher:
            out_channels = [32,64,128,128]
            out_shapes = [1,8,16,32]
            # shrink = 4
            # TODO: add by zhanghn, 2024-5-4
            shrink = shrink_input
        else:
            out_channels = [64,128,256,256]
            out_shapes = [1,8,16,32]
            # shrink = 4
            # TODO: add by zhanghn, 2024-5-4
            shrink = shrink_input
    elif 'shufflev2' in model:
        student = ShuffleV2(num_classes = num_classes)
        in_channels = [116,232,464,1024]
        shapes = [1,4,8,16]
        out_channels = [64,128,256,256]
        out_shapes = [1,8,16,32]
        # shrink = 4
        # TODO: add by zhanghn, 2024-5-4
        shrink = shrink_input
    elif 'wrn' in model:
        student = wrn(depth=int(model[4:6]), widen_factor=int(model[-1:]), num_classes=num_classes)
        r=int(model[-1:])
        in_channels = [16*r,32*r,64*r,64*r]
        out_channels = [32,64,128,128]
        shapes = [1,8,16,32]

        if r == 2:
            # shrink = 2
            # TODO: add by zhanghn, 2024-5-4
            shrink = shrink_input
        else:
            shrink = None
    else:
        assert False

    backbone = PatchKD(
        student=student,
        in_channels=in_channels,
        out_channels=out_channels,
        shapes = shapes,
        out_shapes = out_shapes,
        shrink = shrink
    )
    return backbone

def attention(teacher, x, ori_tfea, patch_size = 8, mode = 'cosin'):

    n, c, w, h = x.shape
    patch_num = int((w/patch_size) ** 2)

    # t_features, _ = teacher(x, is_feat=True, preact=True)
    # t_feature = t_features[-1]
    n_pen, c_pen, _, _ = ori_tfea.shape
    t_fea_tmp = ori_tfea.squeeze(-1).squeeze(-1)

    attention_map = torch.ones_like(x).cuda()
    x_mask_all = torch.ones(n*patch_num,c,w,h).cuda()
    i = 1
    j = 1
    for iter in range(1, patch_num+1):
        """ Generate mask """
        mask = torch.zeros_like(x).cuda()
        mask[:,:, patch_size*(i-1):patch_size*i, patch_size*(j-1):patch_size*j] = 1
        """ mask the image """
        x_mask_all[n*(iter-1):n*iter,:,:,:] = x * mask
        j += 1
        if iter % int(w/patch_size) == 0:
            i += 1
            j = 1

    with torch.no_grad():
        teacher.eval()
        t_features_masked, _ = teacher(x_mask_all, is_feat=True, preact=True)
    t_feature_masked = t_features_masked[-1]

    i = 1
    j = 1
    # tfea_masked_dist_dic = torch.zeros(n_pen, patch_num).cuda()
    for iter in range(1, patch_num+1):
        t_fea_masked_tmp = t_feature_masked[n*(iter-1):n*iter,:,:,:].squeeze(-1).squeeze(-1)
        ''' distance between masked and un-masked '''
        ''' by myself '''
        if mode == 'kl':
            distance = (F.softmax(t_fea_tmp, dim=-1)*torch.log(F.softmax(t_fea_tmp, dim=-1)
                                                               /F.softmax(t_fea_masked_tmp, dim=-1))).sum(-1)
        elif mode == 'mse':
            dist_tmp = (t_fea_masked_tmp - t_fea_tmp) ** 2
            distance = dist_tmp.sum(-1)/c_pen
        elif mode == 'cosin':
            distance = torch.exp(torch.bmm(F.normalize(t_fea_tmp).unsqueeze(-2),F.normalize(t_fea_masked_tmp).unsqueeze(-1)))
            distance = distance.view(-1)
        else:
            assert False

        '''resize to attention map'''
        attention_map[:, :, patch_size*(i-1):patch_size*i, patch_size*(j-1):patch_size*j] = distance.view(-1, 1, 1, 1)
        j += 1
        if iter % int(w/patch_size) == 0:
            i += 1
            j = 1

    atten_tmp = attention_map[:, 0, :, :].unsqueeze(1)
    atten_tmp_min, _ = atten_tmp.reshape(n, 1, w * h).min(dim=-1, keepdim=True)
    atten_tmp_min = atten_tmp_min.unsqueeze(-1)
    atten_tmp_max, _ = atten_tmp.reshape(n, 1, w * h).max(dim=-1, keepdim=True)
    atten_tmp_max = atten_tmp_max.unsqueeze(-1)
    out_atten = (atten_tmp - atten_tmp_min) / (atten_tmp_max - atten_tmp_min)

    return out_atten

def attention_overlap(teacher, x, ori_tfea, patch_size, stride, mode = 'cosin'):

    n, c, w, h = x.shape
    n_pen, c_pen, _, _ = ori_tfea.shape
    t_fea_tmp = ori_tfea.squeeze(-1).squeeze(-1)

    if patch_size == 16 and stride == 8:
        x_mask_all = torch.ones(n * 9, c, w, h).cuda()
        str_i = 1
        str_j = 1
        for iter in range(1, 10):
            """ Generate mask """
            mask = torch.zeros_like(x).cuda()
            row_l = stride * (str_i - 1)
            row_r = row_l + patch_size
            col_l = stride * (str_j - 1)
            col_r = col_l + patch_size

            mask[:, :, row_l:row_r, col_l:col_r] = 1
            """ mask the image """
            x_mask_all[n * (iter - 1):n * iter, :, :, :] = x * mask
            str_j += 1
            if iter % 3 == 0:
                str_i += 1
                str_j = 1

        with torch.no_grad():
            teacher.eval()
            t_features_masked, _ = teacher(x_mask_all, is_feat=True, preact=True)
        t_feature_masked = t_features_masked[-1]

        attention_map = torch.zeros_like(x).cuda()
        attention_map_mask = torch.zeros_like(x).cuda()
        i = 1
        j = 1
        # tfea_masked_dist_dic = torch.zeros(n_pen, patch_num).cuda()
        for iter in range(1, 10):
            cur_map = torch.zeros_like(x).cuda()
            cur_mask = torch.zeros_like(x).cuda()
            t_fea_masked_tmp = t_feature_masked[n * (iter - 1):n * iter, :, :, :].squeeze(-1).squeeze(-1)
            ''' distance between masked and un-masked '''
            ''' by myself '''
            if mode == 'kl':
                distance = (F.softmax(t_fea_tmp, dim=-1) * torch.log(F.softmax(t_fea_tmp, dim=-1)
                                                                     / F.softmax(t_fea_masked_tmp, dim=-1))).sum(-1)
            elif mode == 'mse':
                dist_tmp = (t_fea_masked_tmp - t_fea_tmp) ** 2
                distance = dist_tmp.sum(-1) / c_pen
            elif mode == 'cosin':
                distance = torch.exp(
                    torch.bmm(F.normalize(t_fea_tmp).unsqueeze(-2), F.normalize(t_fea_masked_tmp).unsqueeze(-1)))
                distance = distance.view(-1)
            else:
                assert False

            '''resize to attention map'''
            row_l = stride * (i - 1)
            row_r = row_l + patch_size
            col_l = stride * (j - 1)
            col_r = col_l + patch_size
            cur_map[:, :, row_l:row_r, col_l:col_r] = distance.view(-1, 1, 1, 1)
            cur_mask[:, :, row_l:row_r, col_l:col_r] = 1.0
            attention_map += cur_map
            attention_map_mask += cur_mask
            j += 1
            if iter % 3 == 0:
                i += 1
                j = 1
        attention_map = attention_map / attention_map_mask

        atten_tmp = attention_map[:, 0, :, :].unsqueeze(1)
        atten_tmp_min, _ = atten_tmp.reshape(n, 1, w * h).min(dim=-1, keepdim=True)
        atten_tmp_min = atten_tmp_min.unsqueeze(-1)
        atten_tmp_max, _ = atten_tmp.reshape(n, 1, w * h).max(dim=-1, keepdim=True)
        atten_tmp_max = atten_tmp_max.unsqueeze(-1)
        out_atten = (atten_tmp - atten_tmp_min) / (atten_tmp_max - atten_tmp_min)

    elif patch_size == 16 and stride == 4:
        x_mask_all = torch.ones(n * 25, c, w, h).cuda()
        str_i = 1
        str_j = 1
        for iter in range(1, 26):
            """ Generate mask """
            mask = torch.zeros_like(x).cuda()
            row_l = stride * (str_i - 1)
            row_r = row_l + patch_size
            col_l = stride * (str_j - 1)
            col_r = col_l + patch_size

            mask[:, :, row_l:row_r, col_l:col_r] = 1
            """ mask the image """
            x_mask_all[n * (iter - 1):n * iter, :, :, :] = x * mask
            str_j += 1
            if iter % 5 == 0:
                str_i += 1
                str_j = 1

        with torch.no_grad():
            teacher.eval()
            t_features_masked, _ = teacher(x_mask_all, is_feat=True, preact=True)
        t_feature_masked = t_features_masked[-1]

        attention_map = torch.zeros_like(x).cuda()
        attention_map_mask = torch.zeros_like(x).cuda()
        i = 1
        j = 1
        # tfea_masked_dist_dic = torch.zeros(n_pen, patch_num).cuda()
        for iter in range(1, 26):
            cur_map = torch.zeros_like(x).cuda()
            cur_mask = torch.zeros_like(x).cuda()
            t_fea_masked_tmp = t_feature_masked[n * (iter - 1):n * iter, :, :, :].squeeze(-1).squeeze(-1)
            ''' distance between masked and un-masked '''
            ''' by myself '''
            if mode == 'kl':
                distance = (F.softmax(t_fea_tmp, dim=-1) * torch.log(F.softmax(t_fea_tmp, dim=-1)
                                                                     / F.softmax(t_fea_masked_tmp, dim=-1))).sum(-1)
            elif mode == 'mse':
                dist_tmp = (t_fea_masked_tmp - t_fea_tmp) ** 2
                distance = dist_tmp.sum(-1) / c_pen
            elif mode == 'cosin':
                distance = torch.exp(
                    torch.bmm(F.normalize(t_fea_tmp).unsqueeze(-2), F.normalize(t_fea_masked_tmp).unsqueeze(-1)))
                distance = distance.view(-1)
            else:
                assert False

            '''resize to attention map'''
            row_l = stride * (i - 1)
            row_r = row_l + patch_size
            col_l = stride * (j - 1)
            col_r = col_l + patch_size
            cur_map[:, :, row_l:row_r, col_l:col_r] = distance.view(-1, 1, 1, 1)
            cur_mask[:, :, row_l:row_r, col_l:col_r] = 1.0
            attention_map += cur_map
            attention_map_mask += cur_mask
            j += 1
            if iter % 5 == 0:
                i += 1
                j = 1
        attention_map = attention_map / attention_map_mask

        atten_tmp = attention_map[:, 0, :, :].unsqueeze(1)
        atten_tmp_min, _ = atten_tmp.reshape(n, 1, w * h).min(dim=-1, keepdim=True)
        atten_tmp_min = atten_tmp_min.unsqueeze(-1)
        atten_tmp_max, _ = atten_tmp.reshape(n, 1, w * h).max(dim=-1, keepdim=True)
        atten_tmp_max = atten_tmp_max.unsqueeze(-1)
        out_atten = (atten_tmp - atten_tmp_min) / (atten_tmp_max - atten_tmp_min)

    elif patch_size == 24 and stride == 8:
        x_mask_all = torch.ones(n * 4, c, w, h).cuda()
        str_i = 1
        str_j = 1
        for iter in range(1, 5):
            """ Generate mask """
            mask = torch.zeros_like(x).cuda()
            row_l = stride * (str_i - 1)
            row_r = row_l + patch_size
            col_l = stride * (str_j - 1)
            col_r = col_l + patch_size

            mask[:, :, row_l:row_r, col_l:col_r] = 1
            """ mask the image """
            x_mask_all[n * (iter - 1):n * iter, :, :, :] = x * mask
            str_j += 1
            if iter % 2 == 0:
                str_i += 1
                str_j = 1

        with torch.no_grad():
            teacher.eval()
            t_features_masked, _ = teacher(x_mask_all, is_feat=True, preact=True)
        t_feature_masked = t_features_masked[-1]

        attention_map = torch.zeros_like(x).cuda()
        attention_map_mask = torch.zeros_like(x).cuda()
        i = 1
        j = 1
        # tfea_masked_dist_dic = torch.zeros(n_pen, patch_num).cuda()
        for iter in range(1, 5):
            cur_map = torch.zeros_like(x).cuda()
            cur_mask = torch.zeros_like(x).cuda()
            t_fea_masked_tmp = t_feature_masked[n * (iter - 1):n * iter, :, :, :].squeeze(-1).squeeze(-1)
            ''' distance between masked and un-masked '''
            ''' by myself '''
            if mode == 'kl':
                distance = (F.softmax(t_fea_tmp, dim=-1) * torch.log(F.softmax(t_fea_tmp, dim=-1)
                                                                     / F.softmax(t_fea_masked_tmp, dim=-1))).sum(-1)
            elif mode == 'mse':
                dist_tmp = (t_fea_masked_tmp - t_fea_tmp) ** 2
                distance = dist_tmp.sum(-1) / c_pen
            elif mode == 'cosin':
                distance = torch.exp(
                    torch.bmm(F.normalize(t_fea_tmp).unsqueeze(-2), F.normalize(t_fea_masked_tmp).unsqueeze(-1)))
                distance = distance.view(-1)
            else:
                assert False

            '''resize to attention map'''
            row_l = stride * (i - 1)
            row_r = row_l + patch_size
            col_l = stride * (j - 1)
            col_r = col_l + patch_size
            cur_map[:, :, row_l:row_r, col_l:col_r] = distance.view(-1, 1, 1, 1)
            cur_mask[:, :, row_l:row_r, col_l:col_r] = 1.0
            attention_map += cur_map
            attention_map_mask += cur_mask
            j += 1
            if iter % 2 == 0:
                i += 1
                j = 1
        attention_map = attention_map / attention_map_mask

        atten_tmp = attention_map[:, 0, :, :].unsqueeze(1)
        atten_tmp_min, _ = atten_tmp.reshape(n, 1, w * h).min(dim=-1, keepdim=True)
        atten_tmp_min = atten_tmp_min.unsqueeze(-1)
        atten_tmp_max, _ = atten_tmp.reshape(n, 1, w * h).max(dim=-1, keepdim=True)
        atten_tmp_max = atten_tmp_max.unsqueeze(-1)
        out_atten = (atten_tmp - atten_tmp_min) / (atten_tmp_max - atten_tmp_min)
        

    elif patch_size == 24 and stride == 4:
        x_mask_all = torch.ones(n * 9, c, w, h).cuda()
        str_i = 1
        str_j = 1
        for iter in range(1, 10):
            """ Generate mask """
            mask = torch.zeros_like(x).cuda()
            row_l = stride * (str_i - 1)
            row_r = row_l + patch_size
            col_l = stride * (str_j - 1)
            col_r = col_l + patch_size

            mask[:, :, row_l:row_r, col_l:col_r] = 1
            """ mask the image """
            x_mask_all[n * (iter - 1):n * iter, :, :, :] = x * mask
            str_j += 1
            if iter % 3 == 0:
                str_i += 1
                str_j = 1

        with torch.no_grad():
            teacher.eval()
            t_features_masked, _ = teacher(x_mask_all, is_feat=True, preact=True)
        t_feature_masked = t_features_masked[-1]

        attention_map = torch.zeros_like(x).cuda()
        attention_map_mask = torch.zeros_like(x).cuda()
        i = 1
        j = 1
        # tfea_masked_dist_dic = torch.zeros(n_pen, patch_num).cuda()
        for iter in range(1, 10):
            cur_map = torch.zeros_like(x).cuda()
            cur_mask = torch.zeros_like(x).cuda()
            t_fea_masked_tmp = t_feature_masked[n * (iter - 1):n * iter, :, :, :].squeeze(-1).squeeze(-1)
            ''' distance between masked and un-masked '''
            ''' by myself '''
            if mode == 'kl':
                distance = (F.softmax(t_fea_tmp, dim=-1) * torch.log(F.softmax(t_fea_tmp, dim=-1)
                                                                     / F.softmax(t_fea_masked_tmp, dim=-1))).sum(-1)
            elif mode == 'mse':
                dist_tmp = (t_fea_masked_tmp - t_fea_tmp) ** 2
                distance = dist_tmp.sum(-1) / c_pen
            elif mode == 'cosin':
                distance = torch.exp(
                    torch.bmm(F.normalize(t_fea_tmp).unsqueeze(-2), F.normalize(t_fea_masked_tmp).unsqueeze(-1)))
                distance = distance.view(-1)
            else:
                assert False

            '''resize to attention map'''
            row_l = stride * (i - 1)
            row_r = row_l + patch_size
            col_l = stride * (j - 1)
            col_r = col_l + patch_size
            cur_map[:, :, row_l:row_r, col_l:col_r] = distance.view(-1, 1, 1, 1)
            cur_mask[:, :, row_l:row_r, col_l:col_r] = 1.0
            attention_map += cur_map
            attention_map_mask += cur_mask
            j += 1
            if iter % 3 == 0:
                i += 1
                j = 1
        attention_map = attention_map / attention_map_mask

        atten_tmp = attention_map[:, 0, :, :].unsqueeze(1)
        atten_tmp_min, _ = atten_tmp.reshape(n, 1, w * h).min(dim=-1, keepdim=True)
        atten_tmp_min = atten_tmp_min.unsqueeze(-1)
        atten_tmp_max, _ = atten_tmp.reshape(n, 1, w * h).max(dim=-1, keepdim=True)
        atten_tmp_max = atten_tmp_max.unsqueeze(-1)
        out_atten = (atten_tmp - atten_tmp_min) / (atten_tmp_max - atten_tmp_min)

    return out_atten


def instance_imp(t_pred, label, temp = 1.0):
    l_logsoftmax = nn.LogSoftmax(dim=1)
    l_nll = nn.NLLLoss(reduction='none')

    t_pred_softmax = l_logsoftmax(t_pred)
    dist_ce = l_nll(t_pred_softmax, label)

    inst_imp = torch.exp(-dist_ce/temp)/(torch.exp(-dist_ce/temp).sum()) * len(label)

    return inst_imp

def instance_imp_reverse(t_pred, label, temp = 1.0, mode = 'dir'):
    l_logsoftmax = nn.LogSoftmax(dim=1)
    l_nll = nn.NLLLoss(reduction='none')

    t_pred_softmax = l_logsoftmax(t_pred)
    dist_ce = l_nll(t_pred_softmax, label)

    # inst_imp = torch.exp(-dist_ce/temp)/(torch.exp(-dist_ce/temp).sum()) * len(label)
    # TODO: change by zhanghn. 2024-5-4
    if mode == 'dir':
        inst_imp = torch.exp(dist_ce / temp) / (torch.exp(dist_ce / temp).sum()) * len(label)
    elif mode == 'reverse':
        inst_imp = torch.exp(-dist_ce / temp) / (torch.exp(-dist_ce / temp).sum()) * len(label)
        # print(inst_imp.max() - inst_imp.min())
        v, k = torch.sort(inst_imp)
        # print(inst_imp.sort())
        inst_imp_reverse = inst_imp
        k = torch.flip(k, [0])  
        inst_imp_reverse[k] = v
        inst_imp = inst_imp_reverse
        # print(inst_imp.sort())
        # exit()
        # print(inst_imp.max()-inst_imp.min())
        # exit()
    elif mode == 'prob':
        dist_ce = dist_ce - (dist_ce.max() + dist_ce.min())
        inst_imp = torch.exp(dist_ce / temp) / (torch.exp(dist_ce / temp).sum()) * len(label)
        # print(inst_imp.max() - inst_imp.min())
        # exit()
    elif mode == 'prob_refine':
        inst_imp_refine = torch.exp(-dist_ce / temp) / (torch.exp(-dist_ce / temp).sum()) * len(label)
        maxB, minB = inst_imp_refine.max(), inst_imp_refine.min()
        
        dist_ce = dist_ce - (dist_ce.max() + dist_ce.min())
        inst_imp = torch.exp(dist_ce / temp) / (torch.exp(dist_ce / temp).sum()) * len(label)
        
        # Refine
        maxA, minA = inst_imp.max(), inst_imp.min()
        inst_imp = ( (inst_imp - minA) / (maxA - minA) ) * (maxB - minB) + minB
        
        # print(inst_imp.max() - inst_imp.min())
        # exit()
    elif mode == 'dir_refine':
        inst_imp_refine = torch.exp(-dist_ce / temp) / (torch.exp(-dist_ce / temp).sum()) * len(label)
        maxB, minB = inst_imp_refine.max(), inst_imp_refine.min()
        
        inst_imp = torch.exp(dist_ce / temp) / (torch.exp(dist_ce / temp).sum()) * len(label)
        
        # Refine
        maxA, minA = inst_imp.max(), inst_imp.min()
        inst_imp = ( (inst_imp - minA) / (maxA - minA) ) * (maxB - minB) + minB
        
        # print(inst_imp.max() - inst_imp.min())
        # exit()
    elif mode == 'dirreverse':
        # print('aa')
        # exit()
        inst_imp = torch.exp(dist_ce / temp) / (torch.exp(dist_ce / temp).sum()) * len(label)
        # print(inst_imp.max() - inst_imp.min())
        v, k = torch.sort(inst_imp)
        inst_imp_reverse = inst_imp
        k = torch.flip(k, [0])
        inst_imp_reverse[k] = v
        inst_imp = inst_imp_reverse
        # print(inst_imp.max()-inst_imp.min())
        # exit()
    else:
        assert False

    return inst_imp

def patch_at_loss(fstudent, fteacher, attention_map, inst_imp):
    loss_all = 0.0
    inst_imp = inst_imp.reshape(len(inst_imp),1,1,1)

    for count, fs, ft in zip(range(len(fteacher)), fstudent, fteacher):
        n,c,h,w = fs.shape

        if count > 2:
            loss = F.mse_loss(inst_imp*fs, inst_imp*ft, reduction='mean')
        else:
            attention_map_resized = F.adaptive_avg_pool2d(attention_map, (h, w))
            attention_map_resized = attention_map_resized.repeat(1,c,1,1)
            loss = F.mse_loss(inst_imp*attention_map_resized*fs, inst_imp*attention_map_resized*ft, reduction='mean')

        loss_all = loss_all + loss

    return loss_all

def patch_at_loss_vanilla(fstudent, fteacher):
    loss_all = 0.0
    for count, fs, ft in zip(range(len(fteacher)), fstudent, fteacher):
        loss = F.mse_loss(fs, ft, reduction='mean')
        loss_all = loss_all + loss

    return loss_all

def patch_at_loss_instance(fstudent, fteacher, inst_imp):
    loss_all = 0.0
    inst_imp = inst_imp.reshape(len(inst_imp),1,1,1)

    for count, fs, ft in zip(range(len(fteacher)), fstudent, fteacher):
        n,c,h,w = fs.shape

        loss = F.mse_loss(inst_imp*fs, inst_imp*ft, reduction='mean')

        loss_all = loss_all + loss

    return loss_all   
    


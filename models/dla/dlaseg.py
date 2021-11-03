'''
Still shows several errors
DLA for Semantic Segmentation.
Code taken from:
https://github.com/ucbdrive/dla/blob/master/dla_up.py
'''
import torch
import torch.nn as nn
import numpy as np
import math
from ..helpers import Registry, group_norm, _fill_up_weights, _fill_fc_weights
from .dlafpn import dla34


# -----------------------------------------------------------------------------
# DLA models
# -----------------------------------------------------------------------------
DLA34DCN = {
    "levels": [1, 1, 1, 2, 2, 1],
    "channels": [16, 32, 64, 128, 256, 512],
    "block": "BasicBlock"
}

# -----------------------------------------------------------------------------

_STAGE_SPECS = Registry({
    "DLA-34-DCN": DLA34DCN,
})

BatchNorm = nn.BatchNorm2d

class IDAUp(nn.Module):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(c, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    BatchNorm(out_dim),
                    nn.ReLU(inplace=True))
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                    output_padding=0, groups=out_dim, bias=False)
                _fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim,
                          kernel_size=node_kernel, stride=1,
                          padding=node_kernel // 2, bias=False),
                BatchNorm(out_dim),
                nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class DLAUpSeg(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUpSeg, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        # self.in_features = ['dla3', 'dla4', 'dla5']
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        #scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
        return x

        

class DLASeg(nn.Module):
    '''
    Input arguments:
    base_name = for now can only use the default dla34.
    classes = the number of classes.
    cfg = for now can only use the default value.
    Return:
    light ver: 
    tuples of (softmax output of orientation, softmax output of depth, logits of orientation, logits of depth)
    '''
    def __init__(self, 
                base_name, 
                cfg, 
                down_ratio=4, 
                norm_func=nn.BatchNorm2d,
                light_ver=True):
        super(DLASeg, self).__init__()
        down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.light_ver = light_ver

        channels = [16, 32, 64, 128, 256, 512]
        # levels = [1, 1, 1, 2, 2, 1]

        pretrained = True if not cfg.TRAINING.CONTINUE else False
        self.base = dla34(None, pretrained=pretrained) 
        channels = self.base.channels # [16, 32, 64, 128, 256, 512] 
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUpSeg(channels=channels[self.first_level:], scales=scales)
        self.fc_rot = nn.Sequential(
            nn.Conv2d(channels[self.first_level], 256,
                kernel_size=5, padding='same', bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,
                kernel_size=3, padding='same', bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 6, 
                kernel_size=1, stride=1, 
                padding='same', bias=True)
        )
        if cfg.MODEL.HEAD.DEPTH_GAUSSIAN: depth_cls = cfg.MODEL.HEAD.DEPTH_BIN
        else: depth_cls = 1    
        self.fc_depth = nn.Sequential(
            nn.Conv2d(channels[self.first_level], 256,
                kernel_size=5, padding='same', bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,
                kernel_size=3, padding='same', bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, depth_cls, 
                kernel_size=1, stride=1, 
                padding=0, bias=True)
        )

        self.softmax = nn.LogSoftmax(dim=1)


        '''
        ========================== Head ==============================
        '''
        '''
        Semantic segmentation head for discrete orientation and depth
        '''
        # for m in self.fc_depth.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        '''
        Instance center/offset regression head
        '''
        self.with_center_regression = cfg.MODEL.HEAD.WITH_CENTER_REGRESSION
        if self.with_center_regression:
            self.fc_c_regres = nn.Sequential(
                                nn.Conv2d(channels[self.first_level], 256,
                                    kernel_size=5, padding='same', bias=False),
                                nn.GroupNorm(32, 256),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 64,
                                    kernel_size=3, padding='same', bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 2, 
                                    kernel_size=1, stride=1, 
                                    padding=0, bias=True)
                                    )
                                    
            _fill_fc_weights(self.fc_c_regres)
            self.fc_c_regres[-1].bias.data.fill_(-2.19)

        '''
        Heatmap/classification head
        '''
        self.fc_center = nn.Sequential(
                            nn.Conv2d(channels[self.first_level], 256,
                                kernel_size=3, padding='same', bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 1, 
                                kernel_size=1, stride=1, 
                                padding=0, bias=True))
        _fill_fc_weights(self.fc_center)
        self.fc_center[-1].bias.data.fill_(-2.19)

        '''
        3D bounding box dimension regression head
        '''
        self.fc_dim_3d = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], 256,
                        kernel_size=5, padding='same', bias=False),
                    nn.GroupNorm(32, 256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 3, 
                        kernel_size=1, stride=1, 
                        padding=0, bias=True))
        _fill_fc_weights(self.fc_dim_3d)
        

        '''
        2D to 3D bounding box center disparity
        '''
        self.fc_dispar = nn.Sequential(
                            nn.Conv2d(channels[self.first_level], 256,
                                kernel_size=3, padding='same', bias=False),
                            nn.GroupNorm(32, 256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 2, 
                                kernel_size=1, stride=1, 
                                padding=0, bias=True))

        # _fill_fc_weights(self.fc_dispar)

        self.dd_feat = nn.Sequential(
                            nn.Conv2d(channels[self.first_level], 256,
                                kernel_size=1, padding='same', bias=False),
                            nn.GroupNorm(64, 256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256,
                                kernel_size=3, padding='same', bias=False),
                            nn.GroupNorm(32, 256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, channels[self.first_level], 
                                kernel_size=5, stride=1, 
                                padding='same', bias=True),
                            nn.ReLU(inplace=True))
        # _fill_fc_weights(self.dd_feat)
        self.ctr_feat = nn.Sequential(
                            nn.Conv2d(channels[self.first_level], 256,
                                kernel_size=1, padding='same', bias=False),
                            nn.GroupNorm(32, 256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, channels[self.first_level], 
                                kernel_size=5, stride=1, 
                                padding='same', bias=True),
                            nn.ReLU(inplace=True))
        # _fill_fc_weights(self.ctr_feat)
        self.od_feat = nn.Sequential(
                            nn.Conv2d(channels[self.first_level], 256,
                                kernel_size=1, padding='same', bias=False),
                            nn.GroupNorm(32, 256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, channels[self.first_level], 
                                kernel_size=5, stride=1, 
                                padding='same', bias=True),
                            nn.ReLU(inplace=True))
        # _fill_fc_weights(self.od_feat)
                                


    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])
        # x = self.dla_up(x)

        x_dd = self.dd_feat(x)
        x_ctr = self.ctr_feat(x)
        x_od = self.od_feat(x)

        ret = {}
        
        # ret['c_reg'] = self.fc_c_regres(x)
        # ret['heatmap'] = self.fc_center(x)
        # ret['dim'] = self.fc_dim_3d(x)
        # ret['depth'] = self.fc_depth(x)
        # ret['orientation'] = self.fc_orientation(x)
        # ret['c_dis'] = self.fc_dispar(x)
        if self.with_center_regression:
            ret['c_reg'] = self.fc_c_regres(x_ctr)
        ret['heatmap'] = self.fc_center(x_ctr)
        ret['dim'] = self.fc_dim_3d(x_dd)
        ret['depth'] = self.fc_depth(x_dd)
        ret['orientation'] = self.fc_rot(x_od)
        ret['c_dis'] = self.fc_dispar(x_od)

        return ret
        # y_orientation = self.fc_n_orientation(x)
        # y_depth = self.fc_depth(x)
        # y_c_reg = self.fc_c_regres(x)

        # y_orientation = self.softmax(self.up_orientation(y_orientation))
        # y_depth = self.softmax(self.up_depth(y_depth))
        # y_c_reg = self.softmax(self.up_c_reg(y_c_reg))
        
        # y_center = self.fc_center(x)
        # y_dimension = self.fc_dim_3d(x)
        # y_disparity = self.fc_dispar(x)
        
        # return y_orientation, y_depth, y_c_reg, y_center, y_dimension, y_disparity

        # return dict(
        #     orientation=y_orientation, 
        #     depth=y_depth, 
        #     c_reg=y_c_reg, 
        #     heatmap=y_center, 
        #     dim=y_dimension, 
        #     c_dis=y_disparity
        #     )

        # y_orientation = self.softmax(self.up_orientation(y_orientation))
        # y_depth = self.softmax(self.up_depth(y_depth))

        # ret = {}
        # for head in self.heads:
        #     ret[head] = self.__getattr__(head)(x)
        # return [ret]

    # def optim_parameters(self, memo=None):
    #     for param in self.base.parameters():
    #         yield param
    #     for param in self.dla_up.parameters():
    #         yield param
    #     for param in self.fc_n_orientation.parameters():
    #         yield param
    #     for param in self.fc_depth.parameters():
    #         yield param
    #     for param in self.fc_c_regres.parameters():
    #         yield param
    #     for param in self.fc_center.parameters():
    #         yield param
    #     for param in self.fc_dim_3d.parameters():
    #         yield param
    #     for param in self.fc_dispar.parameters():
    #         yield param
        

def dla34up(**kwargs):
    model = DLASeg('dla34', **kwargs)
    return model

def build_models(cfg=None):
    if cfg is None:
        raise Exception('Error cfg is none!')
    model = dla34up(cfg=cfg)
    return model

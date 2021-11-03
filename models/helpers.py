import math

import torch
from torch import nn

def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module

def oneHot2Int(input):
    return torch.argmax(input, axis=1)

# def combineCenterandMasks(prediction):
#     centerxoff = 0.001 # (2/2048)
#     centeryoff = 0.002 # (2/1024)
#     centers = prediction['center']
#     objects = []
#     for img in prediction['mask']:
#         for center in centers:
#             index = torch.where((img[-2] in range(center.x -centerxoff, center.x +centerxoff)) & \
#                                  img[-1] in range(center.y -centeryoff, center.y +centeryoff))
#             maskInstance = torch.mean(img[index])
#             # depth  = maskInstance.depth # Find a way to get the average depth from these pixels
#             # orientation = maskInstance.orientation # Find a way to get the average orientation from these pixels
#             obj = center#, depth, orientation
#             objects.append(obj)
#     return objects

class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creating a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_module_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    '''

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn

def _fill_up_weights(up):
    # todo: we can replace math here?
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def _fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# -----------------------------------------------------------------------------
# Custom Layers
# -----------------------------------------------------------------------------
def _make_conv_level(in_channels, out_channels, num_convs, norm_func,
                     stride=1, dilation=1):
    """
    make conv layers based on its number.
    """
    modules = []
    for i in range(num_convs):
        modules.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride if i == 0 else 1,
                      padding=dilation, bias=False, dilation=dilation),
            norm_func(out_channels),
            nn.ReLU(inplace=True)])
        in_channels = out_channels

    return nn.Sequential(*modules)

def group_norm(out_channels, NUM_GROUPS=32):
    num_groups = NUM_GROUPS
    if out_channels % 32 == 0:
        return nn.GroupNorm(num_groups, out_channels)
    else:
        return nn.GroupNorm(num_groups // 2, out_channels)

def sigmoid_hm(hm_features):
    x = hm_features.sigmoid_()
    x = x.clamp(min=1e-4, max=1 - 1e-4)

    return x
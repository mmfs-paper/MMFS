from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator
import pdb
#from new_modules import Cat

count_ops = 0
count_params = 0


def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)
    ### ops_conv
    if type_name in ['Conv2d', 'Conv2dQuant', 'ConvTranspose2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        print('Layer:', layer)
        print('Number of parameters: %.3f M' % (delta_ops / 1e6))
        delta_params = get_layer_param(layer)

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        #delta_ops = x.numel()
        delta_ops = 0.0
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        #delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_ops = 0.0
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        #delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_ops = 0.0
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        #delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_ops = 0.0
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout', 'Upsample', 'Softmax']:
        delta_params = get_layer_param(layer)
        delta_ops = 0.0

    elif type_name in ['Upsample', 'Hardtanh', 'QuantizedHardtanh', 'MaxPool2d', 'Cat', 'AvgQuant']:
        delta_params = 0
        delta_ops = 0.0

    elif type_name in ['InstanceNorm2d', 'LeakyReLU', 'Tanh', 'ReflectionPad2d']:
        delta_params = get_layer_param(layer)
        delta_ops = 0.0
    ### unknown layer type

    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, H, W):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 3, H, W))

    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params

def calc_computation(pose_model, ch, H=224, W=224, DDP_device=None):
    list_conv = []
    list_hook = []
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_conv.append(flops)
    def calc(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, nn.Conv2d):# | isinstance(net, nn.ConvTranspose2d):
                list_hook.append(net.register_forward_hook(conv_hook))
            return
        for c in childrens:
            calc(c)
    calc(pose_model)
    if next(pose_model.parameters()).is_cuda:
        if DDP_device is None:
            y = pose_model((torch.ones(2, ch, H, W).cuda()))
        else:
            y = pose_model((torch.ones(2, ch, H, W).to(DDP_device)))
    else:
        y = pose_model((torch.ones(2, ch, H, W)))
    print('total_compuation:', sum(list_conv) / 2e6, 'M')
    [item.remove() for item in list_hook]

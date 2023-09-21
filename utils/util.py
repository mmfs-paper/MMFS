"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import torchvision
import sys
import shutil
import datetime


def check_path_is_static_data(path):
    last_extension = path.split(".")[-1]
    acceptable_extensions = ['png', 'jpg', 'jpeg', 'npy', 'npz']
    if last_extension.lower() in acceptable_extensions:
        return True
    return False

def check_path(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        print(e)


def tensor2file(input_image, file_path, ext_name):
    """Convert a tensor into a file.

    Parameters:
        input_image  --  the input image tensor
        file_path    --  the file path without extension name
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

    if image_numpy.shape[2] <= 3:
        image_numpy = image_numpy.astype(np.uint8)
        # save as image
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(file_path + '.' + ext_name)
    else:
        # save as numpy
        np.save(file_path + '.npy', image_numpy)


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        if image_numpy.shape[2] > 3: # clip to 3 channel
            print('Warning: the channel count of output image exceeds 3.')
            image_numpy = image_numpy[:,:,:3]
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def print_losses(epoch, iters, len_dataset, current_losses, average_losses, mode='training'):
    """print current losses on console"""
    if mode=='training':
        message = mode + ': (epoch: %d, iters: %d / %d) ' % (epoch, iters, len_dataset)
    else:
        message = mode + ': (epoch: %d) ' % epoch
    for k, v in current_losses.items():
        message += '%s: %.3f ' % (k, v)
    for k, v in average_losses.items():
        message += 'average %s: %.3f ' % (k, v.avg())
    message += datetime.datetime.now().strftime("%Y_%m_%d %H:%M:%S")
    print(message)  # print the message

def make_grid(model):
    """
    create an image grid to be visualized by tensorboard.
    """
    visuals = model.get_current_visuals()
    names, grids = [], []
    for name, img in visuals.items():
        names.append(name)
        grid = torchvision.utils.make_grid(img[:,:3,:,:], nrow=img.size()[0], normalize=True)
        grids.append(grid)
    return grids, names

class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.num_item = 0

    def update(self, value):
        self.sum += value
        self.num_item += 1

    def avg(self):
        return self.sum / self.num_item

    def clear(self):
        self.sum = 0
        self.num_item = 0


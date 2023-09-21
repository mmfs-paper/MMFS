import utils.augmentation as transforms
import sys
from PIL import Image
import numpy as np
import random
import cv2
import os


class Transforms():
    def __init__(self, config, input_grayscale_flag=False, output_grayscale_flag=False, method=Image.BICUBIC, convert=True):
        self.config = config
        self.input_grayscale_flag = input_grayscale_flag
        self.output_grayscale_flag = output_grayscale_flag
        self.method = method
        self.convert = convert
        self.transform_list = []

    def create_transforms_from_list(self, preprocess_list):
        if self.input_grayscale_flag:
            if self.output_grayscale_flag:
                self.transform_list.append(transforms.Grayscale())
            else:
                self.transform_list.append(transforms.Grayscale(1, 3))
        elif self.output_grayscale_flag:
            self.transform_list.append(transforms.Grayscale(3, 1))

        if 'resize' in preprocess_list:
            if self.config['dataset']['load_size'] < 10000:
                osize = [self.config['dataset']['load_size'], self.config['dataset']['load_size']]
            else:
                osize = [self.config['dataset']['load_size'] // 10000, self.config['dataset']['load_size'] % 10000]
            self.transform_list.append(transforms.Resize(osize, self.method))
        elif 'scale_width' in preprocess_list:
            self.transform_list.append(transforms.ScaleWidth(self.config['dataset']['load_size'], self.method))

        if 'crop' in preprocess_list:
            if 'crop_pos' in self.config['dataset']:
                self.transform_list.append(transforms.Crop(self.config['dataset']['crop_pos'], self.config['dataset']['crop_size']))
            else:
                self.transform_list.append(transforms.RandomCrop(self.config['dataset']['crop_size']))

        if 'add_lighting' in preprocess_list:
            self.transform_list.append(transforms.ColorJitter())

        if 'random_affine' in preprocess_list:
            self.transform_list.append(transforms.RandomAffine(20, translate=(0.2, 0.2), scale=(0.2, 0.2)))

        if 'random_rotate' in preprocess_list:
            self.transform_list.append(transforms.RandomRotation(20))

        if 'random_blur' in preprocess_list:
            self.transform_list.append(transforms.RandomBlur(0.2))

        if 'add_gauss_noise' in preprocess_list:
            self.transform_list.append(transforms.NoiseTransform("gauss"))
        if 'add_s&p_noise' in preprocess_list:
            self.transform_list.append(transforms.NoiseTransform("s&p"))
        if 'add_poisson_noise' in preprocess_list:
            self.transform_list.append(transforms.NoiseTransform("poisson"))
        if 'add_speckle_noise' in preprocess_list:
            self.transform_list.append(transforms.NoiseTransform("speckle"))
        if 'add_band_noise' in preprocess_list:
            self.transform_list.append(transforms.NoiseTransform("band"))

        if preprocess_list == 'none':
            self.transform_list.append(transforms.MakePower2(base=4, method=self.method))

        if not self.config['dataset']['no_flip']:
            if 'flip' in self.config['dataset']:
                self.transform_list.append(transforms.Flip(self.config['dataset']['flip']))
            else:
                self.transform_list.append(transforms.RandomHorizontalFlip())

        if self.convert:
            self.transform_list += [transforms.ToTensor()]
            if self.input_grayscale_flag:
                if self.output_grayscale_flag:
                    self.transform_list += [transforms.Normalize((0.5,), (0.5,))]
                else:
                    self.transform_list += [transforms.Normalize((0.5,), (0.5,), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            elif self.output_grayscale_flag:
                self.transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5,), (0.5,))]
            else:
                self.transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    def get_transforms(self):
        return self.transform_list

    def compose_transforms(self):
        return transforms.JointCompose(self.transform_list)


def check_create_shuffled_order(data_list, order):
    # returns the order used to shuffle all paired data
    if order is None:  # Does not perform shuffling. Return normal order.
        order = np.arange(0, len(data_list)).tolist()
    else:
        if not isinstance(order, list):  # order is -1, which means has not been created.
            order = np.arange(0, len(data_list)).tolist()  # create the shuffle order.
            random.shuffle(order)
        # otherwise shuffle order already exists and we do nothing.
    return order


def check_equal_length(list1, list2, data):
    if len(list1) != len(list2):
        print("different length in paired data types. Please double check your data.")
        print("length of current data type: ", len(list1))
        print("----------------current lengths for all data types-------------------")
        for k, v in data.items():
            print("%s:   %d" % (k, len(v)))
        sys.exit()

def check_img_loaded(path):
    img = cv2.imread(path)
    if img is None or img.size == 0:
        print("image loading failed for " + path + '. Please double check.')
        return False
    return True

def check_numpy_loaded(path):
    try:
        arr = np.load(path)
    except Exception as e:
        print("numpy loading failed for " + path + '. Please double check.')
        return False
    return True

# custom, paired, numpy_paired, unpaired, numpy_unpaired, landmark
def check_old_config_val_possible(old_style_config):
    for data_type in old_style_config['dataset']['data_type']:
        if data_type == 'custom':
            if old_style_config['dataset']['custom_val_data'] == {}:
                return False
        elif data_type == 'paired' or data_type == 'numpy_paired':
            keyword = ''.join(data_type.split('_'))
            filelist_not_exist = old_style_config['dataset']['paired_val_filelist'] == ''
            filefolders_not_exist = old_style_config['dataset']['paired_valA_folder'] == '' or \
                old_style_config['dataset']['paired_valB_folder'] == ''
            dataroot_contains_no_val_folders = not os.path.exists(
                os.path.join(old_style_config['dataset']['dataroot'], 'val' + keyword + 'A')) \
                                              or not os.path.exists(
                os.path.join(old_style_config['dataset']['dataroot'], 'val' + keyword + 'B'))
            if filelist_not_exist and filefolders_not_exist and dataroot_contains_no_val_folders:
                return False
        elif data_type == 'unpaired' or data_type == 'numpy_unpaired':
            keyword = ''.join(data_type.split('_'))
            filelist_not_exist = old_style_config['dataset']['unpaired_valA_filelist'] == '' or \
                                 old_style_config['dataset']['unpaired_valB_filelist'] == ''
            filefolders_not_exist = old_style_config['dataset']['unpaired_valA_folder'] == '' or \
                                    old_style_config['dataset']['unpaired_valB_folder'] == ''
            dataroot_contains_no_val_folders = not os.path.exists(
                os.path.join(old_style_config['dataset']['dataroot'], 'val' + keyword + 'A')) \
                                               or not os.path.exists(
                os.path.join(old_style_config['dataset']['dataroot'], 'val' + keyword + 'B'))
            if filelist_not_exist and filefolders_not_exist and dataroot_contains_no_val_folders:
                return False
        elif data_type == 'landmark':
            filelist_not_exist = old_style_config['dataset']['paired_val_filelist'] == ''
            filefolders_not_exist = old_style_config['dataset']['paired_valA_folder'] == '' or \
                old_style_config['dataset']['paired_valB_folder'] == '' or \
                not os.path.exists(old_style_config['dataset']['paired_valA_lmk_folder']) or \
                not os.path.exists(old_style_config['dataset']['paired_valB_lmk_folder'])
            dataroot_contains_no_val_folders = not os.path.exists(
                os.path.join(old_style_config['dataset']['dataroot'], 'valpairedA_lmk')) \
                                               or not os.path.exists(
                os.path.join(old_style_config['dataset']['dataroot'], 'valpairedB_lmk'))
            if filelist_not_exist and filefolders_not_exist and dataroot_contains_no_val_folders:
                return False

    return True

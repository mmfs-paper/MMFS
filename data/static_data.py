import os, sys
import random
import numpy as np
from utils.augmentation import ImagePathToImage, NumpyToTensor
from utils.data_utils import Transforms
from utils.util import check_path_is_static_data
import torch
from PIL import Image


def check_dataname_folder_correspondence(data_names, group, group_name):
    for data_name in data_names:
        if data_name + '_folder' not in group:
            print("%s not found in config file. Going to use dataroot mode to load group %s." % (data_name + '_folder', group_name))
            return False
    return True


def custom_check_path_exists(str1):
    return True if (str1 == "None" or os.path.exists(str1)) else False


def custom_getsize(str1):
    return 1 if str1 == "None" else os.path.getsize(str1)


def check_different_extension_path_exists(str1):
    acceptable_extensions = ['png', 'jpg', 'jpeg', 'npy', 'npz', 'PNG', 'JPG', 'JPEG']
    curr_extension = str1.split('.')[-1]
    for extension in acceptable_extensions:
        str2 = str1.replace(curr_extension, extension)
        if os.path.exists(str2):
            return str2
    return None


class StaticData(object):

    def __init__(self, config, shuffle=False):
        # private variables
        self.file_groups = []
        self.type_groups = []
        self.group_names = []
        self.pair_type_groups = []
        self.len_of_groups = []
        self.transforms = {}
        # parameters
        self.shuffle = shuffle
        self.config = config


    def load_static_data(self):
        data_dict = self.config['dataset'][self.config['common']['phase'] + '_data']
        print("----------------loading %s static data.---------------------" % self.config['common']['phase'])

        if len(data_dict) == 0:
            self.len_of_groups.append(0)
            return

        self.group_names = list(data_dict.keys())
        for i, group in enumerate(data_dict.values()):  # examples: (0, group_1),  (1, group_2)
            data_types = group['data_types']  # examples: 'image', 'patch'
            data_names = group['data_names']  # examples: 'real_A', 'patch_A'
            self.file_groups.append({})
            self.type_groups.append({})
            self.len_of_groups.append(0)
            self.pair_type_groups.append(group['paired'])

            # exclude patch data since they are not stored on disk. They will be handled later.
            data_types, data_names = self.exclude_patch_data(data_types, data_names)
            assert(len(data_types) == len(data_names))

            if len(data_names) == 0:
                continue

            for data_name, data_type in zip(data_names, data_types):
                self.file_groups[i][data_name] = []
                self.type_groups[i][data_name] = data_type


            # paired data
            if group['paired']:
                # First way to load data: load a file list
                if 'file_list' in group:
                    file_list = group['file_list']
                    paired_file = open(file_list, 'rt')
                    lines = paired_file.readlines()
                    if self.shuffle:
                        random.shuffle(lines)
                    for line in lines:
                        items = line.strip().split(' ')
                        if len(items) == len(data_names):
                            ok = True
                            for item in items:
                                ok = ok and os.path.exists(item) and os.path.getsize(item) > 0
                            if ok:
                                for data_name, item in zip(data_names, items):
                                    self.file_groups[i][data_name].append(item)
                    paired_file.close()
                # second and third way to load data: specify one folder for each dataname, or specify a dataroot folder
                elif check_dataname_folder_correspondence(data_names, group, self.group_names[i]) or 'dataroot' in group:
                    dataname_to_dir_dict = {}
                    for data_name, data_type in zip(data_names, data_types):
                        if 'dataroot' in group:
                            # In new data config format, data is stored in dataroot_name/mode/dataname. e.g. FFHQ/train/pairedA
                            # In old format, data is stored in dataroot_name/mode_dataname. e.g. FFHQ/train_pairedA
                            # So we need to check both.
                            dir = os.path.join(group['dataroot'], self.config['common']['phase'], data_name)
                            if not os.path.exists(dir):
                                old_dir = os.path.join(group['dataroot'], self.config['common']['phase'] + data_name.replace('_', ''))
                                if 'numpy' in data_type:
                                    old_dir += 'numpy'
                                if not os.path.exists(old_dir):
                                    print("Both %s and %s does not exist. Please check." % (dir, old_dir))
                                    sys.exit()
                                else:
                                    dir = old_dir
                        else:
                            dir = group[data_name + '_folder']
                            if not os.path.exists(dir):
                                print("directory %s does not exist. Please check." % dir)
                                sys.exit()
                        dataname_to_dir_dict[data_name] = dir

                    filenames = os.listdir(dataname_to_dir_dict[data_names[0]])
                    if self.shuffle:
                        random.shuffle(filenames)
                    for filename in filenames:
                        if not check_path_is_static_data(filename):
                            continue
                        file_paths = []
                        for data_name in data_names:
                            file_path = os.path.join(dataname_to_dir_dict[data_name], filename)
                            checked_extension = check_different_extension_path_exists(file_path)
                            if checked_extension is not None:
                                file_paths.append(checked_extension)

                        if len(file_paths) != len(data_names):
                            print("for file %s , looks like some of the other pair data is missing. Ignoring and proceeding." % filename)
                            continue
                        else:
                            for j in range(len(data_names)):
                                data_name = data_names[j]
                                self.file_groups[i][data_name].append(file_paths[j])
                else:
                    print("method for loading data is incorrect/unspecified for data group %s." % self.group_names)
                    sys.exit()

                self.len_of_groups[i] = len(self.file_groups[i][data_names[0]])

            # unpaired data
            else:
                # First way to load data: load a file list
                if 'file_list' in group:
                    file_list = group['file_list']
                    unpaired_file = open(file_list, 'rt')
                    lines = unpaired_file.readlines()
                    if self.shuffle:
                        random.shuffle(lines)
                    item_count = 0
                    for line in lines:
                        items = line.strip().split(' ')
                        if len(items) == len(data_names):
                            ok = True
                            for item in items:
                                ok = ok and custom_check_path_exists(item) and custom_getsize(item) > 0
                            if ok:
                                has_data = False
                                for data_name, item in zip(data_names, items):
                                    if item != 'None':
                                        self.file_groups[i][data_name].append(item)
                                        has_data = True
                                if has_data:
                                    item_count += 1
                    unpaired_file.close()
                    self.len_of_groups[i] = item_count
                # second and third way to load data: specify one folder for each dataname, or specify a dataroot folder
                elif check_dataname_folder_correspondence(data_names, group, self.group_names[i]) or 'dataroot' in group:
                    max_length = 0
                    for data_name, data_type in zip(data_names, data_types):
                        if 'dataroot' in group:
                            # In new data config format, data is stored in dataroot_name/mode/dataname. e.g. FFHQ/train/pairedA
                            # In old format, data is stored in dataroot_name/mode_dataname. e.g. FFHQ/train_pairedA
                            # So we need to check both.
                            dir = os.path.join(group['dataroot'], self.config['common']['phase'], data_name)
                            if not os.path.exists(dir):
                                old_dir = os.path.join(group['dataroot'], self.config['common']['phase'] + data_name.replace('_', ''))
                                if 'numpy' in data_type:
                                    old_dir += 'numpy'
                                if not os.path.exists(old_dir):
                                    print("Both %s and %s does not exist. Please check." % (dir, old_dir))
                                    sys.exit()
                                else:
                                    dir = old_dir
                        else:
                            dir = group[data_name + '_folder']
                            if not os.path.exists(dir):
                                print("directory %s does not exist. Please check." % dir)
                                sys.exit()
                        filenames = os.listdir(dir)
                        if self.shuffle:
                            random.shuffle(filenames)

                        item_count = 0
                        for filename in filenames:
                            if not check_path_is_static_data(filename):
                                continue
                            fullpath = os.path.join(dir, filename)
                            if os.path.exists(fullpath):
                                self.file_groups[i][data_name].append(fullpath)
                                item_count += 1
                        max_length = max(item_count, max_length)
                    self.len_of_groups[i] = max_length
                else:
                    print("method for loading data is incorrect/unspecified for data group %s." % self.group_names)
                    sys.exit()


    def create_transforms(self):
        btoA = self.config['dataset']['direction'] == 'BtoA'
        input_nc = self.config['model']['output_nc'] if btoA else self.config['model']['input_nc']
        output_nc = self.config['model']['input_nc'] if btoA else self.config['model']['output_nc']
        input_grayscale_flag = (input_nc == 1)
        output_grayscale_flag = (output_nc == 1)

        data_dict = self.config['dataset'][self.config['common']['phase'] + '_data']
        for i, group in enumerate(data_dict.values()):  # examples: (0, group_1),  (1, group_2)

            if i not in self.transforms:
                self.transforms[i] = {}

            data_types = group['data_types']  # examples: 'image', 'patch'
            data_names = group['data_names']  # examples: 'real_A', 'patch_A'
            data_types, data_names = self.exclude_patch_data(data_types, data_names)
            for data_name, data_type in zip(data_names, data_types):
                if data_type in self.transforms[i]:
                    continue
                self.transforms[i][data_type] = Transforms(self.config, input_grayscale_flag=input_grayscale_flag,
                                                                    output_grayscale_flag=output_grayscale_flag)
                self.transforms[i][data_type].create_transforms_from_list(group['preprocess'])
                if '.png' in self.file_groups[i][data_name][0] or '.jpg' in self.file_groups[i][data_name][0] or \
                    '.jpeg' in self.file_groups[i][data_name][0]:
                    self.transforms[i][data_type].get_transforms().insert(0, ImagePathToImage())
                elif '.npy' in self.file_groups[i][data_name][0] or '.npz' in self.file_groups[i][data_name][0]:
                    self.transforms[i][data_type].get_transforms().insert(0, NumpyToTensor())
                self.transforms[i][data_type] = self.transforms[i][data_type].compose_transforms()


    def apply_transformations_to_images(self, img_list, img_dataname_list, transform, return_dict,
                                        next_img_paths_bucket, next_img_dataname_list):

        if len(img_list) == 1:
            return_dict[img_dataname_list[0]], _ = transform(img_list[0], None)
        elif len(img_list) > 1:
            next_data_count = len(next_img_paths_bucket)
            img_list += next_img_paths_bucket
            img_dataname_list += next_img_dataname_list

            input1, input2 = img_list[0], img_list[1:]
            output1, output2 = transform(input1, input2)  # output1 is one image. output2 is a list of images.

            if next_data_count != 0:
                output2, next_outputs = output2[:-next_data_count], output2[-next_data_count:]
                for i in range(next_data_count):
                    return_dict[img_dataname_list[-next_data_count+i] + '_next'] = next_outputs[i]

            return_dict[img_dataname_list[0]] = output1
            for j in range(0, len(output2)):
                return_dict[img_dataname_list[j+1]] = output2[j]

        return return_dict


    def calculate_landmark_scale(self, data_path, data_type, i):
        if data_type == 'image':
            original_image = Image.open(data_path)
            original_width, original_height = original_image.size
        else:
            original_image = np.load(data_path)
            original_height, original_width = original_image.shape[0], original_image.shape[1]
        transformed_image, _ = self.transforms[i][data_type](data_path, None)
        transformed_height, transformed_width = transformed_image.size()[1:]
        landmark_scale = (transformed_width / original_width, transformed_height / original_height)
        return landmark_scale


    def get_item(self, idx):

        return_dict = {}
        data_dict = self.config['dataset'][self.config['common']['phase'] + '_data']

        for i, group in enumerate(data_dict.values()):
            if self.file_groups[i] == {}:
                continue

            paired_type = self.pair_type_groups[i]
            inner_idx = idx if idx < self.len_of_groups[i] else random.randint(0, self.len_of_groups[i] - 1)

            landmark_scale = None

            # for patches since they might need to be loaded from different images.
            next_img_paths_bucket = []
            next_img_dataname_list = []
            next_numpy_paths_bucket = []
            next_numpy_dataname_list = []

            # First, handle all non-patch data
            if paired_type:
                img_paths_bucket = []
                img_dataname_list = []
                numpy_paths_bucket = []
                numpy_dataname_list = []

            for data_name, data_list in self.file_groups[i].items():
                data_type = self.type_groups[i][data_name]
                if data_type in ['image', 'numpy']:
                    if paired_type:
                        # augmentation will be applied to all images in paired group all at once so need to gather the images here.
                        if data_type == 'image':
                            img_paths_bucket.append(data_list[inner_idx])
                            img_dataname_list.append(data_name)
                        else:
                            numpy_paths_bucket.append(data_list[inner_idx])
                            numpy_dataname_list.append(data_name)
                        return_dict[data_name + '_path'] = data_list[inner_idx]
                        if landmark_scale is None:
                            landmark_scale = self.calculate_landmark_scale(data_list[inner_idx], data_type, i)
                        if 'diff_patch' in self.config['dataset'] and self.config['dataset']['diff_patch'] and \
                                data_name in group['patch_sources']:
                            next_idx = (inner_idx + 1) % (len(data_list) - 1)
                            if data_type == 'image':
                                next_img_paths_bucket.append(data_list[next_idx])
                                next_img_dataname_list.append(data_name)
                            else:
                                next_numpy_paths_bucket.append(data_list[next_idx])
                                next_numpy_dataname_list.append(data_name)
                    else:
                        unpaired_inner_idx = random.randint(0, len(data_list) - 1)
                        return_dict[data_name], _ = self.transforms[i][data_type](data_list[unpaired_inner_idx], None)
                        if landmark_scale is None:
                            landmark_scale = self.calculate_landmark_scale(data_list[unpaired_inner_idx], data_type, i)
                        if 'diff_patch' in self.config['dataset'] and self.config['dataset']['diff_patch'] and \
                                data_name in group['patch_sources']:
                            next_idx = (unpaired_inner_idx + 1) % (len(data_list) - 1)
                            return_dict[data_name + '_next'], _ = self.transforms[i][data_type](data_list[next_idx], None)
                        return_dict[data_name + '_path'] = data_list[unpaired_inner_idx]
                elif self.type_groups[i][data_name] == 'landmark':
                    # We do not apply transformations on landmarks. Only scales landmarks to transformed image's size.
                    # Also numpy data is passed into network as numpy array and not tensor.
                    lmk = np.load(data_list[inner_idx])
                    if self.config['dataset']['landmark_scale'] is not None:
                        lmk[:, 0] *= self.config['dataset']['landmark_scale'][0]
                        lmk[:, 1] *= self.config['dataset']['landmark_scale'][1]
                    else:
                        if landmark_scale is None:
                            print("landmark_scale is None. If you have not defined it in config file, please specify "
                                  "image and numpy data before landmark data and the proper scale will be automatically calculated.")
                        else:
                            lmk[:, 0] *= landmark_scale[0]
                            lmk[:, 1] *= landmark_scale[1]
                    return_dict[data_name] = lmk
                    return_dict[data_name + '_path'] = data_list[inner_idx]


            if paired_type:
                # apply augmentations to all images and numpy arrays
                if 'image' in self.transforms[i]:
                    return_dict = self.apply_transformations_to_images(img_paths_bucket, img_dataname_list,
                                                              self.transforms[i]['image'], return_dict,
                                                                       next_img_paths_bucket,
                                                                       next_img_dataname_list)
                if 'numpy' in self.transforms[i]:
                    return_dict = self.apply_transformations_to_images(numpy_paths_bucket, numpy_dataname_list,
                                                              self.transforms[i]['numpy'], return_dict,
                                                                       next_numpy_paths_bucket,
                                                                       next_numpy_dataname_list)

            # Handle patch data
            data_types = group['data_types']  # examples: 'image', 'patch'
            data_names = group['data_names']  # examples: 'real_A', 'patch_A'
            data_types, data_names = self.filter_patch_data(data_types, data_names)

            if 'patch_sources' in group:
                patch_sources = group['patch_sources']
                return_dict = self.load_patches(
                    data_names,
                    self.config['dataset']['patch_batch_size'],
                    self.config['dataset']['batch_size'],
                    self.config['dataset']['patch_size'],
                    self.config['dataset']['patch_batch_size'] // self.config['dataset']['batch_size'],
                    self.config['dataset']['diff_patch'],
                    patch_sources,
                    return_dict,
                )

        return return_dict


    def get_len(self):
        if len(self.len_of_groups) == 0:
            return 0
        else:
            return max(self.len_of_groups)


    def exclude_patch_data(self, data_types, data_names):
        data_types_patch_excluded = []
        data_names_patch_excluded = []
        for data_name, data_type in zip(data_names, data_types):
            if data_type != 'patch':
                data_types_patch_excluded.append(data_type)
                data_names_patch_excluded.append(data_name)
        return data_types_patch_excluded, data_names_patch_excluded


    def filter_patch_data(self, data_types, data_names):
        data_types_patch = []
        data_names_patch = []
        for data_name, data_type in zip(data_names, data_types):
            if data_type == 'patch':
                data_types_patch.append(data_type)
                data_names_patch.append(data_name)
        return data_types_patch, data_names_patch


    def load_patches(self, data_names, patch_batch_size, batch_size, patch_size,
                     num_patch, diff_patch, patch_sources, return_dict):

        if patch_size > 0:
            assert (patch_batch_size % batch_size == 0), \
                "patch_batch_size is not divisible by batch_size."
            assert (len(patch_sources) == len(data_names)), \
                "length of patch_sources is not the same as number of patch data specified. Please check again in config file."

            rlist = []  # used for cropping patches
            clist = []  # used for cropping patches
            for _ in range(num_patch):
                r = random.randint(0, self.config['dataset']['crop_size'] - patch_size - 1)
                c = random.randint(0, self.config['dataset']['crop_size'] - patch_size - 1)
                rlist.append(r)
                clist.append(c)

            for i in range(len(data_names)):
                # load transformed image
                patch = return_dict[patch_sources[i]] if not diff_patch else return_dict[patch_sources[i] + '_next']

                # crop patch
                patchs = []
                _, h, w = patch.size()

                for j in range(num_patch):
                    patchs.append(patch[:, rlist[j]:rlist[j] + patch_size, clist[j]:clist[j] + patch_size])
                patchs = torch.cat(patchs, 0)

                return_dict[data_names[i]] = patchs

        return return_dict

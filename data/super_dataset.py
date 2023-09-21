import copy
import torch.utils.data as data
from utils.data_utils import check_img_loaded, check_numpy_loaded

from data.test_data import add_test_data, apply_test_transforms
from data.test_video_data import TestVideoData
from data.static_data import StaticData

from multiprocessing import Pool
import sys


class DataBin(object):
    def __init__(self, filegroups):
        self.filegroups = filegroups


class SuperDataset(data.Dataset):
    def __init__(self, config, shuffle=False, check_all_data=False, DDP_device=None):
        self.config = config

        self.check_all_data = check_all_data
        self.DDP_device = DDP_device

        self.data = {}  # Will be dictionary. Keys are data names, e.g. paired_A, patch_A. Values are lists containing associated data.
        self.transforms = {}

        if self.config['common']['phase'] == 'test':
            if not self.config['testing']['test_video'] is None:
                self.test_video_data = TestVideoData(self.config)
            else:
                add_test_data(self.data, self.transforms, self.config)
            return

        self.static_data = StaticData(self.config, shuffle)


    def convert_old_config_to_new(self):
        data_types = self.config['dataset']['data_type']
        if len(data_types) == 1 and data_types[0] == 'custom':
            # convert custom data configuration to new data configuration
            old_dict = self.config['dataset']['custom_' + self.config['common']['phase'] + '_data']
            preprocess_list = self.config['dataset']['preprocess']
            new_datadict = self.config['dataset'][self.config['common']['phase'] + '_data'] = old_dict
            for i, group in enumerate(new_datadict.values()):  # examples: (0, group_1),  (1, group_2)
                group['paired'] = True
                group['preprocess'] = preprocess_list
                # custom data does not support patch so we skip patch logic.
        else:
            new_datadict = self.config['dataset'][self.config['common']['phase'] + '_data'] = {}
            preprocess_list = self.config['dataset']['preprocess']
            new_datadict['paired_group'] = {}
            new_datadict['paired_group']['paired'] = True
            new_datadict['paired_group']['data_types'] = []
            new_datadict['paired_group']['data_names'] = []
            new_datadict['paired_group']['preprocess'] = preprocess_list
            new_datadict['unpaired_group'] = {}
            new_datadict['unpaired_group']['paired'] = False
            new_datadict['unpaired_group']['data_types'] = []
            new_datadict['unpaired_group']['data_names'] = []
            new_datadict['unpaired_group']['preprocess'] = preprocess_list

            for i in range(len(self.config['dataset']['data_type'])):
                data_type = self.config['dataset']['data_type'][i]
                if data_type == 'paired' or data_type == 'paired_numpy':
                    if self.config['dataset']['paired_' + self.config['common']['phase'] + '_filelist'] != '':
                        new_datadict['paired_group']['file_list'] = self.config['dataset'][
                            'paired_' + self.config['common']['phase'] + '_filelist']
                    elif self.config['dataset']['paired_' + self.config['common']['phase'] + 'A_folder'] != '' and \
                            self.config['dataset']['paired_' + self.config['common']['phase'] + 'B_folder'] != '':
                        new_datadict['paired_group']['paired_A_folder'] = self.config['dataset']['paired_' + self.config['common']['phase'] + 'A_folder']
                        new_datadict['paired_group']['paired_B_folder'] = self.config['dataset']['paired_' + self.config['common']['phase'] + 'B_folder']
                    else:
                        new_datadict['paired_group']['dataroot'] = self.config['dataset']['dataroot']

                    new_datadict['paired_group']['data_names'].append('paired_A')
                    new_datadict['paired_group']['data_names'].append('paired_B')
                    if data_type == 'paired':
                        new_datadict['paired_group']['data_types'].append('image')
                        new_datadict['paired_group']['data_types'].append('image')
                    else:
                        new_datadict['paired_group']['data_types'].append('numpy')
                        new_datadict['paired_group']['data_types'].append('numpy')

                elif data_type == 'unpaired' or data_type == 'unpaired_numpy':
                    if self.config['dataset']['unpaired_' + self.config['common']['phase'] + 'A_filelist'] != ''\
                            and self.config['dataset']['unpaired_' + self.config['common']['phase'] + 'B_filelist'] != '':
                        # combine those two filelists into one filelist
                        self.combine_two_filelists_into_one(
                            self.config['dataset']['unpaired_' + self.config['common']['phase'] + 'A_filelist'],
                            self.config['dataset']['unpaired_' + self.config['common']['phase'] + 'B_filelist']
                        )
                        new_datadict['unpaired_group']['file_list'] = './tmp_filelist.txt'
                    elif self.config['dataset']['unpaired_' + self.config['common']['phase'] + 'A_folder'] != '' and \
                            self.config['dataset']['unpaired_' + self.config['common']['phase'] + 'B_folder'] != '':
                        new_datadict['unpaired_group']['unpaired_A_folder'] = self.config['dataset']['unpaired_' + self.config['common']['phase'] + 'A_folder']
                        new_datadict['unpaired_group']['unpaired_B_folder'] = self.config['dataset']['unpaired_' + self.config['common']['phase'] + 'B_folder']
                    else:
                        new_datadict['unpaired_group']['dataroot'] = self.config['dataset']['dataroot']

                    new_datadict['unpaired_group']['data_names'].append('unpaired_A')
                    new_datadict['unpaired_group']['data_names'].append('unpaired_B')
                    if data_type == 'unpaired':
                        new_datadict['unpaired_group']['data_types'].append('image')
                        new_datadict['unpaired_group']['data_types'].append('image')
                    else:
                        new_datadict['unpaired_group']['data_types'].append('numpy')
                        new_datadict['unpaired_group']['data_types'].append('numpy')

                elif data_type == 'landmark':
                    if self.config['dataset']['paired_' + self.config['common']['phase'] + '_filelist'] != '':
                        new_datadict['paired_group']['file_list'] = self.config['dataset'][
                            'paired_' + self.config['common']['phase'] + '_filelist']
                    elif 'paired_' + self.config['common']['phase'] + 'A_lmk_folder' in self.config['dataset'] and \
                            'paired_' + self.config['common']['phase'] + 'B_lmk_folder' in self.config['dataset'] and \
                            self.config['dataset']['paired_' + self.config['common']['phase'] + 'A_lmk_folder'] != '' and \
                            self.config['dataset']['paired_' + self.config['common']['phase'] + 'B_lmk_folder'] != '':
                        new_datadict['paired_group']['lmk_A_folder'] = self.config['dataset']['paired_' + self.config['common']['phase'] + 'A_lmk_folder']
                        new_datadict['paired_group']['lmk_B_folder'] = self.config['dataset']['paired_' + self.config['common']['phase'] + 'B_lmk_folder']
                    else:
                        new_datadict['paired_group']['dataroot'] = self.config['dataset']['dataroot']

                    new_datadict['paired_group']['data_names'].append('lmk_A')
                    new_datadict['paired_group']['data_names'].append('lmk_B')
                    new_datadict['paired_group']['data_types'].append('landmark')
                    new_datadict['paired_group']['data_types'].append('landmark')

            # Handle patches. This needs to happen after all non-patch data are added first.
            if 'patch' in self.config['dataset']['data_type']:
                # determine if patch comes from paired or unpaired image
                if 'paired_A' in new_datadict['paired_group']['data_names']:
                    new_datadict['paired_group']['data_types'].append('patch')
                    new_datadict['paired_group']['data_names'].append('patch_A')
                    new_datadict['paired_group']['data_types'].append('patch')
                    new_datadict['paired_group']['data_names'].append('patch_B')

                    if 'patch_sources' not in new_datadict['paired_group']:
                        new_datadict['paired_group']['patch_sources'] = []
                    new_datadict['paired_group']['patch_sources'].append('paired_A')
                    new_datadict['paired_group']['patch_sources'].append('paired_B')
                else:
                    new_datadict['unpaired_group']['data_types'].append('patch')
                    new_datadict['unpaired_group']['data_names'].append('patch_A')
                    new_datadict['unpaired_group']['data_types'].append('patch')
                    new_datadict['unpaired_group']['data_names'].append('patch_B')

                    if 'patch_sources' not in new_datadict['unpaired_group']:
                        new_datadict['unpaired_group']['patch_sources'] = []
                    new_datadict['unpaired_group']['patch_sources'].append('unpaired_A')
                    new_datadict['unpaired_group']['patch_sources'].append('unpaired_B')

                if 'diff_patch' not in self.config['dataset']:
                    self.config['dataset']['diff_patch'] = False

        new_datadict = {key: value for key, value in new_datadict.items() if len(value['data_names']) > 0}

        print('-----------------------------------------------------------------------')
        print("converted %s data configuration: " % self.config['common']['phase'])
        for key, value in new_datadict.items():
            print(key + ': ', value)
        print('-----------------------------------------------------------------------')

        return self.config


    def combine_two_filelists_into_one(self, filelist1, filelist2):
        tmp_file = open('./tmp_filelist.txt', 'w+')
        f1 = open(filelist1, 'r')
        f2 = open(filelist2, 'r')
        f1_lines = f1.readlines()
        f2_lines = f2.readlines()
        min_index = min(len(f1_lines), len(f2_lines))
        for i in range(min_index):
            tmp_file.write(f1_lines[i].strip() + ' ' + f2_lines[i].strip() + '\n')
        if min_index == len(f1_lines):
            for i in range(min_index, len(f2_lines)):
                tmp_file.write('None ' + f2_lines[i].strip() + '\n')
        else:
            for i in range(min_index, len(f1_lines)):
                tmp_file.write(f1_lines[i].strip() + ' None\n')

        tmp_file.close()
        f1.close()
        f2.close()


    def __len__(self):
        if self.config['common']['phase'] == 'test':
                if self.config['testing']['test_video'] is not None:
                    return self.test_video_data.get_len()
                else:
                    if len(self.data.keys()) == 0:
                        return 0
                    else:
                        min_len = 999999
                        for k, v in self.data.items():
                            length = len(v)
                            if length < min_len:
                                min_len = length
                        return min_len
        else:
            return self.static_data.get_len()



    def get_item_logic(self, index):
        return_dict = {}

        if self.config['common']['phase'] == 'test':
            if not self.config['testing']['test_video'] is None:
                return self.test_video_data.get_item()
            else:
                apply_test_transforms(index, self.data, self.transforms, return_dict)
                return return_dict

        return_dict = self.static_data.get_item(index)

        return return_dict


    def __getitem__(self, index):
        if self.config['dataset']['accept_data_error']:
            while True:
                try:
                    return self.get_item_logic(index)
                except Exception as e:
                    print("Exception encountered in super_dataset's getitem function: ", e)
                    index = (index + 1) % self.__len__()
        else:
            return self.get_item_logic(index)


    def split_data(self, value_mode, value, mode='split'):
        new_dataset = copy.deepcopy(self)
        ret1, new_dataset.static_data = self.split_data_helper(self.static_data, new_dataset.static_data, value_mode, value, mode=mode)
        if ret1 is not None:
            self.static_data = ret1
        return self, new_dataset


    def split_data_helper(self, dataset, new_dataset, value_mode, value, mode='split'):
        for i in range(len(dataset.file_groups)):
            max_split_index = 0
            for k in dataset.file_groups[i].keys():
                length = len(dataset.file_groups[i][k])
                if value_mode == 'count':
                    split_index = min(length, value)
                else:
                    split_index = int((1 - value) * length)
                max_split_index = max(max_split_index, split_index)
                new_dataset.file_groups[i][k] = new_dataset.file_groups[i][k][split_index:]
                if mode == 'split':
                    dataset.file_groups[i][k] = dataset.file_groups[i][k][:split_index]
            new_dataset.len_of_groups[i] -= max_split_index
            if mode == 'split':
                dataset.len_of_groups[i] = max_split_index
        if mode == 'split':
            return dataset, new_dataset
        else:
            return None, new_dataset


    def check_data_helper(self, databin):
        all_pass = True
        for group in databin.filegroups:
            for data_name, data_list in group.items():
                for data in data_list:
                    if '.npy' in data:  # case: numpy array or landmark
                        all_pass = all_pass and check_numpy_loaded(data)
                    else:  # case: image
                        all_pass = all_pass and check_img_loaded(data)
        return all_pass


    def check_data(self):
        if self.DDP_device is None or self.DDP_device == 0:
            print("-----------------------Checking all data-------------------------------")
            data_ok = True
            if self.config['dataset']['n_threads'] == 0:
                data_ok = data_ok and self.check_data_helper(self.static_data)
            else:
                # start n_threads number of workers to perform data checking
                with Pool(processes=self.config['dataset']['n_threads']) as pool:
                    checks = pool.map(self.check_data_helper,
                                      self.split_data_into_bins(self.config['dataset']['n_threads']))
                for check in checks:
                    data_ok = data_ok and check
            if data_ok:
                print("---------------------all data passed check.-----------------------")
            else:
                print("---------------------The above data have failed in data checking. "
                      "Please fix first.---------------------------")
                sys.exit()


    def split_data_into_bins(self, num_bins):
        bins = []
        for i in range(num_bins):
            bins.append(DataBin(filegroups=[]))

        # handle static data
        bins = self.split_data_into_bins_helper(bins, self.static_data)
        return bins


    def split_data_into_bins_helper(self, bins, dataset):
        num_bins = len(bins)
        for bin in bins:
            for group_idx in range(len(dataset.file_groups)):
                bin.filegroups.append({})

        for group_idx in range(len(dataset.file_groups)):
            file_group = dataset.file_groups[group_idx]
            for data_name, data_list in file_group.items():
                num_items_in_bin = len(data_list) // num_bins
                for data_index in range(len(data_list)):
                    which_bin = min(data_index // num_items_in_bin, num_bins - 1)
                    if data_name not in bins[which_bin].filegroups[group_idx]:
                        bins[which_bin].filegroups[group_idx][data_name] = []
                    bins[which_bin].filegroups[group_idx][data_name].append(data_list[data_index])
        return bins

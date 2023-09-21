import yaml
import copy
from typing import Union

class BaseConfig():

    def __init__(self):
        self.__config_dict = {}
        self.__check_func_dict = {}

        is_greater_than_0 = lambda x: x > 0

        # common config
        self._add_option('common', 'name', str, 'style_master')
        self._add_option('common', 'model', str, 'cycle_gan')
        self._add_option('common', 'phase', str, 'train', check_func=lambda x: x in ['train', 'test'])
        self._add_option('common', 'gpu_ids', list, [0])
        self._add_option('common', 'verbose', bool, False)

        # model config
        self._add_option('model', 'input_nc', int, 3, check_func=is_greater_than_0)
        self._add_option('model', 'output_nc', int, 3, check_func=is_greater_than_0)

        # dataset config
        # common dataset options
        self._add_option('dataset', 'use_absolute_datafile', bool, True)
        self._add_option('dataset', 'batch_size', int, 1, check_func=is_greater_than_0)
        self._add_option('dataset', 'n_threads', int, 4, check_func=is_greater_than_0)
        self._add_option('dataset', 'dataroot', str, './')
        self._add_option('dataset', 'drop_last', bool, False)
        self._add_option('dataset', 'landmark_scale', list, None)
        self._add_option('dataset', 'check_all_data', bool, False)
        self._add_option('dataset', 'accept_data_error', bool, True)  # Upon loading a bad data, if this is true,
                                                                        # dataloader will throw an exception and
                                                                        # load the next good data.
                                                                        # If this is false, process will crash.

        self._add_option('dataset', 'train_data', dict, {})
        self._add_option('dataset', 'val_data', dict, {})

        # paired data config
        self._add_option('dataset', 'paired_trainA_folder', str, '')
        self._add_option('dataset', 'paired_trainB_folder', str, '')
        self._add_option('dataset', 'paired_train_filelist', str, '')
        self._add_option('dataset', 'paired_valA_folder', str, '')
        self._add_option('dataset', 'paired_valB_folder', str, '')
        self._add_option('dataset', 'paired_val_filelist', str, '')

        # unpaired data config
        self._add_option('dataset', 'unpaired_trainA_folder', str, '')
        self._add_option('dataset', 'unpaired_trainB_folder', str, '')
        self._add_option('dataset', 'unpaired_trainA_filelist', str, '')
        self._add_option('dataset', 'unpaired_trainB_filelist', str, '')
        self._add_option('dataset', 'unpaired_valA_folder', str, '')
        self._add_option('dataset', 'unpaired_valB_folder', str, '')
        self._add_option('dataset', 'unpaired_valA_filelist', str, '')
        self._add_option('dataset', 'unpaired_valB_filelist', str, '')

        # custom data
        self._add_option('dataset', 'custom_train_data', dict, {})
        self._add_option('dataset', 'custom_val_data', dict, {})

        # training config
        self._add_option('training', 'checkpoints_dir', str, './checkpoints')
        self._add_option('training', 'log_dir', str, './logs')
        self._add_option('training', 'use_new_log', bool, False)
        self._add_option('training', 'continue_train', bool, False)
        self._add_option('training', 'which_epoch', str, 'latest')
        self._add_option('training', 'n_epochs', int, 100, check_func=is_greater_than_0)
        self._add_option('training', 'n_epochs_decay', int, 100, check_func=is_greater_than_0)
        self._add_option('training', 'save_latest_freq', int, 5000, check_func=is_greater_than_0)
        self._add_option('training', 'print_freq', int, 200, check_func=is_greater_than_0)
        self._add_option('training', 'save_epoch_freq', int, 5, check_func=is_greater_than_0)
        self._add_option('training', 'epoch_as_iter', bool, False)
        self._add_option('training', 'lr', float, 2e-4, check_func=is_greater_than_0)
        self._add_option('training', 'lr_policy', str, 'linear',
            check_func=lambda x: x in ['linear', 'step', 'plateau', 'cosine'])
        self._add_option('training', 'lr_decay_iters', int, 50, check_func=is_greater_than_0)
        self._add_option('training', 'DDP', bool, False)
        self._add_option('training', 'num_nodes', int, 1, check_func=is_greater_than_0)
        self._add_option('training', 'DDP_address', str, '127.0.0.1')
        self._add_option('training', 'DDP_port', str, '29700')
        self._add_option('training', 'find_unused_parameters', bool, False) # a DDP option that allows backward on a subgraph of the model
        self._add_option('training', 'val_percent', float, 5.0, check_func=is_greater_than_0)  # Uses x% of training data to validate
        self._add_option('training', 'val', bool, True)  # perform validation every epoch
        self._add_option('training', 'save_training_progress', bool, False)  # save images to create a training progression video

        # testing config
        self._add_option('testing', 'results_dir', str, './results')
        self._add_option('testing', 'load_size', int, 512, check_func=is_greater_than_0)
        self._add_option('testing', 'crop_size', int, 512, check_func=is_greater_than_0)
        self._add_option('testing', 'preprocess', list, ['scale_width'])
        self._add_option('testing', 'visual_names', list, [])
        self._add_option('testing', 'num_test', int, 999999, check_func=is_greater_than_0)
        self._add_option('testing', 'image_format', str, 'jpg', check_func=lambda x: x in ['input', 'jpg', 'jpeg', 'png'])

    def _add_option(self, group_name, option_name, value_type, default_value, check_func=None):
        # check name type
        if not type(group_name) is str or not type(option_name) is str:
            raise Exception('Type of {} and {} must be str.'.format(group_name, option_name))

        # add group
        if not group_name in self.__config_dict:
            self.__config_dict[group_name] = {}
            self.__check_func_dict[group_name] = {}

        # check type & default value
        if not type(value_type) is type:
            try:
                if value_type.__origin__ is not Union:
                    raise Exception('{} is not a type.'.format(value_type))
            except Exception as e:
                print(e)
        if not type(default_value) is value_type:
            try:
                if value_type.__origin__ is not Union:
                    raise Exception('Type of {} must be {}.'.format(default_value, value_type))
            except Exception as e:
                print(e)

        # add option to dict
        if not option_name in self.__config_dict[group_name]:
            if not check_func is None and not check_func(default_value):
                raise Exception('Checking {}/{} failed.'.format(group_name, option_name))
            self.__config_dict[group_name][option_name] = default_value
            self.__check_func_dict[group_name][option_name] = check_func
        else:
            raise Exception('{} has been already added.'.format(option_name))

    def parse_config(self, cfg_file):
        # load config from yaml file
        with open(cfg_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
        if not type(yaml_config) is dict:
            raise Exception('Loading yaml file failed.')

        # replace default options
        config_dict = copy.deepcopy(self.__config_dict)
        for group in config_dict:
            if group in yaml_config:
                for option in config_dict[group]:
                    if option in yaml_config[group]:
                        value = yaml_config[group][option]
                        if not type(value) is type(config_dict[group][option]):
                            try: # if <config_dict[group][option]> is not union, it won't have __origin__ attribute. So will throw an error.
                                # The line below is necessary because we check if <config_dict[group][option]> has __origin__ attribute.
                                if config_dict[group][option].__origin__ is Union:
                                    # check to see if type of <value> belongs to a type in the union.
                                    if not isinstance(value, config_dict[group][option].__args__):
                                        raise Exception('Type of {}/{} must be {}.'.format(group, option,
                                                                config_dict[group][option].__args__))
                            except Exception as e: # if the error was thrown, we know there's a type error.
                                print(e)
                        else:
                            check_func = self.__check_func_dict[group][option]
                            if not check_func is None and not check_func(value):
                                raise Exception('Checking {}/{} failed.'.format(group, option))
                            config_dict[group][option] = value
        return config_dict


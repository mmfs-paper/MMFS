import os
from utils.util import check_path_is_static_data
from utils.data_utils import Transforms
from utils.augmentation import ImagePathToImage, NumpyToTensor

def add_test_data(data, transforms, config):
    A_paths = []
    B_paths = []

    if not config['testing']['test_img'] is None:
        A_paths.append(config['testing']['test_img'])
        B_paths.append(config['testing']['test_img'])
    else:
        files = os.listdir(config['testing']['test_folder'])
        for fn in files:
            if not check_path_is_static_data(fn):
                continue
            full_path = os.path.join(config['testing']['test_folder'], fn)
            A_paths.append(full_path)
            B_paths.append(full_path)

    btoA = config['dataset']['direction'] == 'BtoA'
    # get the number of channels of input image
    input_nc = config['model']['output_nc'] if btoA else config['model']['input_nc']
    output_nc = config['model']['input_nc'] if btoA else config['model']['output_nc']

    transform = Transforms(config, input_grayscale_flag=(input_nc == 1), output_grayscale_flag=(output_nc == 1))
    transform.create_transforms_from_list(config['testing']['preprocess'])
    transform.get_transforms().insert(0, ImagePathToImage())
    transform = transform.compose_transforms()

    transform_np = Transforms(config, input_grayscale_flag=(input_nc == 1), output_grayscale_flag=(output_nc == 1))
    transform_np.transform_list.append(NumpyToTensor())
    transform_np = transform_np.compose_transforms()

    data['test_A_path'] = A_paths
    data['test_B_path'] = B_paths
    transforms['test'] = transform
    transforms['test_np'] = transform_np

def apply_test_transforms(index, data, transforms, return_dict):
    if len(data['test_A_path']) > 0:
        ext_name = os.path.splitext(data['test_A_path'][index])[1]
        if not ext_name.lower() in ['.npy', '.npz']:
            return_dict['test_A'], return_dict['test_B'] = transforms['test'] \
                (data['test_A_path'][index], data['test_B_path'][index])
        else:
            return_dict['test_A'], return_dict['test_B'] = transforms['test_np'] \
                (data['test_A_path'][index], data['test_B_path'][index])
        return_dict['test_A_path'] = data['test_A_path'][index]
        return_dict['test_B_path'] = data['test_B_path'][index]

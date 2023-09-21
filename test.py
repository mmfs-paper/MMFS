import argparse
import os
import cv2
from tqdm import tqdm

from utils.util import *
from data import CustomDataLoader
from data.super_dataset import SuperDataset
from models import create_model
from configs import parse_config

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Style Master')
    parser.add_argument('--cfg_file', type=str, default='./exp/sp2pII-phase2.yaml')
    parser.add_argument('--test_img', type=str, default='', help='path to your test img')
    parser.add_argument('--test_video', type=str, default='')
    parser.add_argument('--test_folder', type=str, default='') # ./example/source
    parser.add_argument('--ckpt', type=str, default='') # ./pretrained_models/phase2_pretrain_90000.pth
    parser.add_argument('--overwrite_output_dir', type=str, default='') # ./example/outputs/multi-model
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()

    # parse config
    config = parse_config(args.cfg_file)

    # fix gpu ordering
    gpu_string = ','.join(map(str, config['common']['gpu_ids']))
    gpu_ids_fix = list(range(len(config['common']['gpu_ids'])))  # wants GPU ids match nvidia-smi output order

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    #os.environ['CUDA_VISIBLE_DEVICES'] = gpu_string
    config['common']['gpu_ids'] = gpu_ids_fix

    # hard-code some parameters for test
    config['common']['phase'] = 'test'
    config['dataset']['n_threads'] = 0   # test code only supports num_threads = 0
    config['dataset']['batch_size'] = 1    # test code only supports batch_size = 1
    config['dataset']['serial_batches'] = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    config['dataset']['no_flip'] = True    # no flip; comment this line if results on flipped images are needed.

    # override data augmentation
    config['dataset']['load_size'] = config['testing']['load_size']
    config['dataset']['crop_size'] = config['testing']['crop_size']
    config['dataset']['preprocess'] = config['testing']['preprocess']

    # add testing path
    config['testing']['test_img'] = None if args.test_img == '' else args.test_img
    config['testing']['test_video'] = None if args.test_video == '' else args.test_video
    config['testing']['test_folder'] = args.test_folder

    config['training']['pretrained_model'] = args.ckpt
    dataset = SuperDataset(config)
    dataloader = CustomDataLoader(config, dataset)

    model = create_model(config)      # create a model given opt.model and other options
    model.load_networks(0, ckpt=args.ckpt)

    model.eval()

    if args.overwrite_output_dir != '':
        save_path = args.overwrite_output_dir
    else:
        save_path = os.path.join(config['testing']['results_dir'], os.path.splitext(os.path.split(args.cfg_file)[1])[0],
                                config['common']['name'])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def reduce(x):
        return reduce(x[0]) if not type(x) is str else x

    ext_name = config['testing']['image_format']
    use_input_format = (ext_name == 'input')
    output_video = (not config['testing']['test_video'] is None)
    vw_dict = {}
    video_paths = []

    for i, data in enumerate(tqdm(dataloader)):
        if i >= config['testing']['num_test']:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        # save result
        items = os.path.splitext(os.path.split(reduce(img_path))[1])
        img_fn = items[0]
        if use_input_format:
            ext_name = items[1][1:]

        for k, v in visuals.items():
            if not output_video:
                tensor2file(v, os.path.join(save_path, img_fn + '_' + k), ext_name)
            else:
                img = tensor2im(v)
                if not k in vw_dict:
                    h, w = img.shape[:2]
                    video_path = os.path.join(save_path, k + '_.mp4')
                    video_paths.append(video_path)
                    vw_dict[k] = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
                vw_dict[k].write(img[:,:,::-1])

    for _, v in vw_dict.items():
        v.release()

    # convert to libx264
    for video_path in video_paths:
        os.system('ffmpeg -i {} -c:v libx264 {}'.format(video_path, video_path[:-5] + '.mp4'))
        os.system('rm {}'.format(video_path))

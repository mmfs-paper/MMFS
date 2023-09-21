import argparse
import time
import datetime
from data import CustomDataLoader
from data.super_dataset import SuperDataset

from models import create_model
from configs import parse_config
from utils.util import print_losses, check_path, make_grid, AverageMeter
from utils.data_utils import check_old_config_val_possible
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import copy
import sys



def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Style Master')
    parser.add_argument('--cfg_file', type=str, default='') # ./exp/sp2pII-phase4.yaml
    
    args = parser.parse_args()

    # parse config
    config = parse_config(args.cfg_file)

    for group in config:
        print(group + ':')
        for k, v in config[group].items():
            print('  {}: {}'.format(k, v))

    # we want GPU ids match nvidia-smi output order, so do some manipulations here.
    # GPU ids need to always start from 0, but the system variable CUDA_VISIBLE_DEVICES can be set to e.g. GPU 2 and 3.
    gpu_string = ','.join(map(str, config['common']['gpu_ids']))
    gpu_ids_fix = list(range(len(config['common']['gpu_ids'])))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_string
    config['common']['gpu_ids'] = gpu_ids_fix

    if config['training']['DDP']:
        num_nodes = config['training']['num_nodes']
        num_gpus = len(config['common']['gpu_ids'])
        config['training']['world_size'] = num_gpus * num_nodes
        os.environ['MASTER_ADDR'] = config['training']['DDP_address']
        os.environ['MASTER_PORT'] = config['training']['DDP_port']
        mp.spawn(train_val, nprocs=num_gpus, args=(config,))
    else:
        # under DP mode, shall set batch size to (actual effective batch size * num_gpu)
        config['dataset']['batch_size'] *= len(config['common']['gpu_ids'])
        if 'patch_batch_size' in config['dataset']:
            config['dataset']['patch_batch_size'] *= len(config['common']['gpu_ids'])
        train_val(None, config)

# GPU parameter is automatically filled when using DDP. It is an irrelevant placeholder if not using DDP.
def train_val(gpu, config):
    import torch
    if config['training']['val']:
        config_val = copy.deepcopy(config)
        config_val['common']['phase'] = 'val'

    if config['training']['DDP']:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=config['training']['world_size'],
            rank=gpu
        )
        torch.cuda.set_device(gpu)  # DDP requirement

    # Dataset and dataloader construction.
    # prepare train data
    train_dataset = SuperDataset(config, shuffle=True, check_all_data=config['dataset']['check_all_data'],
                                 DDP_device=gpu)
    if config['dataset']['train_data'] == {}:
        print("old data config format detected. Converting to new format.")
        train_dataset.config = train_dataset.convert_old_config_to_new()

    train_dataset.static_data.load_static_data()
    train_dataset.static_data.create_transforms()
    if train_dataset.check_all_data:
        train_dataset.check_data()

    if len(train_dataset) == 0:
        if gpu == 0 or gpu is None:
            print("Train set has 0 data samples. Exiting.")
        sys.exit(0)

    _, train_video_dataset = train_dataset.split_data('count', 5, mode='copy_partial')

    # prepare val data
    force_use_train_data = False
    if config['training']['val']:
        val_dataset = SuperDataset(config_val)
        if len(config_val['dataset']['train_data']) == 0:
            if check_old_config_val_possible(config_val):
                val_dataset.convert_old_config_to_new()
                val_dataset.static_data.load_static_data()
            else:
                force_use_train_data = True
        else:
            val_dataset.static_data.load_static_data()


        if len(val_dataset) == 0 or force_use_train_data:
            if gpu == 0 or gpu is None:
                print("Validation set has 0 data samples. Using part of training data for validation.")
            validation_ratio = config['training']['val_percent']/100

            train_dataset, val_dataset = train_dataset.split_data('ratio', validation_ratio, mode='split')

            if len(val_dataset) == 0:
                print("There are too few training data to establish a validation set. "
                      "Use the training set as validation set.")
                val_dataset = copy.deepcopy(train_dataset)

        val_dataset.static_data.create_transforms()
        if val_dataset.check_all_data:
            val_dataset.check_data()

    # print dataset info
    if gpu == 0 or gpu is None:
        print("--------train dataset static data content-----------")
        for i, cnt in enumerate(train_dataset.static_data.len_of_groups):
            print("%s:   %d" % (train_dataset.static_data.group_names[i], cnt))
        print("----------------------------------------")

        if config['training']['val']:
            print("--------val dataset static data content-----------")
            for i, cnt in enumerate(val_dataset.static_data.len_of_groups):
                print("group %s:   %d" % (val_dataset.static_data.group_names[i], cnt))
            print("----------------------------------------")


    # prepare dataloaders
    train_dataloader = CustomDataLoader(config, train_dataset, DDP_gpu=gpu, drop_last=config['dataset']['drop_last'])
    if config['training']['val']:
        val_dataloader = CustomDataLoader(config_val, val_dataset, DDP_gpu=gpu, drop_last=config['dataset']['drop_last'])
    config_train_video = copy.deepcopy(config)
    config_train_video['dataset']['serial_batches'] = True
    config_train_video['batch_size'] = 1
    train_video_dataloader = CustomDataLoader(config_train_video, train_video_dataset, DDP_gpu=None, drop_last=False)

    if config['training']['DDP']:
        model = create_model(config, DDP_device=gpu)      # create a DDP model given opt.model and other options
        model.setup(config, DDP_device=gpu)  # regular setup: load and print networks; create schedulers
    else:
        model = create_model(config)  # create a model (singleGPU or dataparallel) given opt.model and other options
        model.setup(config)               # regular setup: load and print networks; create schedulers
    total_iters = model.total_iters                # the total number of training iterations

    # visualization setups
    if gpu == 0 or gpu is None:
        log_dir = os.path.join(config['training']['log_dir'], config['common']['name'] + '_' +
                               datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        check_path(log_dir)
        writer = SummaryWriter(log_dir)
    train_loss_averages = {}
    val_loss_averages = {}

    if config['training']['epoch_as_iter']:
        iter_ub = config['training']['n_epochs'] + config['training']['n_epochs_decay']
        while total_iters < iter_ub:
            # Training loop
            if gpu == 0 or gpu is None:
                print('-------------------------------Training----------------------------------')
            model.train()
            quit = False
            for i, data in enumerate(train_dataloader):
                total_iters += 1
                if total_iters >= iter_ub:
                    quit = True
                    break
                model.curr_epoch = total_iters
                model.total_iters = total_iters

                model.set_input(data)
                model.optimize_parameters()

                losses = model.get_current_losses()
                for k, v in losses.items():
                    if k not in train_loss_averages:
                        train_loss_averages[k] = AverageMeter()
                    train_loss_averages[k].update(v)

                if (total_iters % config['training']['print_freq'] == 0) and ((gpu == 0) or (gpu is None)):
                    print_losses(total_iters, 1, 1, losses, train_loss_averages)
                    tmp = make_grid(model)
                    for j in range(0, len(tmp[0])):
                        img_grid, name = tmp[0][j], tmp[1][j]
                        if config['training']['use_new_log']:
                            writer.add_image('Training/' + name, img_grid, total_iters)
                        else:
                            writer.add_image('iteration ' + str(total_iters) + ' training ' + name, img_grid)
                    for k, v in losses.items():
                        writer.add_scalar('Training/' + k, train_loss_averages[k].avg(), total_iters)

                if (total_iters % config['training']['save_latest_freq'] == 0) and ((gpu == 0) or (gpu is None)):
                    print('saving the latest model (total_iters %d)' % total_iters)
                    model.save_networks('latest')

                if (total_iters % config['training']['save_epoch_freq'] == 0) and ((gpu == 0) or (gpu is None)):
                    print('saving the model at the end of iters %d' % total_iters)
                    model.save_networks('latest')
                    #model.save_networks(total_iters)

                model.update_learning_rate()
                for k, v in losses.items():
                    train_loss_averages[k].clear()

            if quit:
                exit(0)

            if config['training']['val']:
                # Validation loop
                if gpu == 0 or gpu is None:
                    print('-------------------------------Validating----------------------------------')
                model.eval()
                for i, data in enumerate(val_dataloader):
                    with torch.no_grad():
                        model.set_input(data)
                        model.eval_step()

                        losses = model.get_current_losses()
                        for k, v in losses.items():
                            if k not in val_loss_averages:
                                val_loss_averages[k] = AverageMeter()
                            val_loss_averages[k].update(v)

                if gpu == 0 or gpu is None:
                    print_losses(total_iters, 1, 1, losses, val_loss_averages, mode='validating')
                    tmp = make_grid(model)
                    for j in range(0, len(tmp[0])):
                        img_grid, name = tmp[0][j], tmp[1][j]
                        if config['training']['use_new_log']:
                            writer.add_image('Validation/' + name, img_grid, total_iters)
                        else:
                            writer.add_image('iteration ' + str(total_iters) + ' validating ' + name, img_grid)
                    for k, v in losses.items():
                        writer.add_scalar('Validation/' + k, val_loss_averages[k].avg(), total_iters)

                main_loss = 'G' if 'G' in losses else model.loss_names[0]
                if val_loss_averages[main_loss].avg() < model.best_val_loss and (gpu == 0 or gpu is None):
                    model.best_val_loss = val_loss_averages[main_loss].avg()
                    print('New validation best loss. saving the model.')
                    model.save_networks('', val_loss=model.best_val_loss)

                for k, v in losses.items():
                    val_loss_averages[k].clear()

            if config['training']['save_training_progress']:
                # produce images on the same images every epoch to visualize how training is progressing.
                if gpu == 0 or gpu is None:
                    for i, data in enumerate(train_video_dataloader):
                        with torch.no_grad():
                            model.set_input(data)
                            model.forward()
                            tmp = make_grid(model)
                            for j in range(0, len(tmp[0])):
                                img_grid, name = tmp[0][j], tmp[1][j]
                                if config['training']['use_new_log']:
                                    writer.add_image('Training Video/' + name + ' ' + str(i), img_grid, total_iters)
                                else:
                                    writer.add_image('epoch 0 iteration ' + str(total_iters) + ' training_video ' + name + ' ' + str(i), img_grid)


    for epoch in range(model.curr_epoch, config['training']['n_epochs'] + config['training']['n_epochs_decay'] + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.curr_epoch = epoch

        # Training loop
        if gpu == 0 or gpu is None:
            print('-------------------------------Training----------------------------------')
        model.train()
        for i, data in enumerate(train_dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            for k, v in losses.items():
                if k not in train_loss_averages:
                    train_loss_averages[k] = AverageMeter()
                train_loss_averages[k].update(v)

            if epoch_iter % config['training']['print_freq'] == 0 and ((gpu == 0) or (gpu is None)):
                print_losses(epoch, epoch_iter, (len(train_dataset) + config['dataset']['batch_size'] - 1) // config['dataset']['batch_size'], losses, train_loss_averages)
                tmp = make_grid(model)
                for j in range(0, len(tmp[0])):
                    img_grid, name = tmp[0][j], tmp[1][j]
                    if config['training']['use_new_log']:
                        writer.add_image('Training/' + name, img_grid, total_iters)
                    else:
                        writer.add_image('epoch ' + str(epoch) + ' iteration ' + str(total_iters) + ' training ' + name, img_grid)
                for k, v in losses.items():
                    writer.add_scalar('Training/' + k, train_loss_averages[k].avg(), total_iters)

            if total_iters % config['training']['save_latest_freq'] == 0 and ((gpu == 0) or (gpu is None)):   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                model.save_networks('latest')

            iter_data_time = time.time()

        if gpu == 0 or gpu is None:
            print_losses(epoch, epoch_iter, (len(train_dataset) + config['dataset']['batch_size'] - 1) // config['dataset']['batch_size'], losses, train_loss_averages)
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, config['training']['n_epochs'] + config['training']['n_epochs_decay'], time.time() - epoch_start_time))
            tmp = make_grid(model)
            for j in range(0, len(tmp[0])):
                img_grid, name = tmp[0][j], tmp[1][j]
                if config['training']['use_new_log']:
                    writer.add_image('Training/' + name, img_grid, total_iters)
                else:
                    writer.add_image('epoch ' + str(epoch) + ' iteration ' + str(total_iters) + ' training ' + name, img_grid)
            for k, v in losses.items():
                writer.add_scalar('Training/' + k, train_loss_averages[k].avg(), total_iters)

        if epoch % config['training']['save_epoch_freq'] == 0 and ((gpu == 0) or (gpu is None)):              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()  # update learning rates at the end of every epoch.
        for k, v in losses.items():
            train_loss_averages[k].clear()

        if config['training']['val']:
            # Validation loop
            if gpu == 0 or gpu is None:
                print('-------------------------------Validating----------------------------------')
            model.eval()
            epoch_iter = 0
            for i, data in enumerate(val_dataloader):  # inner loop within one epoch
                epoch_iter += 1

                with torch.no_grad():
                    model.set_input(data)
                    model.eval_step()

                    losses = model.get_current_losses()
                    for k, v in losses.items():
                        if k not in val_loss_averages:
                            val_loss_averages[k] = AverageMeter()
                        val_loss_averages[k].update(v)

            if gpu == 0 or gpu is None:
                print_losses(epoch, epoch_iter, (len(val_dataset) + config['dataset']['batch_size'] - 1)
                             // config['dataset']['batch_size'], losses, val_loss_averages, mode='validating')
                tmp = make_grid(model)
                for j in range(0, len(tmp[0])):
                    img_grid, name = tmp[0][j], tmp[1][j]
                    if config['training']['use_new_log']:
                        writer.add_image('Validation/' + name, img_grid, total_iters)
                    else:
                        writer.add_image('epoch ' + str(epoch) + ' iteration ' + str(total_iters) + ' validating ' + name, img_grid)
                for k, v in losses.items():
                    writer.add_scalar('Validation/' + k, val_loss_averages[k].avg(), total_iters)

            main_loss = 'G' if 'G' in losses else model.loss_names[0]
            if val_loss_averages[main_loss].avg() < model.best_val_loss and gpu == 0 or gpu is None:
                model.best_val_loss = val_loss_averages[main_loss].avg()
                print('New validation best loss. saving the model.')
                model.save_networks('', val_loss=model.best_val_loss)

            for k, v in losses.items():
                val_loss_averages[k].clear()

        if config['training']['save_training_progress']:
            # produce images on the same images every epoch to visualize how training is progressing.
            if gpu == 0 or gpu is None:
                for i, data in enumerate(train_video_dataloader):
                    with torch.no_grad():
                        model.set_input(data)
                        model.forward()
                        tmp = make_grid(model)
                        for j in range(0, len(tmp[0])):
                            img_grid, name = tmp[0][j], tmp[1][j]
                            if config['training']['use_new_log']:
                                writer.add_image('Training Video/' + name + ' ' + str(i), img_grid, total_iters)
                            else:
                                writer.add_image('epoch ' + str(epoch) + ' iteration ' + str(total_iters) + ' training_video ' + name + ' ' + str(i), img_grid)

    # If we shut down process now, writer could save incomplete data. Wait a bit to let it finish.
    time.sleep(5)

if __name__ == '__main__':
    main()

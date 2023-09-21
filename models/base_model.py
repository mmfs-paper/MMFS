import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from models.modules import networks
from utils.util import check_path
from  utils.net_size import calc_computation


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, config, DDP_device=None):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.config = config
        self.gpu_ids = config['common']['gpu_ids']
        self.isTrain = config['common']['phase'] == 'train'
        if DDP_device is None:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
            self.DDP_device = None
            self.on_cpu = (self.device.type == 'cpu')
        else:
            self.device = DDP_device
            self.DDP_device = DDP_device
            self.on_cpu = False
        self.save_dir = os.path.join(config['training']['checkpoints_dir'], config['common']['name'])  # save all the checkpoints to save_dir
        if config['dataset']['preprocess'] != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.curr_epoch = 0
        self.total_iters = 0
        self.best_val_loss = 999999

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <configimize_parameters> and <test>."""
        pass

    @abstractmethod
    def trace_jit(self, input):
        """trace torchscript model for C++. Called by <trace_jit.py>"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    @abstractmethod
    def eval_step(self):
        """Forward and backward pass but without upgrading weights; called in every validation iteration"""
        pass

    def setup(self, config, DDP_device=None):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, config) for optimizer in self.optimizers]
        if not self.isTrain:
            load_suffix = '{}'.format(config['testing']['which_epoch'])
            self.load_networks(load_suffix)
        elif config['training']['continue_train']:
            load_suffix = '{}'.format(config['training']['which_epoch'])
            self.load_networks(load_suffix)
        self.print_networks(config['common']['verbose'], DDP_device=DDP_device)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models train mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.config['training']['lr_policy'] == 'plateau':
                scheduler.step(self.metric, epoch=self.curr_epoch)
            else:
                scheduler.step(epoch=self.curr_epoch)

        # lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        if not self.isTrain and len(self.config['testing']['visual_names']) > 0:
            visual_names = list(set(self.visual_names).intersection(set(self.config['testing']['visual_names'])))
        else:
            visual_names = self.visual_names
        visual_ret = OrderedDict()
        for name in visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch, val_loss=None):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        check_path(self.save_dir)
        save_filename = 'epoch_%s.pth' % epoch if val_loss is None else 'best_val_epoch.pth'
        checkpoint = {}
        # save all the models
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # if use DDP, save only on rank 0. If using dataparallel, second condition meets.
                    if self.DDP_device == 0 or self.DDP_device is None:
                        checkpoint[name+'_model'] = net.module.state_dict()
                else:
                    checkpoint[name+'_model'] = net.state_dict()

        # save all the optimizers
        optimizer_index = 0
        for optimizer in self.optimizers:
            checkpoint['optimizer_'+str(optimizer_index)] = optimizer.state_dict()
            optimizer_index += 1

        # save all the schedulers
        scheduler_index = 0
        for scheduler in self.schedulers:
            checkpoint['scheduler_' + str(scheduler_index)] = scheduler.state_dict()
            scheduler_index += 1

        # save other information
        checkpoint['epoch'] = self.curr_epoch
        checkpoint['total_iters'] = self.total_iters
        checkpoint['metric'] = self.metric
        if val_loss is not None:
            checkpoint['best_val_loss'] = val_loss

        torch.save(checkpoint, os.path.join(self.save_dir, save_filename))

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch, ckpt=None):
        """Load all the networks from the disk.

        Parameters:
            epoch (str) -- current epoch; used in the file name 'epoch_%s.pth' % epoch. Models in the old format
            with the names '%s_net_%s.pth' % (epoch, name) are also supported. Models in the new format takes priority.
        """
        load_filename = 'epoch_%s.pth' % epoch
        if ckpt is None:
            final_load_path = os.path.join(self.save_dir, load_filename)
        else:
            final_load_path = ckpt
        if os.path.exists(final_load_path):
            # new checkpoint format.
            print('loading the model in new format from %s' % final_load_path)
            if self.DDP_device is not None:
                # unpack the tensors on GPU 0, then transfer to whatever device it needs to be on
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.DDP_device}
                checkpoint = torch.load(final_load_path, map_location=map_location)
            else:
                checkpoint = torch.load(final_load_path)
            for k, v in checkpoint.items():
                # load models
                if 'model' in k:
                    name = k.split('_model')[0]
                    if not self.isTrain and 'D' in name: # does not load discriminator when not training
                        continue
                    if not hasattr(self, 'net' + name):
                        continue
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel) or isinstance(net, torch.nn.parallel.DistributedDataParallel):
                        net = net.module

                    # if you are using PyTorch newer than 0.4 (e.g., built from
                    # GitHub source), you can remove str() on self.device
                    if hasattr(v, '_metadata'):
                        del v._metadata

                    # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(v.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(v, net, key.split('.'))
                    net.load_state_dict(v)
                # load optimizers
                elif 'optimizer' in k:
                    if not self.isTrain:
                        continue
                    index = int(k.split('_')[-1])
                    self.optimizers[index].load_state_dict(v)
                # load schedulers
                elif 'scheduler' in k:
                    if not self.isTrain:
                        continue
                    index = int(k.split('_')[-1])
                    self.schedulers[index].load_state_dict(v)
                # load other stuffs
                elif k == 'epoch':
                    self.curr_epoch = int(v) + 1
                elif k == 'total_iters':
                    self.total_iters = int(v)
                elif k == 'metric':
                    self.metric = float(v)
                elif k == 'best_val_loss':
                    self.best_val_loss = float(v)
                else:
                    print('Checkpoint load error. Unrecognized parameter saved in checkpoint: ', k)
            return

        # old checkpoint format.
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel) or isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose, DDP_device=None):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        if DDP_device is None or DDP_device == 0:
            print('---------- Networks initialized -------------')
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(self, 'net' + name)
                    num_params = 0
                    for param in net.parameters():
                        num_params += param.numel()
                    if verbose:
                        print(net)
                    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                    if 'G' in name:
                        calc_computation(net, self.config['model']['input_nc'], self.config['dataset']['crop_size'],self.config['dataset']['crop_size'], DDP_device=DDP_device)
            print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

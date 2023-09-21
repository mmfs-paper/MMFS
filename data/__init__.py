"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler

class CustomDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, config, dataset, DDP_gpu=None, drop_last=False):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.config = config
        self.dataset = dataset

        if DDP_gpu is None:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=config['dataset']['batch_size'],
                shuffle=not config['dataset']['serial_batches'],
                num_workers=int(config['dataset']['n_threads']), drop_last=drop_last)
        else:
            sampler = DistributedSampler(self.dataset, num_replicas=self.config['training']['world_size'],
                                         rank=DDP_gpu)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=config['dataset']['batch_size'],
                shuffle=False,
                num_workers=int(config['dataset']['n_threads']),
                sampler=sampler,
                drop_last=drop_last)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), 1e9)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.config['dataset']['batch_size'] >= 1e9:
                break
            yield data

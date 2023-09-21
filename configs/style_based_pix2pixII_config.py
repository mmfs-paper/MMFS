from .base_config import BaseConfig
from typing import Union as Union

class StyleBasedPix2PixIIConfig(BaseConfig):

    def __init__(self):
        super(StyleBasedPix2PixIIConfig, self).__init__()

        is_greater_than_0 = lambda x: x > 0

        # model config
        self._add_option('model', 'ngf', int, 64, check_func=is_greater_than_0)
        self._add_option('model', 'min_feats_size', list, [4, 4])

        # dataset config
        self._add_option('dataset', 'data_type', list, ['unpaired'])
        self._add_option('dataset', 'direction', str, 'AtoB')
        self._add_option('dataset', 'serial_batches', bool, False)
        self._add_option('dataset', 'load_size', int, 512, check_func=is_greater_than_0)
        self._add_option('dataset', 'crop_size', int, 512, check_func=is_greater_than_0)
        self._add_option('dataset', 'preprocess', Union[list, str], ['resize'])
        self._add_option('dataset', 'no_flip', bool, True)

        # training config
        self._add_option('training', 'beta1', float, 0.1, check_func=is_greater_than_0)
        self._add_option('training', 'data_aug_prob', float, 0.0, check_func=lambda x: x >= 0.0)
        self._add_option('training', 'style_mixing_prob', float, 0.0, check_func=lambda x: x >= 0.0)
        self._add_option('training', 'phase', int, 1, check_func=lambda x: x in [1, 2, 3, 4])
        self._add_option('training', 'pretrained_model', str, 'model.pth')
        self._add_option('training', 'src_text_prompt', str, 'photo')
        self._add_option('training', 'text_prompt', str, 'a portrait in style of sketch')
        self._add_option('training', 'image_prompt', str, 'style.png')
        self._add_option('training', 'lambda_L1', float, 1.0)
        self._add_option('training', 'lambda_Feat', float, 4.0)
        self._add_option('training', 'lambda_ST', float, 1.0)
        self._add_option('training', 'lambda_GAN', float, 1.0)
        self._add_option('training', 'lambda_CLIP', float, 1.0)
        self._add_option('training', 'lambda_PROJ', float, 1.0)
        self._add_option('training', 'ema', float, 0.999)

        # testing config
        self._add_option('testing', 'aspect_ratio', float, 1.0, check_func=is_greater_than_0)

from packaging import version
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision.transforms.transforms import Lambda, Compose
from torchvision.transforms import functional as F
from collections.abc import Iterable
import torch, torchvision
import numbers
import copy

if version.parse(torchvision.__version__) <= version.parse('0.7.0'):
    from torchvision.transforms.transforms import _get_image_size

def check_input_type_perform_action(input, type, action, *args, **kwargs):
    output = input
    if isinstance(input, list):
        for i in range(0, len(input)):
            if type is None:
                if input[i] is not None:  # do not combine with last line, to avoid calling isinstance on None.
                    output[i] = action(input[i], *args, **kwargs)
            elif isinstance(input[i], type):
                output[i] = action(input[i], *args, **kwargs)
    elif type is None:
        if input is not None:
            output = action(input, *args, **kwargs)
    elif isinstance(input, type):
        output = action(input, *args, **kwargs)
    return output


"""
Most of these functions are imported from torchvision.transforms.transforms and edited to support 2 or more inputs.
"""

class JointCompose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input1, input2):
        for t in self.transforms:
            input1, input2 = t(input1, input2)
        return input1, input2


class Grayscale(object):

    def __init__(self, input1_output_channels=1, input2_output_channels=1):
        self.input1_output_channels = input1_output_channels
        self.input2_output_channels = input2_output_channels

    def __call__(self, input1, input2):
        output1 = F.to_grayscale(input1, num_output_channels=self.input1_output_channels) if self.input1_output_channels == 1 else input1
        output2 = check_input_type_perform_action(input2, Image.Image, F.to_grayscale, num_output_channels=self.input2_output_channels) \
            if self.input2_output_channels == 1 else input2
        return output1, output2


class Resize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, input1, input2):
        output1 = F.resize(input1, self.size, self.interpolation)
        output2 = check_input_type_perform_action(input2, Image.Image, F.resize, self.size, self.interpolation)
        return output1, output2


class ScaleWidth:

    def __init__(self, target_size, method=Image.BICUBIC):
        self.target_size = target_size
        self.method = method

    def scalewidth(self, img):
        ow, oh = img.size
        w = self.target_size
        h = int(self.target_size * oh / ow)
        img_resized = img.resize((w, h), self.method)

        if h > w:
            # if resized image's height is larger than its width, crop the center
            left = 0
            top = h // 2 - self.target_size // 2
            right = self.target_size
            bottom = top + self.target_size
            img_resized = img_resized.crop((left, top, right, bottom))
        elif h < w:
            # pad the heights
            delta_w = self.target_size - w
            delta_h = self.target_size - h
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            img_resized = ImageOps.expand(img_resized, padding)

        return img_resized

    def __call__(self, input1, input2):
        output1 = self.scalewidth(input1)
        output2 = check_input_type_perform_action(input2, Image.Image, self.scalewidth)
        return output1, output2


class RandomCrop(object):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        if version.parse(torchvision.__version__) <= version.parse('0.7.0'):
            w, h = _get_image_size(img)
        else:
            w, h = F._get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def pad(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        return img

    def get_crop_range(self, img):
        return self.get_params(img, self.size)

    def pad_and_crop(self, input, i, j, h, w):
        return F.crop(self.pad(input), i, j, h, w)

    def __call__(self, input1, input2):
        output1 = self.pad(input1)
        i, j, h, w = self.get_crop_range(output1)
        output1 = F.crop(output1, i, j, h, w)
        output2 = check_input_type_perform_action(input2, Image.Image, self.pad_and_crop, i, j, h, w)
        return output1, output2


class Crop:

    def __init__(self, pos, size):
        self.pos = pos
        self.size = size

    def crop(self, img):
        ow, oh = img.size
        x1, y1 = self.pos
        tw = th = self.size
        if (ow > tw or oh > th):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img

    def __call__(self, input1, input2):
        output1 = self.crop(input1)
        output2 = check_input_type_perform_action(input2, Image.Image, self.crop)
        return output1, output2


class ColorJitter(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, input1, input2):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        output1 = transform(input1)
        output2 = check_input_type_perform_action(input2, Image.Image, transform)
        return output1, output2


class RandomAffine(object):

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, input1, input2):
        params = self.get_params(self.degrees, self.translate, self.scale, self.shear, input1.size)
        output1 = F.affine(input1, *params, resample=self.resample, fillcolor=self.fillcolor)
        output2 = check_input_type_perform_action(input2, Image.Image, F.affine, *params, resample=self.resample, fillcolor=self.fillcolor)
        return output1, output2


class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, input1, input2):
        angle = self.get_params(self.degrees)
        output1 = F.rotate(input1, angle, self.resample, self.expand, self.center, self.fill)
        output2 = check_input_type_perform_action(input2, Image.Image, F.rotate, angle, self.resample, self.expand, self.center, self.fill)
        return output1, output2


class RandomBlur:
    def __init__(self, blur_chance):
        self.blur_chance = blur_chance

    def get_params(self):
        if self.blur_chance > random.random():
            kernel = random.randint(3, 12)
            while kernel % 2 != 1:
                kernel = random.randint(3, 12)
        else:
            kernel = None
        return kernel

    def blur(self, image, kernel):
        image = image.filter(ImageFilter.GaussianBlur(radius=kernel))
        return image

    def __call__(self, input1, input2):
        kernel = self.get_params()
        if kernel is None:
            return input1, input2
        else:
            output1 = self.blur(input1, kernel)
            output2 = check_input_type_perform_action(input2, Image.Image, self.blur, kernel)
            return output1, output2


class NoiseTransform:
    """code is partly from http://www.xiaoliangbai.com/2016/09/09/more-on-image-noise-generation and edited by Oliver."""

    def __init__(self, noise_type):
        self.noise_type = noise_type

    def get_params(self, image):
        params = []
        image_np = np.array(image)
        row, col, ch = image_np.shape
        if random.random() < 0.5:
            return None
        if self.noise_type == "gauss":
            mean = 0.0
            std = random.uniform(0.001, 0.3)
            gauss = np.random.normal(mean, std, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            params.append(gauss)
            return params
        elif self.noise_type == "s&p":
            s_vs_p = 0.5
            amount = random.uniform(0.001, 0.01)

            # Generate Salt '1' noise
            num_salt = np.ceil(amount * image_np.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image_np.shape]
            coords[2] = np.random.randint(0, 3, int(num_salt))
            params.append(copy.deepcopy(coords))

            # Generate Pepper '0' noise
            num_pepper = np.ceil(amount * image_np.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image_np.shape]
            params.append(copy.deepcopy(coords))
            return params
        elif self.noise_type == "poisson":
            noisy = np.random.poisson(image_np)
            params.append(noisy)
            return params
        elif self.noise_type == "speckle":
            factor = random.uniform(0.01, 0.4)
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch) * factor
            params.append(gauss)
            return params
        elif self.noise_type == "band":
            smaller_dim = min(col, row)
            num_bands = random.randrange(smaller_dim // 2, smaller_dim)
            scale = random.uniform(1.0, 10.0)

            offset = np.zeros(image_np.shape).astype(np.float64)

            # horizontal branding
            num_list = list(range(image.width))  # list of integers from 0 to image width-1
            # adjust this boundaries to fit your needs
            random.shuffle(num_list)
            horizontal_bands = num_list[:num_bands]
            for w in horizontal_bands:
                offset[w, :, :] += random.uniform(-1, 1) * scale

            # vertical branding
            num_list = list(range(image.height))  # list of integers from 0 to image height-1
            # adjust this boundaries to fit your needs
            random.shuffle(num_list)
            vertical_bands = num_list[:num_bands]
            for h in vertical_bands:
                offset[:, h, :] += random.uniform(-1, 1) * scale

            params.append(offset)
            return params
        else:
            return params

    def apply(self, image, params):
        """
        image: ndarray (input image data. It will be converted to float)
        """
        if params is None:
            return image
        image_np = np.array(image)
        if self.noise_type == "gauss":
            gauss = params[0]
            noisy = image_np + image_np * gauss
            noisy = np.clip(noisy, 0, 255)
            return Image.fromarray(noisy.astype('uint8'))
        elif self.noise_type == "s&p":
            out = image_np
            # Generate Salt '1' noise
            coords = params[0]
            out[tuple(coords)] = 255
            # Generate Pepper '0' noise
            coords = params[1]
            out[tuple(coords)] = 0
            out = np.clip(out, 0, 255)
            return Image.fromarray(out.astype('uint8'))
        elif self.noise_type == "poisson":
            noisy = params[0]
            noisy = np.clip(noisy, 0, 255)
            return Image.fromarray(noisy.astype('uint8'))
        elif self.noise_type == "speckle":
            gauss = params[0]
            noisy = image_np + image_np * gauss
            noisy = np.clip(noisy, 0, 255)
            return Image.fromarray(noisy.astype('uint8'))
        elif self.noise_type == "band":
            offset = params[0]
            noisy = image_np + offset
            noisy = np.clip(noisy, 0, 255)
            return Image.fromarray(noisy.astype('uint8'))
        else:
            return image

    def __call__(self, input1, input2):
        params = self.get_params(input1)
        output1 = self.apply(input1, params)
        output2 = check_input_type_perform_action(input2, Image.Image, self.apply, params)
        return output1, output2


class MakePower2:
    def __init__(self, base, method=Image.BICUBIC):
        self.base = base
        self.method = method
        self.print_size_warning = PrintSizeWarning()

    def apply(self, img):
        ow, oh = img.size
        h = int(round(oh / self.base) * self.base)
        w = int(round(ow / self.base) * self.base)
        if h == oh and w == ow:
            return img

        self.print_size_warning(ow, oh, w, h)
        return img.resize((w, h), self.method)

    def __call__(self, input1, input2):
        output1 = self.apply(input1)
        output2 = check_input_type_perform_action(input2, Image.Image, self.apply)
        return output1, output2


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def get_params(self):
        if random.random() < self.p:
            return True
        else:
            return False

    def __call__(self, input1, input2):
        flip = self.get_params()
        if flip:
            output1 = F.hflip(input1)
            output2 = check_input_type_perform_action(input2, Image.Image, F.hflip)
        else:
            output1, output2 = input1, input2
        return output1, output2


class Flip:
    def __init__(self, flip):
        self.flip = flip

    def transpose(self, input):
        return input.transpose(Image.FLIP_LEFT_RIGHT)

    def __call__(self, input1, input2):
        if self.flip:
            output1 = input1.transpose(Image.FLIP_LEFT_RIGHT)
            output2 = check_input_type_perform_action(input2, Image.Image, self.transpose)
        else:
            output1, output2 = input1, input2
        return output1, output2


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, input1, input2):
        output1 = F.to_tensor(input1)
        output2 = check_input_type_perform_action(input2, None, F.to_tensor)
        return output1, output2


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, first_input_mean, first_input_std, second_input_mean=None, second_input_std=None, inplace=False):
        self.first_input_mean = first_input_mean
        self.first_input_std = first_input_std
        self.second_input_mean = second_input_mean if second_input_mean is not None else first_input_mean
        self.second_input_std = second_input_std if second_input_std is not None else first_input_std
        self.inplace = inplace

    def __call__(self, tensor1, tensor2):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        output1 = F.normalize(tensor1, self.first_input_mean, self.first_input_std, self.inplace)
        output2 = check_input_type_perform_action(tensor2, None, F.normalize, self.second_input_mean, self.second_input_std, self.inplace)
        return output1, output2


class PrintSizeWarning:
    def __init__(self):
        self.has_printed = False

    def __call__(self, ow, oh, w, h):
        if not self.has_printed:
            print("The image size needs to be a multiple of 4. "
                "The loaded image size was (%d, %d), so it was adjusted to "
                "(%d, %d). This adjustment will be done to all images "
                "whose sizes are not multiples of 4" % (ow, oh, w, h))
            self.has_printed = True


class ImagePathToImage:
    """Convert an image path to an image.

    Parameters:
        filename  --  the input file path.
    """

    def load_img(self, path):
        return Image.open(path).convert('RGB')

    def __call__(self, filename1, filename2):
        img1 = self.load_img(filename1)
        img2 = check_input_type_perform_action(filename2, None, self.load_img)
        return img1, img2


class NumpyToTensor:
    """Convert a numpy array to a tensor.

    Parameters:
        filename  --  the input file path.
    """

    def load_numpy(self, filename):
        npy = np.load(filename)
        if isinstance(npy, np.lib.npyio.NpzFile):
            npy = npy['data']
        if len(npy.shape) == 2:
            npy = np.tile(npy, (1, 1, 1))
        else:
            npy = np.transpose(npy, (2, 0, 1))
        return torch.from_numpy(npy).float()

    def __call__(self, filename1, filename2):
        tensor1 = self.load_numpy(filename1)
        tensor2 = check_input_type_perform_action(filename2, None, self.load_numpy)
        return tensor1, tensor2

# code from https://github.com/omerbt/Splice/blob/master/util/losses.py

from torchvision.transforms import Resize
from torchvision import transforms
import torch
import torch.nn.functional as F

from .extractor import VitExtractor


class LossG:

    def __init__(self, device):
        super().__init__()

        self.extractor = VitExtractor(model_name='dino_vitb8', device=device)

        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(224, max_size=480)

        self.global_transform = transforms.Compose([ global_resize_transform, imagenet_norm ])

    def get_cls_token(self, inputs):
        tokens = []
        for x in inputs:
            x = self.global_transform(x).unsqueeze(0)
            cls_token = self.extractor.get_feature_from_input(x)[-1][0, 0, :]
            tokens.append(cls_token.unsqueeze(0))
        return torch.cat(tokens, dim=0) # shape: B x 768

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0).to(device)
            b = self.global_transform(b).unsqueeze(0).to(device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_global_id_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                keys_a = self.extractor.get_keys_from_input(a.unsqueeze(0), 11)
            keys_b = self.extractor.get_keys_from_input(b.unsqueeze(0), 11)
            loss += F.mse_loss(keys_a, keys_b)
        return loss

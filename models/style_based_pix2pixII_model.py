import copy
import clip
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from PIL import Image
from torch import autograd
from .base_model import BaseModel
from models.modules import networks
from models.modules.stylegan2.model import Generator, Discriminator, StyledConv, ToRGB, EqualLinear, ResBlock, ConvLayer, PixelNorm
from models.modules.stylegan2.op import conv2d_gradfix
from models.modules.stylegan2.non_leaking import augment
from models.modules.vit.losses import LossG


class TrainingPhase(Enum):
    ENCODER = 1
    BASE_MODEL = 2
    CLIP_MAPPING = 3
    FEW_SHOT = 4


class CLIPFeats2Wplus(nn.Module):

    def __init__(self, n_tokens=16, embedding_dim=512):
        super().__init__()

        self.position_embedding = nn.Parameter(embedding_dim ** -0.5 * torch.randn(n_tokens, embedding_dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, norm_first=True), num_layers=4)

    def forward(self, x):
        x_in = x.view(x.shape[0], 1, x.shape[1]) + self.position_embedding
        return F.leaky_relu(self.transformer(x_in.permute(1, 0, 2)), negative_slope=0.2)


class Stylizer(nn.Module):

    def __init__(self, ngf=64, phase=TrainingPhase.ENCODER, model_weights=None):
        super(Stylizer, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            ConvLayer(3, ngf, 3),          # 512
            ResBlock(ngf * 1, ngf * 1),    # 256
            ResBlock(ngf * 1, ngf * 2),    # 128
            ResBlock(ngf * 2, ngf * 4),    # 64
            ResBlock(ngf * 4, ngf * 8),    # 32
            ConvLayer(ngf * 8, ngf * 8, 3) # 32
        )

        # mapping network
        self.mapping_z = nn.Sequential(*([ PixelNorm() ] + [ EqualLinear(512, 512, activation='fused_lrelu', lr_mul=0.01) for _ in range(8) ]))

        # style-based decoder
        channels = {
            32 : ngf * 8,
            64 : ngf * 8,
            128: ngf * 4,
            256: ngf * 2,
            512: ngf * 1
        }
        self.decoder0 = StyledConv(channels[32], channels[32], 3, 512)
        self.to_rgb0 = ToRGB(channels[32], 512, upsample=False)
        for i in range(4):
            ichan = channels[2 ** (i + 5)]
            ochan = channels[2 ** (i + 6)]
            setattr(self, f'decoder{i + 1}a', StyledConv(ichan, ochan, 3, 512, upsample=True))
            setattr(self, f'decoder{i + 1}b', StyledConv(ochan, ochan, 3, 512))
            setattr(self, f'to_rgb{i + 1}', ToRGB(ochan, 512))
        self.n_latent = 10

        # random style for testing
        self.test_z = torch.randn(1, 512)

        # load pretrained model weights
        if phase == TrainingPhase.ENCODER:
            # load pretrained stylegan2 and freeze these params
            for param in self.mapping_z.parameters():
                param.requires_grad = False
            for i in range(4):
                for key in [f'decoder{i + 1}a', f'decoder{i + 1}b', f'to_rgb{i + 1}']:
                    for param in getattr(self, key).parameters():
                        param.requires_grad = False
            self.load_state_dict(self._convert_stylegan2_dict(model_weights), strict=False)
        elif phase == TrainingPhase.BASE_MODEL:
            # load pretrained encoder and stylegan2 decoder
            self.load_state_dict(model_weights)
        elif phase == TrainingPhase.CLIP_MAPPING:
            self.clip_mapper = CLIPFeats2Wplus(n_tokens=self.n_latent)
            # load pretraned base model and freeze all params except clip mapper
            self.load_state_dict(model_weights, strict=False)
            params = dict(self.named_parameters())
            for k in params.keys():
                if 'clip_mapper' in k:
                    print(f'{k} not freezed !')
                    continue
                params[k].requires_grad = False
        elif phase == TrainingPhase.FEW_SHOT:
            self.clip_mapper = CLIPFeats2Wplus(n_tokens=self.n_latent)
            # load pretrained base model and freeze encoder & mapping
            self.load_state_dict(model_weights)
            self.encoder.requires_grad_(False)
            self.mapping_z.requires_grad_(False)
            self.clip_mapper.requires_grad_(False)

    def _convert_stylegan2_dict(self, src):
        res = {}
        for k, v in src.items():
            if k.startswith('style.'):
                res[k.replace('style.', 'mapping_z.')] = v
            else:
                name, idx = k.split('.')[:2]
                if name == 'convs':
                    idx = int(idx)
                    if idx >= 6:
                        res[k.replace(f'{name}.{idx}.', f'decoder{idx // 2 - 2}{chr(97 + idx % 2)}.')] = v
                elif name == 'to_rgbs':
                    idx = int(idx)
                    if idx >= 3:
                        res[k.replace(f'{name}.{idx}.', f'to_rgb{idx - 2}.')] = v
        return res

    def get_styles(self, x, **kwargs):
        if len(kwargs) == 0:
            return self.mapping_z(self.test_z.to(x.device).repeat(x.shape[0], 1)).repeat(self.n_latent, 1, 1)
        elif 'mixing' in kwargs and kwargs['mixing']:
            w0 = self.mapping_z(torch.randn(x.shape[0], 512, device=x.device))
            w1 = self.mapping_z(torch.randn(x.shape[0], 512, device=x.device))
            inject_index = random.randint(1, self.n_latent - 1)
            return torch.cat([
                w0.repeat(inject_index, 1, 1),
                w1.repeat(self.n_latent - inject_index, 1, 1)
            ])
        elif 'z' in kwargs:
            return self.mapping_z(kwargs['z']).repeat(self.n_latent, 1, 1)
        elif 'clip_feats' in kwargs:
            return self.clip_mapper(kwargs['clip_feats'])
        else:
            z = torch.randn(x.shape[0], 512, device=x.device)
            return self.mapping_z(z).repeat(self.n_latent, 1, 1)

    def forward(self, x, **kwargs):
        # encode
        feat = self.encoder(x)

        # get style code
        styles = self.get_styles(x, **kwargs)

        # style-based generate
        feat = self.decoder0(feat, styles[0])
        out = self.to_rgb0(feat, styles[1])
        for i in range(4):
            feat = getattr(self, f'decoder{i + 1}a')(feat, styles[i * 2 + 1])
            feat = getattr(self, f'decoder{i + 1}b')(feat, styles[i * 2 + 2])
            out = getattr(self, f'to_rgb{i + 1}')(feat, styles[i * 2 + 3], out)

        return F.hardtanh(out)


class StyleBasedPix2PixIIModel(BaseModel):
    """
    This class implements the Style-Based Pix2Pix model version II.
    """

    def __init__(self, config, DDP_device=None):
        BaseModel.__init__(self, config, DDP_device=DDP_device)

        self.d_reg_freq = 16
        self.lambda_r1 = 10
        self.step = 0
        self.phase = TrainingPhase(config['training']['phase'])

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.phase == TrainingPhase.ENCODER:
            self.loss_names = ['G', 'G_L1', 'G_Feat']
        elif self.phase == TrainingPhase.BASE_MODEL:
            self.loss_names = ['G', 'G_ST', 'G_GAN', 'D']
        elif self.phase == TrainingPhase.CLIP_MAPPING:
            self.loss_names = ['G', 'G_L1', 'G_Feat']
        elif self.phase == TrainingPhase.FEW_SHOT:
            self.loss_names = ['G', 'G_ST', 'G_CLIP', 'G_PROJ']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'G_ema', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G_ema']

        self.data_aug_prob = config['training']['data_aug_prob']

        min_feats_size = tuple(config['model']['min_feats_size'])

        def __init_net(model):
            return networks.init_net(model, init_type='none', init_gain=0.0, gpu_ids=self.gpu_ids,
                DDP_device=self.DDP_device, find_unused_parameters=config['training']['find_unused_parameters'])

        if self.phase == TrainingPhase.ENCODER: # train a encoder for stylegan2
            # load and init pretrained stylegan2
            model_dict = torch.load(config['training']['pretrained_model'], map_location='cpu')
            self.stylegan2 = Generator(512, 512, 8)
            self.stylegan2.load_state_dict(model_dict['g'])
            self.stylegan2 = __init_net(self.stylegan2)
            self.stylegan2.eval()
            self.stylegan2.requires_grad_(False)

            # init netG
            self.netG = Stylizer(ngf=config['model']['ngf'], phase=self.phase, model_weights=model_dict['g'])
            self.netG = __init_net(self.netG)

            # init netD
            self.netD = Discriminator(min(min_feats_size) * 128, min_feats_size)
            self.netD.load_state_dict(model_dict['d'])
            self.netD = __init_net(self.netD)
            self.netD.eval()
            self.netD.requires_grad_(False)

        elif self.phase == TrainingPhase.BASE_MODEL: # finetune the whole model
            model_dict = torch.load(config['training']['pretrained_model'], map_location='cpu')

            # init netG
            self.netG = Stylizer(ngf=config['model']['ngf'], phase=self.phase, model_weights=model_dict['G_ema_model'])
            self.netG = __init_net(self.netG)

            # init netD
            self.netD = Discriminator(min(min_feats_size) * 128, min_feats_size)
            self.netD.load_state_dict(model_dict['D_model'])
            self.netD = __init_net(self.netD)

        elif self.phase == TrainingPhase.CLIP_MAPPING or self.phase == TrainingPhase.FEW_SHOT: # train the clip mapper or zero/one shot finetune
            # init CLIP
            self.clip_model, self.pil_to_tensor = clip.load('ViT-B/32', device=self.device)
            self.clip_model.eval()
            self.clip_model.requires_grad_(False)

            model_dict = torch.load(config['training']['pretrained_model'], map_location='cpu')

            # init netG
            self.netG = Stylizer(ngf=config['model']['ngf'], phase=self.phase, model_weights=model_dict['G_ema_model'])
            self.netG = __init_net(self.netG)

            # init netD
            self.netD = Discriminator(min(min_feats_size) * 128, min_feats_size)
            self.netD.load_state_dict(model_dict['D_model'])
            self.netD = __init_net(self.netD)
            self.netD.eval()
            self.netD.requires_grad_(False)

        if self.phase == TrainingPhase.FEW_SHOT: # set hook to get clip vit tokens
            def clip_vit_hook(model, feat_in, feat_out):
                self.clip_vit_tokens = feat_out[1:].permute(1, 0, 2).float() # remove cls token
            self.clip_model.visual.transformer.resblocks[3].register_forward_hook(clip_vit_hook)

        # create netG ema
        self.netG_ema = copy.deepcopy(self.netG)
        self.netG_ema.eval()
        self.netG_ema.requires_grad_(False)
        self.ema(self.netG_ema, self.netG, 0.0)

        # CLIP mean & std
        self.clip_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=self.device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=self.device).view(1, 3, 1, 1)

        if self.isTrain:
            # define loss functions
            if self.phase == TrainingPhase.ENCODER:
                self.criterionL1 = nn.L1Loss()
            elif self.phase == TrainingPhase.BASE_MODEL:
                self.criterionStyleGAN = networks.GANLoss('wgangp').to(self.device)
                self.vitLoss = LossG(self.device)
            elif self.phase == TrainingPhase.CLIP_MAPPING:
                self.criterionL1 = nn.L1Loss()
            elif self.phase == TrainingPhase.FEW_SHOT:
                self.criterionL1 = nn.L1Loss()
                self.vitLoss = LossG(self.device)
                self.cosineSim = nn.CosineSimilarity(dim=1)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=config['training']['lr'], betas=(config['training']['beta1'], 0.999))
            d_reg_ratio = self.d_reg_freq / (self.d_reg_freq + 1)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=config['training']['lr'] * d_reg_ratio, betas=(config['training']['beta1'] ** d_reg_ratio, 0.999 ** d_reg_ratio))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def ema(self, tgt, src, decay=0.999):
        param_tgt = dict(tgt.named_parameters())
        param_src = dict(src.named_parameters())
        for key in param_tgt.keys():
            param_tgt[key].data.mul_(decay).add_(param_src[key].data, alpha=1.0 - decay)

    def preprocess_clip_image(self, x, size):
        x = x * 0.5 + 0.5
        x = F.interpolate(x, (size, size), mode='bilinear', antialias=True, align_corners=False)
        return (x - self.clip_mean) / self.clip_std

    def set_input(self, input):
        if self.phase == TrainingPhase.ENCODER:
            # sample via stylegan2
            self.z = torch.randn(self.config['dataset']['batch_size'], 512, device=self.device)
            with torch.no_grad():
                self.real_A = F.hardtanh(self.stylegan2.forward([self.z])[0])
                self.real_B = self.real_A.clone()
        elif self.phase == TrainingPhase.BASE_MODEL:
            if self.config['common']['phase'] == 'test':
                self.real_A = input['test_A'].to(self.device)
                self.real_B = input['test_B'].to(self.device)
                self.image_paths = input['test_A_path']
            else:
                self.real_A = input['unpaired_A'].to(self.device)
                self.real_B = input['unpaired_B'].to(self.device)
                self.image_paths = input['unpaired_A_path']
        elif self.phase == TrainingPhase.CLIP_MAPPING:
            self.real_A = input['unpaired_A'].to(self.device)
            with torch.no_grad():
                self.real_B = self.netG_ema(self.real_A, mixing=random.random() < self.config['training']['style_mixing_prob'])
                self.clip_feats = self.clip_model.encode_image(self.preprocess_clip_image(self.real_B, self.clip_model.visual.input_resolution))
                self.clip_feats /= self.clip_feats.norm(dim=1, keepdim=True)
        elif self.phase == TrainingPhase.FEW_SHOT:
            self.real_A = input['unpaired_A'].to(self.device)
            self.real_B = self.real_A
            if not hasattr(self, 'clip_feats'):
                with torch.no_grad():
                    if os.path.isfile(self.config['training']['image_prompt']):
                        image = self.pil_to_tensor(Image.open(self.config['training']['image_prompt'])).unsqueeze(0).to(self.device)
                        self.clip_feats = self.clip_model.encode_image(image)
                        ref_tokens = self.clip_vit_tokens
                        ref_tokens /= ref_tokens.norm(dim=2, keepdim=True)
                        D = ref_tokens.shape[2]
                        ref_tokens = ref_tokens.reshape(-1, D).permute(1, 0)
                        U, _, _ = torch.linalg.svd(ref_tokens, full_matrices=False)
                        self.UUT = U @ U.permute(1, 0)
                        self.use_image_prompt = True
                    else:
                        text = clip.tokenize(self.config['training']['text_prompt']).to(self.device)
                        self.clip_feats = self.clip_model.encode_text(text)
                        self.use_image_prompt = False
                        # get source text prompt feature
                        text = clip.tokenize(self.config['training']['src_text_prompt']).to(self.device)
                        self.src_clip_feats = self.clip_model.encode_text(text)
                        self.src_clip_feats /= self.src_clip_feats.norm(dim=1, keepdim=True)
                        self.src_clip_feats = self.src_clip_feats.repeat(self.config['dataset']['batch_size'], 1)
                    self.clip_feats /= self.clip_feats.norm(dim=1, keepdim=True)
                    self.clip_feats = self.clip_feats.repeat(self.config['dataset']['batch_size'], 1)
            # get direction in clip space
            with torch.no_grad():
                self.real_A_clip_feats = self.clip_model.encode_image(self.preprocess_clip_image(self.real_A, self.clip_model.visual.input_resolution))
                self.real_A_clip_feats /= self.real_A_clip_feats.norm(dim=1, keepdim=True)
                if self.use_image_prompt:
                    self.src_clip_feats = self.real_A_clip_feats.mean(dim=0, keepdim=True).repeat(self.config['dataset']['batch_size'], 1)
                self.clip_feats_dir = self.clip_feats - self.src_clip_feats

    def forward(self, use_ema=False):
        if self.phase == TrainingPhase.ENCODER:
            if use_ema:
                self.fake_B = self.netG_ema(self.real_A, z=self.z)
            else:
                self.fake_B = self.netG(self.real_A, z=self.z)
        elif self.phase == TrainingPhase.BASE_MODEL:
            if not self.isTrain:
                self.fake_B = self.netG_ema(self.real_A, mixing=False)
            elif use_ema:
                self.fake_B = self.netG_ema(self.real_A, mixing=random.random() < self.config['training']['style_mixing_prob'])
            else:
                self.fake_B = self.netG(self.real_A, mixing=random.random() < self.config['training']['style_mixing_prob'])
        elif self.phase == TrainingPhase.CLIP_MAPPING or self.phase == TrainingPhase.FEW_SHOT:
            if use_ema:
                self.fake_B = self.netG_ema(self.real_A, clip_feats=self.clip_feats)
            else:
                self.fake_B = self.netG(self.real_A, clip_feats=self.clip_feats)

    def backward_D_r1(self):
        self.real_B.requires_grad = True
        if self.data_aug_prob == 0.0:
            real_aug = self.real_B
        else:
            real_aug, _ = augment(self.real_B, self.data_aug_prob)
        real_pred = self.netD(real_aug)
        with conv2d_gradfix.no_weight_gradients():
            grad, = autograd.grad(outputs=real_pred.sum(), inputs=real_aug, create_graph=True)
        r1_loss = grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()
        (r1_loss * self.lambda_r1 / 2 * self.d_reg_freq + 0 * real_pred[0]).backward()

    def backward_D(self, backward=True):
        if self.data_aug_prob == 0.0:
            loss_fake = self.criterionStyleGAN(self.netD(self.fake_B.detach()), False)
            loss_real = self.criterionStyleGAN(self.netD(self.real_B), True)
        else:
            fake_aug, _ = augment(self.fake_B.detach(), self.data_aug_prob)
            real_aug, _ = augment(self.real_B, self.data_aug_prob)
            loss_fake = self.criterionStyleGAN(self.netD(fake_aug), False)
            loss_real = self.criterionStyleGAN(self.netD(real_aug), True)
        self.loss_D = (loss_fake + loss_real) * 0.5

        if backward:
            self.loss_D.backward()

    def backward_G(self, backward=True):
        self.loss_G = 0

        if self.phase == TrainingPhase.ENCODER or self.phase == TrainingPhase.CLIP_MAPPING:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
            with torch.no_grad():
                real_feats = self.netD(self.real_B, rtn_feats=True)
            fake_feats = self.netD(self.fake_B, rtn_feats=True)
            self.loss_G_Feat = sum([ self.criterionL1(fake, real) for fake, real in zip(fake_feats, real_feats) ])

            self.loss_G += self.loss_G_L1 * self.config['training']['lambda_L1']
            self.loss_G += self.loss_G_Feat * self.config['training']['lambda_Feat']

        elif self.phase == TrainingPhase.BASE_MODEL:
            self.loss_G_ST = self.vitLoss.calculate_global_ssim_loss(self.fake_B * 0.5 + 0.5, self.real_A * 0.5 + 0.5)
            if self.data_aug_prob == 0.0:
                self.loss_G_GAN = self.criterionStyleGAN(self.netD(self.fake_B), True)
            else:
                fake_aug, _ = augment(self.fake_B, self.data_aug_prob)
                self.loss_G_GAN = self.criterionStyleGAN(self.netD(fake_aug), True)

            self.loss_G += self.loss_G_ST * self.config['training']['lambda_ST']
            self.loss_G += self.loss_G_GAN * self.config['training']['lambda_GAN']

        elif self.phase == TrainingPhase.FEW_SHOT:
            self.loss_G_ST = self.vitLoss.calculate_global_ssim_loss(self.fake_B * 0.5 + 0.5, self.real_A * 0.5 + 0.5)
            fake_clip_feats = self.clip_model.encode_image(self.preprocess_clip_image(self.fake_B, self.clip_model.visual.input_resolution))
            fake_clip_feats = fake_clip_feats / fake_clip_feats.norm(dim=1, keepdim=True)
            fake_clip_feats_dir = fake_clip_feats - self.real_A_clip_feats
            self.loss_G_CLIP = (1.0 - self.cosineSim(fake_clip_feats_dir, self.clip_feats_dir)).mean()
            if self.use_image_prompt:
                fake_tokens = self.clip_vit_tokens
                fake_tokens = fake_tokens / fake_tokens.norm(dim=2, keepdim=True)
                D = fake_tokens.shape[2]
                fake_tokens = fake_tokens.reshape(-1, D).permute(1, 0)
                self.loss_G_PROJ = self.criterionL1(self.UUT @ fake_tokens, fake_tokens)
            else:
                self.loss_G_PROJ = 0.0

            self.loss_G += self.loss_G_ST * self.config['training']['lambda_ST']
            self.loss_G += self.loss_G_CLIP * self.config['training']['lambda_CLIP']
            self.loss_G += self.loss_G_PROJ * self.config['training']['lambda_PROJ']

        if backward:
            self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()

        if not self.phase == TrainingPhase.BASE_MODEL:
            # only G
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

            # update G_ema
            self.ema(self.netG_ema, self.netG, decay=self.config['training']['ema'])

        else:
            # G
            self.set_requires_grad([self.netD], False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            # D
            self.set_requires_grad([self.netD], True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

            # update G_ema
            self.ema(self.netG_ema, self.netG, decay=self.config['training']['ema'])

            # r1 reg
            self.step += 1
            if self.step % self.d_reg_freq == 0:
                self.optimizer_D.zero_grad()
                self.backward_D_r1()
                self.optimizer_D.step()

    def eval_step(self):
        self.forward(use_ema=True)
        self.backward_G(False)
        if self.phase == TrainingPhase.BASE_MODEL:
            self.backward_D(False)
        self.step += 1

    def trace_jit(self, input):
        self.netG = self.netG.module.cpu()
        traced_script_module = torch.jit.trace(self.netG, input)
        dummy_output = self.netG_ema(input)
        dummy_output_traced = traced_script_module(input)
        return traced_script_module, dummy_output, dummy_output_traced

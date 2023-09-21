import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import clip
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from models.style_based_pix2pixII_model import Stylizer, TrainingPhase

if __name__ == '__main__':
    # define & parse args
    parser = argparse.ArgumentParser(description='sp2pII test')
    parser.add_argument('--ckpt', type=str, default='') # ./checkpoints/watercolor_painting.pth
    parser.add_argument('--in_folder', type=str, default='') # ./example/source
    parser.add_argument('--out_folder', type=str, default='') # ./example/outputs/zero-shot/watercolor_painting
    parser.add_argument('--phase', type=int, default=3)
    parser.add_argument('--txt_prompt', type=str, default='') # watercolor painting
    parser.add_argument('--img_prompt', type=str, default='') # ./example/reference/04.png 
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    args.phase = TrainingPhase(args.phase)

    os.makedirs(args.out_folder, exist_ok=True)

    # init model
    state_dict = torch.load(args.ckpt, map_location='cpu')
    model = Stylizer(ngf=64, phase=args.phase, model_weights=state_dict['G_ema_model'])
    model.to(args.device)
    model.eval()
    model.requires_grad_(False)

    clip_model, img_preprocess = clip.load('ViT-B/32', device=args.device)
    clip_model.eval()
    clip_model.requires_grad_(False)

    # image transform for stylizer
    img_transform = Compose([
        Resize((512, 512), interpolation=InterpolationMode.LANCZOS),
        ToTensor(),
        Normalize([0.5], [0.5])
    ])

    # get clip features
    with torch.no_grad():
        if os.path.isfile(args.img_prompt):
            img = img_preprocess(Image.open(args.img_prompt)).unsqueeze(0).to(args.device)
            clip_feats = clip_model.encode_image(img)
        else:
            text = clip.tokenize(args.txt_prompt).to(args.device)
            clip_feats = clip_model.encode_text(text)
        clip_feats /= clip_feats.norm(dim=1, keepdim=True)

    # enum image files
    files = os.listdir(args.in_folder)
    for fn in tqdm(files):
        prefix, ext = os.path.splitext(fn)
        if not ext.lower() in ['.png', '.jpg', '.jpeg']:
            continue

        # load image & to tensor
        img = Image.open(os.path.join(args.in_folder, fn))
        if not img.mode == 'RGB':
            img = img.convert('RGB')
        img = img_transform(img).unsqueeze(0).to(args.device)

        # stylize it !
        with torch.no_grad():
            if args.phase == TrainingPhase.CLIP_MAPPING:
                res = model(img, clip_feats=clip_feats)

        # save image
        res = res.cpu().numpy()[0]
        res = np.transpose(res, (1, 2, 0)) * 0.5 + 0.5
        Image.fromarray((res * 255).astype(np.uint8)).save(os.path.join(args.out_folder, prefix + '.png'))

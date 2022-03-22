"""Generate images using pretrained network pickle."""

import os
from typing import List, Optional

import click
import numpy as np
import PIL.Image
import torch

from training.data.gen_loader import get_loader

import warnings
warnings.filterwarnings("ignore")
from colorama import init
from colorama import Fore, Style
from icecream import ic
init(autoreset=True)
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import yaml

#----------------------------------------------------------------------------
    
def denormalize(tensor):
    pixel_mean = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)
    pixel_std = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)
    denormalizer = lambda x: torch.clamp((x * pixel_std) + pixel_mean, 0, 255.)

    return denormalizer(tensor)

def visualize_gen(i, img, inv_mask, msk_type):
    lo, hi = [-1, 1]
    
    comp_img = np.asarray(img[0], dtype=np.float32).transpose(1, 2, 0)
    comp_img = (comp_img - lo) * (255 / (hi - lo))
    comp_img = np.rint(comp_img).clip(0, 255).astype(np.uint8)
    plt.imsave(f'{msk_type}/' + i + '.png', comp_img / 255)
    
    inv_mask = torch.stack([inv_mask[0] * torch.tensor(255.)]*3, dim=0).squeeze(1)
    inv_mask = np.asarray(inv_mask, dtype=np.float32).transpose(1, 2, 0)
    inv_mask = np.rint(inv_mask).clip(0, 255).astype(np.uint8)

    mask = PIL.Image.fromarray(inv_mask)
    mask.save(f'{msk_type}/' + i + '_mask000.png')
    plt.close()


def create_folders(msk_type):
    if not os.path.exists(f'{msk_type}'):
        os.makedirs(f'{msk_type}')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--img_data', help='Training images (directory)', metavar='PATH', required=True)
@click.option('--msk_type', help='mask description', required=True)
@click.option('--lama_cfg', help='lama mask config')
@click.option('--msk_ratio', help='comodgan mask ratio', multiple=True)
@click.option('--resolution', help='Res of Images [default: 256]', type=int, metavar='INT')
@click.option('--num', help='Number of Images [default: 10]', type=int, metavar='INT')
def generate_images(
    ctx: click.Context,
    img_data: str,
    msk_type: str,
    msk_ratio: list,
    lama_cfg: str,
    resolution: int,
    num: int,
):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if resolution is None:
        resolution = 256
    
    if lama_cfg is not None:
        lama_cfg = yaml.safe_load(open(lama_cfg))
        msk_ratio = None
    
    if msk_ratio is not None:
        msk_ratio = [float(x) for x in msk_ratio]
    
    if lama_cfg is None and msk_ratio is None:
        msk_ratio = [0.7, 0.9]

    
    dataloader = get_loader(img_path=img_data, resolution=resolution, msk_ratio=msk_ratio, lama_cfg=lama_cfg)

    create_folders(msk_type)
    
    for _, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc='Generating Evaluation data...'):

        images, _, invisible_masks, fnames = data

        mask  = invisible_masks
        fname = fnames[0]

        visualize_gen(fname, images, mask, msk_type)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

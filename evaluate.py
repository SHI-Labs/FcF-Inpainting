"""Generate images using pretrained network pickle."""

import os
from typing import List, Optional

import click
from numpy.lib.type_check import imag
import dnnlib
import numpy as np
import torch
import tempfile
import legacy
import random

from training.data.pred_loader import ImageDataset

import warnings
warnings.filterwarnings("ignore")
from colorama import init
from colorama import Fore, Style
from icecream import ic
init(autoreset=True)
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_utils import training_stats
from torch_utils import custom_ops
import copy
import pandas as pd

from metrics.evaluation.data import PrecomputedInpaintingResultsDataset
from metrics.evaluation.evaluator import InpaintingEvaluator
from metrics.evaluation.losses.base_loss import SSIMScore, LPIPSScore, FIDScore
from metrics.evaluation.utils import load_yaml

#----------------------------------------------------------------------------

def save_image(img, name):
    x = denormalize(img.detach().cpu())
    x = x.permute(1, 2, 0).numpy()
    x = np.rint(x) / 255.
    plt.imsave(name+'.png', x)
    
def denormalize(tensor):
    pixel_mean = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)
    pixel_std = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)
    denormalizer = lambda x: torch.clamp((x * pixel_std) + pixel_mean, 0, 255.)

    return denormalizer(tensor)

def visualize_gen(i, msk_type, comp_img):
    lo, hi = [-1, 1]
    
    comp_img = np.asarray(comp_img[0].cpu(), dtype=np.float32).transpose(1, 2, 0)
    comp_img = (comp_img - lo) * (255 / (hi - lo))
    comp_img = np.rint(comp_img).clip(0, 255).astype(np.uint8)
    plt.imsave(f'fid_gens/{msk_type}/' + i + '000.png', comp_img / 255)
    plt.close()


def create_folders(msk_type):
    if not os.path.exists(f'fid_gens/{msk_type}'):
        os.makedirs(f'fid_gens/{msk_type}')


def save_gen(G, rank, num_gpus, device, img_data, resolution, label, truncation_psi, msk_type):
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    dataset = ImageDataset(img_data, resolution)
    num_items = len(dataset)

    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    # Main loop.
    item_subset = [(i * num_gpus + rank) % num_items for i in range((num_items - 1) // num_gpus + 1)]
    dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=1, **data_loader_kwargs)
    for _, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f'Visualizing on GPU: {rank}'):

        with torch.no_grad():
            ## data is a tuple of (rgbs, rgbs_erased, amodal_tensors, visible_mask_tensors, erased_modal_tensors) ####
            images, erased_images, invisible_masks, fnames = data

            erased_img = erased_images.to(device)
            mask  = invisible_masks.to(device)
            fname = fnames[0]
            pred_img = G(img=torch.cat([0.5 - mask, erased_img], dim=1), c=label, truncation_psi=truncation_psi, noise_mode='const')
            comp_img = invisible_masks.to(device) * pred_img + (1 - invisible_masks).to(device) * images.to(device)
            visualize_gen(fname, msk_type, comp_img.detach())

def run_gen(rank, num_gpus, temp_dir, G, img_data, resolution, label, truncation_psi):
    # Init torch.distributed.
    if num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    custom_ops.verbosity = 'none'

    # Print network summary.
    device = torch.device('cuda', rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    
    eval_config = load_yaml('metrics/configs/eval2_gpu.yaml')
    msk_type = img_data.split('/')[-1]
    label = label.to(device)
    
    if rank == 0:
        netG_params = sum(p.numel() for p in G.parameters())
        print(Fore.BLUE +"Generator Params: {} M".format(netG_params/1e6))
        print(Style.BRIGHT + Fore.GREEN + "Starting Visualization...")
        create_folders(msk_type)
    
    save_gen(G, rank, num_gpus, device, img_data, resolution, label, truncation_psi, msk_type)
    
    if rank == 0:
        eval_dataset = PrecomputedInpaintingResultsDataset(img_data, f'fid_gens/{msk_type}', **eval_config.dataset_kwargs)
        metrics = {
            'ssim': SSIMScore(),
            'lpips': LPIPSScore(),
            'fid': FIDScore()
        }
        evaluator = InpaintingEvaluator(eval_dataset, scores=metrics, area_grouping=True,
                                integral_title='lpips_fid100_f1', integral_func=None,
                                **eval_config.evaluator_kwargs)
        results = evaluator.dist_evaluate(device, num_gpus=1, rank=0)
        results = pd.DataFrame(results).stack(1).unstack(0)
        results.dropna(axis=1, how='all', inplace=True)
        results.to_csv(f'fid_gens/{msk_type}.csv', sep='\t', float_format='%.4f')
        print(results)
    

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--img_data', help='Training images (directory)', metavar='PATH', required=True)
@click.option('--resolution', help='Res of Images [default: 256]', type=int, metavar='INT')
@click.option('--num_gpus', help='Number of gpus [default: 1]', type=int, metavar='INT')

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    img_data: str,
    resolution: int,
    num_gpus: int,
    class_idx: Optional[int],
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
    
    if num_gpus is None:
        num_gpus = 1

    if not num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    # Load network.
    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        ctx.fail('--network must point to a file or URL')

    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    
    # Labels.
    label = torch.zeros([1, G.c_dim])
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')
    
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if num_gpus == 1:
            run_gen(rank=0, num_gpus=num_gpus, temp_dir=temp_dir, G=G, 
                        img_data=img_data, resolution=resolution, label=label, truncation_psi=truncation_psi)
        else:
            torch.multiprocessing.spawn(fn=run_gen, args=(num_gpus, temp_dir, G, img_data, resolution, label, truncation_psi), nprocs=num_gpus)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

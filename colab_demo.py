from typing import Tuple
import dnnlib
from PIL import Image
import numpy as np
import torch
import legacy
from torchvision.transforms import ToTensor, ToPILImage
import argparse
import os
import cv2

image_to_tensor = ToTensor()
tensor_to_image = ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_idx = None
truncation_psi = 0.1

def file_ext(fname):
    return os.path.splitext(fname)[1].lower()

def create_model(network_pkl):
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    
    G = G.eval().to(device)
    netG_params = sum(p.numel() for p in G.parameters())
    print("Generator Params: {} M".format(netG_params/1e6))
    return G

def fcf_inpaint(G, org_img, erased_img, mask):
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ValueError("class_idx can't be None.")
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')
    
    pred_img = G(img=torch.cat([0.5 - mask, erased_img], dim=1), c=label, truncation_psi=truncation_psi, noise_mode='const')
    comp_img = mask.to(device) * pred_img + (1 - mask).to(device) * org_img.to(device)
    return comp_img

def show_images(img):
    """ Display a batch of images inline. """
    return Image.fromarray(img)

def denorm(img):
    img = np.asarray(img[0].cpu(), dtype=np.float32).transpose(1, 2, 0)
    img = (img +1) * 127.5
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    return img

def pil_to_numpy(pil_img: Image) -> Tuple[torch.Tensor, torch.Tensor]:
    img = np.array(pil_img)
    return torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1

def inpaint(input_img, mask, ckpt):
    mask = mask.convert('L')
    mask = np.array(mask) / 255.
    mask = cv2.resize(mask,
        (512, 512), interpolation=cv2.INTER_NEAREST)

    mask_tensor = torch.from_numpy(mask).to(torch.float32)
    mask_tensor = mask_tensor.unsqueeze(0)
    mask_tensor = mask_tensor.unsqueeze(0).to(device)

    rgb = input_img.convert('RGB')
    rgb = np.array(rgb)
    rgb = cv2.resize(rgb,
        (512, 512), interpolation=cv2.INTER_AREA)
    rgb = rgb.transpose(2,0,1)
    rgb = torch.from_numpy(rgb.astype(np.float32)).unsqueeze(0)
    rgb = (rgb.to(torch.float32) / 127.5 - 1).to(device)
    rgb_erased = rgb.clone()
    rgb_erased = rgb_erased * (1 - mask_tensor) # erase rgb
    rgb_erased = rgb_erased.to(torch.float32)
    
    model = create_model(ckpt)
    comp_img = fcf_inpaint(G=model, org_img=rgb.to(torch.float32), erased_img=rgb_erased.to(torch.float32), mask=mask_tensor.to(torch.float32))
    rgb_erased = denorm(rgb_erased)
    comp_img = denorm(comp_img)

    comp_img = show_images(comp_img)
    return comp_img

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FcF-Inpainting Demo')
    parser.add_argument('--img_path', type=str,
                        help='path to img')
    parser.add_argument('--output', type=str,
                        help='output dir')
    parser.add_argument('--ckpt', type=str,
                        help='checkpoint path')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    img = Image.open(args.img_path)
    name = args.img_path.split('/')[-1]
    ext = file_ext(name)
    mask = Image.open(args.img_path.replace(ext, f'_mask{ext}'))
    comp_img = inpaint(img, mask, args.ckpt)
    comp_img.save(os.path.join(args.output, name))
from typing import Tuple
import dnnlib
from PIL import Image
import numpy as np
import torch
import legacy
import cv2
from streamlit_drawable_canvas import st_canvas
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_idx = None
truncation_psi = 0.1

title = "FcF-Inpainting"

description = "<p style='color:royalblue; font-size: 14px; font-weight: w300;'>  \
                [Note: The Inpainted Image display may take up to a minute depending on the Queue. The image and mask are resized to 512x512 before inpainting. The <span style='color:#E0B941;'>Run FcF-Inpainting</span> button will automatically appear after you draw a mask.] To use FcF-Inpainting: <br> \
                (1) <span style='color:#E0B941;'>Upload an Image</span> or <span style='color:#E0B941;'> select a sample image on the left</span>. <br>  \
                (2) Adjust the brush stroke width and <span style='color:#E0B941;'>draw the mask on the image</span>. You may also change the drawing tool on the sidebar. <br>\
                (3) After drawing a mask, click the <span style='color:#E0B941;'>Run FcF-Inpainting</span> and witness the MAGIC! 🪄 ✨ ✨<br> \
                (4) You may <span style='color:#E0B941;'>download/undo/redo/delete</span> the changes on the image using the options below the image box.</p>"

article = "<p style='color: #E0B941; font-size: 16px; font-weight: w500; text-align: center'> <a style='color: #E0B941;' href='https://praeclarumjj3.github.io/fcf-inpainting/' target='_blank'>Project Page</a> | <a style='color: #E0B941;' href='https://arxiv.org/abs/2208.03382' target='_blank'> Keys to Better Image Inpainting: Structure and Texture Go Hand in Hand</a> | <a style='color: #E0B941;' href='https://github.com/SHI-Labs/FcF-Inpainting' target='_blank'>Github</a></p>"

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


def denorm(img):
    img = np.asarray(img[0].cpu(), dtype=np.float32).transpose(1, 2, 0)
    img = (img +1) * 127.5
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    return img

def pil_to_numpy(pil_img: Image) -> Tuple[torch.Tensor, torch.Tensor]:
    img = np.array(pil_img)
    return torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1

def process_mask(input_img, mask):
    rgb = cv2.cvtColor(input_img, cv2.COLOR_RGBA2RGB)
    mask = 255 - mask[:,:,3]
    mask = (mask > 0) * 1

    rgb = np.array(rgb)
    mask_tensor = torch.from_numpy(mask).to(torch.float32)
    mask_tensor = mask_tensor.unsqueeze(0)
    mask_tensor = mask_tensor.unsqueeze(0).to(device)

    rgb = rgb.transpose(2,0,1)
    rgb = torch.from_numpy(rgb.astype(np.float32)).unsqueeze(0)
    rgb = (rgb.to(torch.float32) / 127.5 - 1).to(device)
    rgb_erased = rgb.clone()
    rgb_erased = rgb_erased * (1 - mask_tensor) # erase rgb
    rgb_erased = rgb_erased.to(torch.float32)
    
    rgb_erased = denorm(rgb_erased)
    return rgb_erased

def inpaint(input_img, mask, model):
    rgb = cv2.cvtColor(input_img, cv2.COLOR_RGBA2RGB)
    mask = 255 - mask[:,:,3]
    mask = (mask > 0) * 1

    rgb = np.array(rgb)
    mask_tensor = torch.from_numpy(mask).to(torch.float32)
    mask_tensor = mask_tensor.unsqueeze(0)
    mask_tensor = mask_tensor.unsqueeze(0).to(device)

    rgb = rgb.transpose(2,0,1)
    rgb = torch.from_numpy(rgb.astype(np.float32)).unsqueeze(0)
    rgb = (rgb.to(torch.float32) / 127.5 - 1).to(device)
    rgb_erased = rgb.clone()
    rgb_erased = rgb_erased * (1 - mask_tensor) # erase rgb
    rgb_erased = rgb_erased.to(torch.float32)
    
    comp_img = fcf_inpaint(G=model, org_img=rgb.to(torch.float32), erased_img=rgb_erased.to(torch.float32), mask=mask_tensor.to(torch.float32))
    rgb_erased = denorm(rgb_erased)
    comp_img = denorm(comp_img)
    return comp_img

def run_app(model):
    
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    image_inpainting(model)

    with st.sidebar:
        st.markdown("---")

def image_inpainting(model):
    if 'reuse_image' not in st.session_state:
        st.session_state.reuse_image = None
    
    st.title(title)
    st.markdown(article, unsafe_allow_html=True)
    st.markdown(description, unsafe_allow_html=True)

    image = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    sample_image = st.sidebar.radio('Choose a Sample Image', [
            'wall-1.jpeg',
            'wall-2.jpeg',
            'house.jpeg',
            'door.jpeg',
            'floor.jpeg',
            'church.jpeg',
            'person-cliff.jpeg',
            'person-fence.png',
            'persons-white-fence.jpeg',
        ])
    
    drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line")
)

    image = Image.open(image).convert("RGBA") if image else Image.open(f"./test_512/{sample_image}").convert("RGBA")        
    image = image.resize((512, 512))
    width, height = image.size
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 100, 20)

    canvas_result = st_canvas(
        stroke_color="rgba(255, 0, 255, 0.8)",
        stroke_width=stroke_width,
        background_image=image,
        height=height,
        width=width,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    if canvas_result.image_data is not None and image and len(canvas_result.json_data["objects"]) > 0:
        
        im = canvas_result.image_data.copy()
        background = np.where(
            (im[:, :, 0] == 0) & 
            (im[:, :, 1] == 0) & 
            (im[:, :, 2] == 0)
        )
        drawing = np.where(
            (im[:, :, 0] == 255) & 
            (im[:, :, 1] == 0) & 
            (im[:, :, 2] == 255)
        )
        im[background]=[0,0,0,255]
        im[drawing]=[0,0,0,0] #RGBA
        if st.button('Run FcF-Inpainting'):
            col1, col2 = st.columns([1,1])
            with col1:
                # if st.button('Show Image with Holes'):
                st.write("Masked Image")
                mask_show = process_mask(np.array(image), np.array(im))
                st.image(mask_show)
            with col2:
                st.write("Inpainted Image")
                inpainted_img = inpaint(np.array(image), np.array(im), model)
                st.image(inpainted_img)

if __name__ == "__main__":
    st.set_page_config(
        page_title="FcF-Inpainting", page_icon=":sparkles:"
    )
    st.sidebar.subheader("Configuration")
    model = create_model("models/places_512.pkl")
    run_app(model)
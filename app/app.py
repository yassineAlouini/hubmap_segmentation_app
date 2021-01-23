"""Streamlit web app for glomuerli segmentation"""

import albumentations as albu
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import pad, unpad
from collections import namedtuple
from torch import nn
from torch.utils import model_zoo
from iglovikov_helper_functions.dl.pytorch.utils import rename_layers

from segmentation_models_pytorch import Unet
from pathlib import Path

st.set_option("deprecation.showfileUploaderEncoding", False)

MAX_SIZE = 512
MODEL_PATH = Path(__file__).parent / "model.pth"

def load_model() -> nn.Module:
    model = Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None)
    # TODO: While working on a fix, don't load for now :p
    # state_dict = torch.load(MODEL_PATH)["state_dict"]
    # state_dict = rename_layers(state_dict, {"model.": ""})
    # model.load_state_dict(state_dict)
    return model



@st.cache(allow_output_mutation=True)
def cached_model():
    model = load_model()
    model.eval()
    return model


model = cached_model()
transform = albu.Compose(
    [albu.LongestMaxSize(max_size=MAX_SIZE), albu.Normalize(p=1)], p=1
)

st.title("Segment glomeruli")
# What about a TIFF image?
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    original_image = np.array(Image.open(uploaded_file))
    st.image(original_image, caption="Before", use_column_width=True)
    st.write("")
    st.write("Detecting glomeruli...")

    original_height, original_width = original_image.shape[:2]
    image = transform(image=original_image)["image"]
    padded_image, pads = pad(image, factor=MAX_SIZE, border=cv2.BORDER_CONSTANT)

    x = torch.unsqueeze(tensor_from_rgb_image(padded_image), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    mask = cv2.resize(
        mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST
    )
    mask_3_channels = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    dst = cv2.addWeighted(
        original_image, 1, (mask_3_channels * (0, 255, 0)).astype(np.uint8), 0.5, 0
    )

    st.image(mask * 255, caption="Mask", use_column_width=True)
    st.image(dst, caption="Image + mask", use_column_width=True)

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

from Denoiser.model import load_model as load_denoiser
from Colour.colour_model import load_model as load_colorizer

PATCH_SIZE = 128
OVERLAP = 32


def make_weight_mask(p):
    w = np.maximum(np.hanning(p), 1e-3)
    return np.outer(w, w)[..., None].astype(np.float32)


def patch_inference_denoiser(img, model):
    pad = PATCH_SIZE // 2
    img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    H, W = img.shape[:2]
    stride = PATCH_SIZE - OVERLAP

    test = model(img[:PATCH_SIZE, :PATCH_SIZE][None], training=False)[0]
    c = test.shape[-1]

    out = np.zeros((H, W, c), np.float32)
    wsum = np.zeros_like(out)

    w = make_weight_mask(PATCH_SIZE)
    w = np.repeat(w, c, axis=-1)

    for y in range(0, H - PATCH_SIZE + 1, stride):
        for x in range(0, W - PATCH_SIZE + 1, stride):
            p = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            pred = model(p[None], training=False)[0].numpy()
            out[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred * w
            wsum[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += w

    out /= np.maximum(wsum, 1e-6)
    return out[pad:-pad, pad:-pad]


def rgb_to_lab(img):
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def lab_to_rgb(lab):
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def pad_to_multiple(img, f=32):
    H, W = img.shape[:2]
    H2 = ((H + f - 1) // f) * f
    W2 = ((W + f - 1) // f) * f
    return np.pad(img, ((0, H2-H), (0, W2-W), (0, 0)), "reflect"), H, W


def colorize_image(img, model):
    lab = rgb_to_lab(img)
    L = lab[..., :1] / 255.0
    Lp, H, W = pad_to_multiple(L)
    ab = model(Lp[None], training=False)[0].numpy()[:H, :W]
    lab[..., 1:] = np.clip(ab * 128 + 128, 0, 255)
    return lab_to_rgb(lab)


st.set_page_config("Image Restoration", layout="wide")
st.title("🖼️ Image Restoration")

task = st.selectbox("Task", ["Denoise", "Colorize"])
file = st.file_uploader("Upload", ["png", "jpg", "jpeg"])


@st.cache_resource
def get_denoiser():
    return load_denoiser()


@st.cache_resource
def get_colorizer():
    return load_colorizer()


if file:
    img = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1),
        cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255.0

    c1, c2 = st.columns(2)
    c1.image(img, caption="Input", use_column_width=True)

    with st.spinner("Processing"):
        if task == "Denoise":
            model = get_denoiser()
            noise = patch_inference_denoiser(img, model)
            if noise.shape[-1] == 1:
                noise = np.repeat(noise, 3, -1)
            out = np.clip(img - noise, 0, 1)
        else:
            model = get_colorizer()
            out = np.clip(colorize_image(img, model), 0, 1)

    c2.image(out, caption="Output", use_column_width=True)

    buf = cv2.imencode(".png", cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))[1]
    st.download_button("Download Result", buf.tobytes(), "result.png", "image/png")


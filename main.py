import streamlit as st
from PIL import Image

from options.test_options import TestOptions
from models import create_model
from data.base_dataset import get_transform
from util import util


def generate(image):
    opt = TestOptions().parse()

    opt.dataroot = 'cond_train/testA'
    opt.name = 'animal2pkmn_cond'
    opt.model = 'test'
    opt.netG = 'unet_128'

    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    transform = get_transform(opt, grayscale=False)
    image = transform(image).unsqueeze(0)

    data = {'A': image, 'A_paths': None}
    model = create_model(opt)
    model.setup(opt)

    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    return util.tensor2im(visuals['fake'])


st.title("GAN in Animal")

st.write('''
        This is a GAN model that can transfer the style of a Pokemon to a real animal.
        The model works better with real animal images with white background.
        Note: The result Pokemon image's quality is not promised to be good. Since this is a short project, model is only trained for 200 epochs.
         ''')

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "png", "jpeg"])
col1, col2 = st.columns([0.5, 0.5])
with col1:
    st.markdown('<p style="text-align: center;">Original Image</p>',
                unsafe_allow_html=True)
with col2:
    st.markdown('<p style="text-align: center;">Pokified Image</p>',
                unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    with col1:
        st.image(image, width=300)

    with col2:
        visuals = generate(image)
        st.image(visuals, width=300)

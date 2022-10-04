import streamlit as st
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

#from werkzeug.utils import secure_filename
import logging
from pathlib import Path
from utils import rotate_image, get_angle

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg'}

CWD = Path(os.getcwd())
UPLOAD_DIRECTORY = CWD / "images_app"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def main_loop():

    st.title('Image preprocessing')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None
    original_image = Image.open(image_file)
    original_image = np.array(original_image)
    skew_angle_in_degrees = get_angle(original_image)
    rotated_image = rotate_image(original_image, -skew_angle_in_degrees)

    st.text("Original Image")
    st.image(original_image)

    st.text("Rotated Image (angle : {:0.2f}°)".format(skew_angle_in_degrees))
    st.image(rotated_image)


if __name__ == '__main__':
    main_loop()


import os
import cv2
import numpy as np
import base64
import requests
import logging
import streamlit as st
import yaml

from PIL import Image
from typing import List

logger = logging.getLevelName(__name__)

with open("./configs/streamlit.yaml", 'r') as stream:
    streamlit_config = yaml.safe_load(stream)

st.title("Generate 2D Game Assets")
existen_models = streamlit_config["existing_model"]


def train_request(uploaded_files: List[str], asset_type: str):
    url = streamlit_config["backend_hostname"] + \
        streamlit_config["train_endpoint"]
    files = [
        ('images', file.getvalue())
        for file in uploaded_files
    ]
    result = requests.post(
        url,
        files=files,
        data={"asset_type": asset_type}
    )
    return result


def generate_request(asset_type: str):
    url = streamlit_config["backend_hostname"] + \
        streamlit_config["generate_endpoint"] + f"/{asset_type}/"
    return requests.post(url)


if __name__ == "__main__":
    uploaded_files = st.file_uploader(
        "Please choose a file",
        accept_multiple_files=True
    )

    datasets_names = os.listdir(existen_models)

    if uploaded_files:
        n = st.slider(
            label="Select a number of image in raw",
            min_value=1,
            max_value=len(uploaded_files) + 1,
            value=(len(uploaded_files) + 1) // 2
        )

        groups = []
        for i in range(0, len(uploaded_files), n):
            groups.append(uploaded_files[i: i + n])

        for group in groups:
            cols = st.columns(n)
            for i, image_file in enumerate(group):
                cols[i].image(image_file)

    with st.sidebar:
        if len(datasets_names) != 0:
            selected_dataset = st.selectbox(
                label="Select dataset",
                options=datasets_names
            )

            if selected_dataset:
                if st.button("Generate"):
                    gen_response = generate_request(selected_dataset)

        if uploaded_files:
            asset_type = st.text_input(label="Asset type")
            if asset_type != "":
                if st.button("Train"):
                    train_response = train_request(uploaded_files, asset_type)

    try:
        if gen_response.status_code == 201:
            nparr = np.frombuffer(base64.b64decode(
                gen_response.json()["image"]
            ), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imwrite("1.jpg", img)
            pil_img = Image.fromarray(img)
            st.image(pil_img)
    except:
        pass

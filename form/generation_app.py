import os
import requests
import logging
import streamlit as st
import yaml

from typing import List

logger = logging.getLevelName(__name__)

with open("./configs/streamlit.yaml", 'r') as stream:
    streamlit_config = yaml.safe_load(stream)

st.title("Generate 2D Game Assets")
dataset_path = streamlit_config["dataset_path"]


def upload_request(uploaded_files: List[str]):
    url = streamlit_config["backend_hostname"] + \
        streamlit_config["upload_endpoint"]
    files = [
        ('images', file.getvalue())
        for file in uploaded_files
    ]
    result = requests.post(url, files=files)
    return result


if __name__ == "__main__":
    uploaded_files = st.file_uploader(
        "Please choose a file",
        accept_multiple_files=True
    )

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

        result = upload_request(uploaded_files)

    with st.sidebar:
        datasets_names = os.listdir(dataset_path)
        selected_dataset = st.selectbox(
            label="Select dataset",
            options=datasets_names
        )

        if st.button("Start Training"):
            st.write("PReSsed")

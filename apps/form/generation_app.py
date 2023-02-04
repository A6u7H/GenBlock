import os
import requests
import streamlit as st

st.title("Generate ")
dataset_path = "/home/dkrivenkov/program/genlock/pipeline/data"

if __name__ == "__main__":
    # col1, col2, col3 = st.columns(3)
    uploaded_files = st.file_uploader(
        "Please choose a file",
        accept_multiple_files=True
    )
    if uploaded_files:
        myurl = 'http://127.0.0.1:8000/api/v1/upload'
        files = [
            ('images', file.getvalue()) for file in uploaded_files
        ]
        res = requests.post(myurl, files=files)
    with st.sidebar:
        datasets_names = os.listdir(dataset_path)
        selected_dataset = st.selectbox(
            label="Select dataset",
            options=datasets_names
        )

        if st.button("Start Training"):
            st.write("PReSsed")

import unittest
import torch
import cv2
import os

from transformers import CLIPTokenizer
from PIL import Image

from pipeline.dataset import (
    DreamBoothDataset,
    ImageTransform,
    PreprocessorTransform,
    Preprocessor
)


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        data_root = "./pipeline/data/Shields"
        size = (256, 256)

        name_of_your_concept = "shieldd"
        type_of_thing = "shield from 2D game"
        instance_prompt = f"a picture of {name_of_your_concept} {type_of_thing}"

        model_id = "CompVis/stable-diffusion-v1-4"
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
        )

        image_transform = ImageTransform(size)
        images = [
            Image.open(os.path.join(data_root, path))
            for path in os.listdir(data_root)
        ]
        dataset = DreamBoothDataset(
            images,
            instance_prompt,
            tokenizer,
            image_transform
        )

        self.assertEqual(type(dataset[0]), dict)
        self.assertEqual(dataset[0]["instance_images"].shape, (4, 256, 256))
        self.assertEqual(dataset[0]["instance_prompt_ids"].dim(), 1)

    def test_preprocessor(self):
        src_data_path = "./pipeline/data/Shields"
        tgt_data_path = "./pipeline/data/Shields_preprocessed"
        os.mkdir(tgt_data_path)
        size = (256, 256)
        background_color = (127.5, 127.5, 127.5)

        image_transform = PreprocessorTransform(background_color, size)
        preprocessor = Preprocessor(image_transform)

        preprocessor.preprocess_data(src_data_path, tgt_data_path)


if __name__ == '__main__':
    unittest.main()

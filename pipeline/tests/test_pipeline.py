import unittest
import torch
import cv2

from transformers import CLIPTokenizer

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
        type_of_thing = "shield from game"
        instance_prompt = f"a picture of {name_of_your_concept} {type_of_thing}"

        model_id = "CompVis/stable-diffusion-v1-4"
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
        )

        image_transform = ImageTransform(size)
        dataset = DreamBoothDataset(
            data_root,
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
        size = (256, 256)
        background_color = (0.0, 0.0, 0.0)

        image_transform = PreprocessorTransform(background_color, size)
        preprocessor = Preprocessor(image_transform)

        preprocessor.preprocess_data(src_data_path, tgt_data_path)


if __name__ == '__main__':
    unittest.main()

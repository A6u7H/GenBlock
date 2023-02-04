import os

from torch.utils.data import Dataset
from PIL import Image

from typing import Callable


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        instance_prompt: str,
        tokenizer: Callable,
        transforms: Callable
    ):
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.transforms = transforms

        file_names = os.listdir(data_path)
        self.images = list(map(
            lambda x: os.path.join(data_path, x),
            file_names
        ))

    def __getitem__(self, index):
        example = {}
        image_path = self.images[index]
        image = Image.open(image_path)

        prompt_seq = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        example["instance_images"] = self.transforms(image)
        example["instance_prompt_ids"] = prompt_seq
        return example

    def __len__(self):
        return len(self.dataset)

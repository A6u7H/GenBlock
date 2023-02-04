import torchvision.transforms as T
import numpy as np
import cv2

from torch import Tensor
from PIL import Image
from typing import Tuple


class PreprocessorTransform:
    def __init__(
        self,
        background_color: Tuple[float, float, float],
        size: Tuple[int, int]
    ) -> None:
        self.size = size
        self.background_color = background_color

    def add_padding(self, image):
        h, w, _ = image.shape
        padding = abs((h - w) // 2)
        return cv2.copyMakeBorder(
            image,
            0,
            0,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

    def __call__(self, image: np.ndarray):
        image = self.add_padding(image)
        image = cv2.resize(
            image,
            self.size,
            interpolation=cv2.INTER_NEAREST
        )
        image[image.sum(2) == 0] = self.background_color
        return image

class ImageTransform:
    def __init__(self, size: Tuple[int, int]) -> None:
        self.transpose = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1)  # replace with Normilize()
        ])

    def __call__(self, image: Image) -> Tensor:
        return self.transpose(image)


class ImageReverseTransform:
    def __init__(self) -> None:
        self.transpose = T.Compose([
            T.Lambda(lambda x: (x + 1) * 0.5),
            T.Lambda(lambda x: x.permute(1, 2, 0)),
            T.Lambda(lambda x: (x * 255)),
            T.Lambda(lambda x: x.numpy().astype(np.uint8)),
        ])

    def __call__(self, image: Image) -> Tensor:
        return self.transpose(image)

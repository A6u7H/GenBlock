import io
import torch
import yaml

from PIL import Image
from typing import Tuple, Callable, List
from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    DDPMScheduler
)
from rest_framework.parsers import (
    MultiPartParser,
    JSONParser
)

from pipeline.trainer import DreamboothTrainer
from pipeline.dataset import (
    DreamBoothDataset,
    DreamboothCollate,
    ImageTransform
)

with open("./configs/models.yaml", 'r') as stream:
    models_config = yaml.safe_load(stream)

with open("./configs/trainer.yaml", 'r') as stream:
    trainer_config = yaml.safe_load(stream)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class TrainModelView(APIView):
    parser_classes = (
        MultiPartParser,
        JSONParser,
    )

    @staticmethod
    def train(
        tokenizer: Callable,
        text_encoder: Callable,
        vae: Callable,
        unet: Callable,
        noise_scheduler: Callable,
        feature_extractor: Callable,
        instance_prompt: str,
        files: List[bytes],
        save_path: str,
        size: Tuple[int, int] = (256, 256)
    ):
        background_color = (127.5, 127.5, 127.5)
        dreambooth_collate = DreamboothCollate(tokenizer)
        transform = ImageTransform(size, background_color)
        dataset = DreamBoothDataset(
            files,  # data_path
            instance_prompt,
            tokenizer,
            transform
        )

        trainer = DreamboothTrainer(trainer_config)
        trainer.config_dataloader(dataset, dreambooth_collate)
        trainer.fit(
            text_encoder,
            vae,
            unet,
            noise_scheduler,
            tokenizer,
            feature_extractor,
            save_path
        )

    @staticmethod
    def post(request):
        tokenizer = CLIPTokenizer.from_pretrained(**models_config["tokenizer"])
        text_encoder = CLIPTextModel.from_pretrained(**models_config["clip"])
        vae = AutoencoderKL.from_pretrained(**models_config["vae"])
        unet = UNet2DConditionModel.from_pretrained(**models_config["unet"])
        noise_scheduler = DDPMScheduler(
            **models_config["noise_scheduler"]
        )
        feature_extractor = CLIPFeatureExtractor.from_pretrained(
            **models_config["feature_extractor"]
        )

        files = request.data.getlist("images")
        asset_type = request.data.get("asset_type")
        save_path = f"./model/{asset_type}"
        extra_information = "from 2D game"
        instance_prompt = f"a picture of {asset_type} {extra_information}"

        images = []
        for file in files:
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes))
            images.append(image)

        TrainModelView.train(
            tokenizer,
            text_encoder,
            vae,
            unet,
            noise_scheduler,
            feature_extractor,
            instance_prompt,
            images,
            save_path
        )

        return Response({
            'status': 'success'
        }, status=201)


class GenereateView(APIView):
    parser_classes = (
        MultiPartParser,
        JSONParser,
    )

    @staticmethod
    def post(request):
        guidance_scale = 7
        num_cols = 25

        asset_type = request.data.get("asset_type")
        save_path = f"./model/{asset_type}"
        pipe = StableDiffusionPipeline.from_pretrained(
            save_path,
            torch_dtype=torch.float16,
            revision="fp16",
        ).to("cuda")

        extra_information = "from 2d game"
        prompt = f"a picture of {asset_type} {extra_information}, cinema4D, hd, front"

        def dummy(images, **kwargs):
            return images, False

        pipe.safety_checker = dummy
        all_images = []
        for _ in range(num_cols):
            images = pipe(prompt, guidance_scale=guidance_scale).images
            all_images.extend(images)
        grid = image_grid(all_images, 5, 5)
        grid.save("123213.jpg")

        return Response({
            'status': 'success'
        }, status=201)

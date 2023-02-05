import io
import yaml

from PIL import Image
from typing import Tuple, Callable, List
from rest_framework.views import APIView
from rest_framework.response import Response
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
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
        feature_extractor: Callable,
        instance_prompt: str,
        files: List[bytes],
        save_path: str,
        size: Tuple[int, int] = (256, 256)
    ):
        dreambooth_collate = DreamboothCollate(tokenizer)
        transform = ImageTransform(size)
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
        feature_extractor = CLIPFeatureExtractor.from_pretrained(
            **models_config["feature_extractor"]
        )

        files = request.data.getlist("images")
        asset_type = request.data.get("asset_type")
        save_path = f"./model/{asset_type}"
        extra_information = "from 2D game, front"
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
            feature_extractor,
            instance_prompt,
            images,
            save_path
        )

        return Response({
            'status': 'success'
        }, status=201)

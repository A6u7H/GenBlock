import io

from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import (
    MultiPartParser,
    JSONParser
)


class UploadView(APIView):
    parser_classes = (
        MultiPartParser,
        JSONParser,
    )

    @staticmethod
    def post(request):
        files = request.data.getlist('images')
        for i, file in enumerate(files):
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes))
            image.save(f"test_{i}.jpg")

        return Response({
            'status': 'success'
        }, status=201)

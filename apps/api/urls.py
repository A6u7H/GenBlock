from django.urls import path
from .views import UploadView

urlpatterns = [
    path('api/v1/upload', UploadView.as_view(), name='prediction'),
]

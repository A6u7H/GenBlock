from django.urls import path
from .views import TrainModelView

urlpatterns = [
    path('api/v1/upload', TrainModelView.as_view(), name='prediction'),
]

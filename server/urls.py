from django.urls import path

from . import routes
urlpatterns = [
    path('classify', routes.index, name='index'),
]
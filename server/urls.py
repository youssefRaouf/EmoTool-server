from django.conf.urls import url

from . import routes
urlpatterns = [
    url('classify', routes.index, name='index'),
]
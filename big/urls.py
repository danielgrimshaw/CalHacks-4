from django.conf.urls import url

from . import views

urlpatterns = [
        url(r'^$', views.index, name='index'),
        url(r'^upload/$', views.upload, name='upload'),
        url(r'^thanks/(?P<datasetId>[0-9]+)/$', views.complete, name='complete'),
]

from django.urls import path
from . import views

urlpatterns = [
    path('', views.main_view, name='main'),
    path('cleaning/', views.cleaning_view, name='cleaning'),
    path('visualization/', views.visualization_view, name='visualization'),
    path('ml_algo/', views.ml_algo, name='ml_algo'),
]

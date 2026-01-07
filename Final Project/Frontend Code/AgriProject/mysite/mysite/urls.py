from django.contrib import admin
from django.urls import path
from predictor import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.dashboard, name='home'),             # Home Page
    path('predict/', views.predict_yield, name='predict_yield'), # Yield Prediction Page
    path('recommend/', views.recommend_crop, name='recommend_crop'), # Crop Recommendation Page
    path('about/', views.about, name='about'),      # About Page
    path('dataset/', views.dataset, name='dataset'),# Dataset Page
    path('download-data/', views.download_dataset, name='download_dataset'), # Download Action
]
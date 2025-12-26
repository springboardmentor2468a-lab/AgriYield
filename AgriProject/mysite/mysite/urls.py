from django.contrib import admin
from django.urls import path
from predictor import views  # Ensure this import is correct

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.predict_yield, name='home'),
    path('about/', views.about, name='about'),
    path('dataset/', views.dataset, name='dataset'),
]
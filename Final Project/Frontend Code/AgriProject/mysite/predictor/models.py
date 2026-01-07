from django.db import models

# Existing Yield Prediction History
class PredictionHistory(models.Model):
    crop = models.CharField(max_length=100)
    N = models.IntegerField()
    P = models.IntegerField()
    K = models.IntegerField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    ph = models.FloatField()
    rainfall = models.FloatField()
    predicted_yield = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.crop} - {self.predicted_yield}"

# --- NEW: Crop Recommendation History (Fix Indentation Here) ---
class CropRecommendationHistory(models.Model):
    N = models.FloatField()
    P = models.FloatField()
    K = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    ph = models.FloatField()
    rainfall = models.FloatField()
    
    # The Output
    recommended_crop = models.CharField(max_length=100)
    confidence_score = models.CharField(max_length=10)
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.recommended_crop} ({self.created_at})"
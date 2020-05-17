from django.db import models

class Learning(models.Model):
    fileName = models.CharField(max_length=100)
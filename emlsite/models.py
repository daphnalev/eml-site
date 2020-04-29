from django.conf import settings
from django.db import models


class File(models.Model):
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    content = models.TextField()

    def publish(self):
        self.save()

    def __str__(self):
        return self.title
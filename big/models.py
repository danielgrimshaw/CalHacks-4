from django.db import models

class Dataset(models.Model):
    email = models.EmailField()

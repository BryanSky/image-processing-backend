from __future__ import unicode_literals

from django.db import models

# Create your models here.


class Image(models.Model):
    imagefile = models.FileField(upload_to='previews')


class Graph(models.Model):
    document = models.ForeignKey(Image, on_delete=models.CASCADE)
    field = models.CharField()

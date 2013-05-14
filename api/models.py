from django.db import models
from django.core.urlresolvers import reverse

class Brand(models.Model) :
  name = models.CharField(max_length=60)
  created = models.DateTimeFiled(auto_now_add=True)

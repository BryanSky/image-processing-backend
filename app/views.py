from django.shortcuts import render
from .models import Image, Graph

# Create your views here.


def index(request):
    return render(request, 'index.html')


def upload_image(request):
    if request.method == 'POST':
        img = Image(request.POST)


def get_graph(request):
    print(request)
    return Graph(request.id)

from django.shortcuts import render


def index(request):
    return render(request, 'index.html')


def learn(request):
    return render(request, 'learn.html')

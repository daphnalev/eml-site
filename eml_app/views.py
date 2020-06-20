from django.shortcuts import render


def index(request):
    return render(request, 'index.html')


def learn(request):
    if request.method == "POST":
        cvs_file = request.FILES["fileToUpload"]
        print(cvs_file.name)
    return render(request, 'learn.html')

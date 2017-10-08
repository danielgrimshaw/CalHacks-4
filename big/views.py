from django.shortcuts import render
from django.http import HttpResponse

from .forms import UploadDataset

def index(request):
    return render(request, 'big/index.html', None)

def upload(request):
    if request.method == "POST":
        form = UploadDataset(request.POST, request.FILES)
        if form.is_valid():
            handleUpload(form.cleaned_data['email'], request.FILES['dataset'])
            HttpResponseRedirect(reverse('complete'))
    else:
        form = UploadDataset()

    return render(request, 'big/upload.html', { 'form': form })

from django.shortcuts import render
from django.http import HttpResponse

from .forms import UploadDataset

def index(request):
    return render(request, 'big/index.html', None)

def upload(request):
    if request.method == "POST":
        form = UploadDataset(request.POST, request.FILES)
        if form.is_valid():
            datasetId = handleUpload(form.cleaned_data['email'], request.FILES['dataset'])
            HttpResponseRedirect(reverse('complete', args=(datasetId,)))
    else:
        form = UploadDataset()

    return render(request, 'big/upload.html', { 'form': form })

def handleUpload(email, dataset):
    pass

def complete(request, datasetId):
    return render(request, 'big/thanks.html', { 'dataset': Dataset.objects.get(pk=datasetId) })

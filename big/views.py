from django.shortcuts import render
from django.http import HttpResponse

from .models import Dataset

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
    # Create Dataset object
    # Enqueue Dataset object somewhere, probably SQS or GCP equiv
    # **OUTSIDE PROCESSING** extract dataset and train model on it
    # **OUTSIDE PROCESSING** create extension of dataset
    # **OUTSIDE PROCESSING** Signal user that data is ready/send user the data
    # Return Dataset id
    pass

def complete(request, datasetId):
    return render(request, 'big/thanks.html', { 'dataset': Dataset.objects.get(pk=datasetId) })

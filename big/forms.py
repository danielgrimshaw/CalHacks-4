from django import forms

class UploadDataset(forms.Form):
    dataset = forms.FileField() # view will need to specify req's
    email = forms.EmailField() # should return additional data points to email

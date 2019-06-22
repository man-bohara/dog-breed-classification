import os
import urllib.request
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image

from app  import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename


ALLOWED_EXENSIONS = set(['txt','pdf','png','jpg','jpeg','gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected for upload')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_filename)
            breed = predict_breed_transfer(full_filename)
            flash('Dog breed is  {}'.format(breed))
            return render_template("upload.html", dog_image='./uploads/'+filename)

            #return redirect('/')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)

def predict_breed_transfer(img_path):
    ## TODO: Specify model architecture
    model_transfer = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Get the input of the last layer of VGG-16
    n_inputs = model_transfer.classifier[6].in_features

    last_layer = nn.Linear(n_inputs, 133)

    model_transfer.classifier[6] = last_layer

    use_cuda = torch.cuda.is_available()

    if (use_cuda):
        model_transfer.load_state_dict(torch.load('model_transfer.pt'))
    else:
        model_transfer.load_state_dict(torch.load('model_transfer.pt', map_location='cpu'))


    # load the image and return the predicted breed

    data_dir = './dog_images/'

    # following is needed to uncomment to test locally.
    # data_dir = './dogImages/'

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform)
                      for x in ['train', 'valid', 'test']}

    # list of class names by index, i.e. a name can be accessed like class_names[0]
    class_names = [item[4:].replace("_", " ") for item in image_datasets['train'].classes]

    img = Image.open(img_path)

    img = transform(img)

    img = img.unsqueeze(0)

    output = model_transfer(img)

    _, preds_tensor = torch.max(output, 1)

    return class_names[preds_tensor.item()]

if __name__ == "__main__":
    app.run()

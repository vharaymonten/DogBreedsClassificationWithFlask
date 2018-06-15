from flask import Flask, request, render_template, url_for,redirect, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename
from src.model_loader import load_model, extract_bottleneck_features_resnet
from src.utils import *
from src.predictors import face_detector, predict_breed
from PIL import Image
import io
import os
import numpy as np
loaded = False
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route("/")
def index():
    return render_template("load_model.html")
@app.route("/dogbreed")
def dogbreed():
    return render_template("cnn.html")

@app.route('/setup', methods=['POST'])
def setup():
    global loaded
    if not loaded:
        global model, graph, extract_bottleneck
        model, graph = load_model()
        extract_bottleneck = extract_bottleneck_features_resnet
        loaded = True
    return redirect("dogbreed")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part

        if 'image' in request.files:
            file = request.files['image']
            filename = file.filename
            if allowed_file(filename):
                image = file.read()
                image = Image.open(io.BytesIO(image))
                print(type(image))
                #For CNN's input
                tensor4d = image_to_tensor(image, expand_dims=True)
                #for face_detecto's input and image_saving
                tensor3d = image_to_tensor(image,resize=False, expand_dims=False)
                post_data = {}
                with graph.as_default():
                    x = extract_bottleneck(tensor4d)
                    prediction = predict_breed(x, model, return_proba=True)

                    dogname, proba = prediction
                faces = face_detector(tensor3d)
                if len(faces) > 0:
                    human = True
                    dog_ears = Image.open('uploads/dog_ears.png')
                    dog_ears = image_to_tensor(dog_ears,resize=False,expand_dims=False)
                    for x,y,w,h in faces:
                        array = wear_dog_ears(dog_ears, tensor3d, x,y,w,h)
                    fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    save_image(fp, array)
                    post_data.update({'message':"HUMAN DETECTED! AS A PUNISHMENT YOU WILL BE PUT DOG EARS"})

                else:
                    human = False
                    fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    save_image(fp, tensor3d)


            report = write_report(fp=fp,dogname=dogname,proba=proba,human=human)
            return render_template('cnn.html',post_data=report)
    else:
        return render_template("error404.html")

    return redirect('dog_breed')

@app.route('/udacity-jupyter-notebook')
def jupyter_notebook():
    return render_template('dog_app.html')
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(debug=True)

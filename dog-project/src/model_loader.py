from keras.models import model_from_json

import numpy as np
import tensorflow as tf


def load_model():
    print("Loading Model")
    with open("src/models/model.json", "r") as json_file:
        json = json_file.read()
    dogresnet50 = model_from_json(json)

    dogresnet50.load_weights('src/models/DogResNet50.hdf5')
    dogresnet50.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
    print("Model is loaded")
    graph = tf.get_default_graph()
    return dogresnet50, graph


def extract_bottleneck_features_resnet(tensor):
    from keras.applications.resnet50 import ResNet50, preprocess_input
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

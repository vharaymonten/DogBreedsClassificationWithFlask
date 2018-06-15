import cv2
import numpy as np
with open('dog_names.txt', 'r') as f:
    dog_names = f.read()
    dog_names = dog_names.split('\n')
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def predict_breed(x, model, return_proba=True):
    assert(len(dog_names) == 133)
    output = model.predict_proba(x)
    prediction = np.argmax(output)

    breed = dog_names[prediction].replace('_', ' ')
    if return_proba:
        proba = output[0][prediction]
        return breed[7:], proba*100
    else:
        return breed

def face_detector(_image):
    _image = _image.astype('uint8')
    gray = cv2.cvtColor(_image,cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return faces

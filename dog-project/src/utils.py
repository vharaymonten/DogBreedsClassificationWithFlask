import time
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from scipy.misc import imsave


def image_to_tensor(_image, resize=True, expand_dims=True):
    if _image.mode != 'RGB':
        _image = _image.convert('RGB')
    if resize:
        _image = _image.resize((224,224))
    x = img_to_array(_image)
    if expand_dims:
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)
    else:
        return x

def wear_dog_ears(ears, face, x,y,w,h):

    face_width = w
    face_height = h

    ears_width = int(face_width*1.25)+1
    ears_height = int(0.15*face_height)+1
    ears = cv2.resize(ears,(ears_width,ears_height))

    for row in range(ears_height):
        for col in range(ears_width):
            for channel in range(3):
                #Prevent white values to overwrite pixels in image
                if ears[row][col][channel] < 235:
                    face[y+row-int(0.2*face_height)][x+col-3][channel] = ears[row][col][channel]
    return face

def save_image(fp, array):
    #Assert if image is not an rgb image
    assert(array.shape[2] == 3)
    imsave(fp, array)


def allowed_file(filename):
    for ext in ['.png', '.jpg', '.jpeg']:
        if filename.endswith(ext):
            return True
    else:
        False

def write_report(**kwargs):
    fp = kwargs['fp']
    proba = str(kwargs['proba'])
    dogname = kwargs['dogname']
    if kwargs['human'] == False:
        message = '''
        BREED                   : {}\n
        PROBABILITY             : {}%\n
        HUMAN                   : {}\n
        '''.format(dogname,proba[:4],"NO")
    else:
        message =  '''
        RESEMBELING BREED       : {}\n
        PROBABILITY             : {}%\n
        HUMAN                   : {}\n
        '''.format(dogname,proba[:4],"YES")

    return {'fp':fp,'messages':message}

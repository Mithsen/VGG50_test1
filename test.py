import os
import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
#from keras.applications.inception_v3 import preprocess_input
from keras.applications.vgg16 import preprocess_input

PATH = os.getcwd()
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile(PATH+"/lbl.txt")]


model=load_model('model.hdf5')

"""Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
"""
img = Image.open("test.jpg")
#img = img.resize(299, 299)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)

size = len(preds[0])
for i in range(size):
	print ('%s (score = %.5f)' % (label_lines[i], preds[0][i]),)






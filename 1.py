import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD 
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

PATH = os.getcwd()
train_path = PATH + '/data/train'
valid_path = PATH + '/data/valid'
test_path = PATH + '/data/test'


data_dir_list = os.listdir(train_path)
f = open('lbl.txt','w')
for dataset in data_dir_list:
	print (format(dataset))
	f.write(str(format(dataset))+'\n')
f.close()


def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt



#Use transfer learning and fine-tuning to train a network on a new dataset
nb_train_samples = get_nb_files(train_path)
print("nb_train_samples ="+str(nb_train_samples))
nb_classes = len(glob.glob(train_path + "/*"))
print("nb_classes ="+str(nb_classes))
nb_val_samples = get_nb_files(valid_path)
nb_epoch = 2
batch_size = 10
IM_WIDTH, IM_HEIGHT = 299, 299
																																				
  # data prep
train_datagen =  ImageDataGenerator(
	rescale=1./255,
	rotation_range=30,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
)
test_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=30,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
  )

train_generator = train_datagen.flow_from_directory(
	train_path,
	target_size=(IM_WIDTH, IM_HEIGHT),
	batch_size=batch_size,
)

validation_generator = test_datagen.flow_from_directory(
	valid_path,
	target_size=(IM_WIDTH, IM_HEIGHT),
	batch_size=batch_size,
)

# setup model
base_model = VGG16(weights='imagenet')
#base_model.summary() #include_top=False excludes final FC layer

"""Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
x = base_model.output
x = GlobalAveragePooling2D()(x)
#x = Dense(1024, activation='relu')(x) #new FC layer, random init
predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
model = Model(input=base_model.input, output=predictions)




# transfer learning
#setup_to_transfer_learn(model, base_model)
"""Freeze all layers and compile the model"""
for layer in base_model.layers:
	layer.trainable = False
#model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


history_tl = model.fit_generator(
	train_generator,
	nb_epoch=nb_epoch,
	samples_per_epoch=nb_train_samples,
	validation_data=validation_generator,
	nb_val_samples=nb_val_samples)

# fine-tuning
#setup_to_finetune(model)

model.save('model.hdf5')


# importing Libaries

import os
import glob
import re
import numpy as np

# import Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from keras.preprocessing import image

# Using Transfer Learning training the model

from keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')
model.save(r'resnet.h5')


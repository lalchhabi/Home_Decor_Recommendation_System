import tensorflow 
import numpy as np
import os
from numpy.linalg import norm
import keras
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50, preprocess_input
from tqdm import tqdm
import pickle


model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    imgs = image.load_img(img_path, target_size = (224,224))
    img_arr = image.img_to_array(imgs)
    expanded_img = np.expand_dims(img_arr, axis = 0)
    preprocess_img = preprocess_input(expanded_img)
    result = model.predict(preprocess_img).flatten()
    normalize_result = result/norm(result)
    
    return normalize_result

img_names = []
for file in os.listdir('images'):
    file_path = os.path.join('images',file )
    img_names.append(file_path)

#### Extract features for all the images

feature_list = []
for img_feature in tqdm(img_names):       ##### Here tqdm is used to visualize the progress of for loop
    feature_list.append(extract_features(img_feature,model))

pickle.dump(img_names,open('img_names.pkl','wb'))
pickle.dump(feature_list,open('features.pkl','wb'))
       

##### Importing Libraries
import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

#### Increasing layout size
st.set_page_config(layout="wide")

#### Building the model
model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#### Loading Pickle File
features_list = np.array(pickle.load(open('features.pkl','rb')))
img_name = np.array(pickle.load(open('img_names.pkl','rb')))

#### App Development

st.markdown("<h1 style='text-align: center; color: grey;'>Home Decor Recommendation System</h1>", unsafe_allow_html=True)

#### save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded_images',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 0
    except:
        return 1

model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#### Function for Feature Extraction
def extract_features(img_path, model):
    imgs = image.load_img(img_path, target_size = (224,224))
    img_arr = image.img_to_array(imgs)
    expanded_img = np.expand_dims(img_arr, axis = 0)
    preprocess_img = preprocess_input(expanded_img)
    result = model.predict(preprocess_img).flatten()
    normalize_result = result/norm(result)
    return normalize_result

###### Function for Recommendation
def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=8, algorithm='brute', metric= 'euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

##### Upload file
uploaded_file = st.file_uploader("Please Upload an Image here")
if uploaded_file is not None:
    save_uploaded_file(uploaded_file)
##### Displaying Image
    display_image = Image.open(uploaded_file)
    st.header("Uploaded Image")
    st.image(display_image)
##### Feature Extraction
    features = extract_features(os.path.join('uploaded_images',uploaded_file.name),model)
    
#### Recommendation
    indices = recommend(features, features_list)

    #### Display Images
    st.header("Here are Recommended Images")
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    
    with col1:
        st.image(img_name[indices[0][0]])
    with col2:
        st.image(img_name[indices[0][1]])
    with col3:
        st.image(img_name[indices[0][2]])
    with col4:
        st.image(img_name[indices[0][3]])
    with col5:
        st.image(img_name[indices[0][4]])
    with col6:
        st.image(img_name[indices[0][5]])   
    with col7:
        st.image(img_name[indices[0][6]]) 
    with col8:
        st.image(img_name[indices[0][7]])


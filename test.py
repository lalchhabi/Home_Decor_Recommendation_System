import pickle
import tensorflow
import keras
import numpy as np
from numpy.linalg import norm
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
##### Load the pickle files
features_list = np.array(pickle.load(open('features.pkl','rb')))
img_name = np.array(pickle.load(open('img_names.pkl','rb')))

#### Load the model
model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#### Image Preprocessing of test image
test_img = image.load_img('test_images/dining_table.jpg', target_size = (224,224))
img_arr = image.img_to_array(test_img)
expanded_img = np.expand_dims(img_arr, axis=0)
preprocessed_img = preprocess_input(expanded_img)
result = model.predict(preprocessed_img).flatten()
normalized_result = result/norm(result)

##### Recommendation
neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric= 'euclidean')
neighbors.fit(features_list)
distances, indices = neighbors.kneighbors([normalized_result])
print(indices)
for file in indices[0]:
    result_img = cv2.imread(img_name[file])
    result_img = cv2.resize(result_img, (500,500))
    cv2.imshow("Recommended Images",result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


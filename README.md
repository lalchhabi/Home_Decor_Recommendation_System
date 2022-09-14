# Home_Decor_Recommendation_System
This is Home Decoration Recommendation System which recommends essentials home decoration tools like Sofa, Bed,  Cupboard, wall art, 
Carpet, Curtain, Candles, Hanging Lamps and many more.

Here the main idea of project is to extract the features of 5773 collected images by using this model which then will formed matrix of dimension 
[5773, 2048] each image has 2048 features. Then similarly a features of new image which we will give input to the system to recommend is also extracted 
and formed matrix [1,2048] and then calculate the distances of two matrix and select the 8 images that are closest to the new image. 
Hence these closest images are display as recommended images

This project has following steps:-

Step 1:
Image Collection

In this project at first, images are collected by scraping it from Selenium and google webdriver. To scrap the images at first we should download updated
version of google chrome and placed it into webdriver directory and set the search key and number of images and then run the main.py file the images will
start downloading.There are almost 5773 images in Image directory.
Files path: https://github.com/lalchhabi/Home_Decor_Recommendation_System/tree/master/Image%20Scraping

Step 2: 
Model Building

Here Residual Network(ResNet50) model is used which is already trained with imageNet dataset and GlobalMaxPooling2D layer is added with this model to extract 
the features of collected 57734 and new image to be recommend and save them into pickle files.
file path : https://github.com/lalchhabi/Home_Decor_Recommendation_System/blob/master/model_building.py

Step3:
Calculating the NearestNeighbour

After extracting the features of both existing and new images, distance is calculated using 
Euclidean and the closest/similar images are displays as a recommended images.
file path : https://github.com/lalchhabi/Home_Decor_Recommendation_System/blob/master/test.py

Step4:
Displaying the recommended images

file path : https://github.com/lalchhabi/Home_Decor_Recommendation_System/blob/master/app.py

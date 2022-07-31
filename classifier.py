#use computers camera
import cv2
#mathematical operations
import numpy as np
#treat our data as dataframes
import pandas as pd
#split the data into training and testing the data
from sklearn.model_selection import train_test_split
#Lr classifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps

X,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes =['0','1','2','3','4','5','6','7','8','9']
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

X_train_scaled = X_train/255
X_test_scaled = X_test/255

clf = LogisticRegression(solver='saga', multi_class= 'multinomial').fit(X_train_scaled, y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw_resize((28, 28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resize_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resize_inverted_scaled = np.asarray(image_bw_resize_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resize_inverted_scaled).reshape(1, 782)
    test_pred = clf.predict(test_sample)
    return test_pred(0)
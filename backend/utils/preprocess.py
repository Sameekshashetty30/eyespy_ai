import cv2
import numpy as np

def preprocess_image(img, target_size=(224,224)):
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.preprocessing.image import img_to_array, load_img
import keras
import numpy as np
import time


class Classifier:
    def __init__(self, weights_path):
        self.vgg16 = applications.VGG16(include_top=False, weights='imagenet')
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(8, 8, 512)))  
        self.model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3)))  
        self.model.add(Dropout(0.5))  
        self.model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3)))  
        self.model.add(Dropout(0.3)) 
        self.model.add(Dense(13, activation='softmax'))
        self.model.load_weights(weights_path)
        
    def read_image(self, path):
        image = load_img(path, target_size=(256, 256))  
        image = img_to_array(image)  
        image = np.expand_dims(image, axis=0)
        image /= 255.  
        return image
        
    def probability(self, path, label):
        animals = ["Bird", "Cat", "Dog", "Fish", "Insect", "Mammal A", "Mammal B", "Mammal C", "Mammal D", "Monkey","Reptiles", "Sea Creature A", "Sea Creature B"]
        idx = animals.index(label)
        images = self.read_image(path)
        time.sleep(.5)
        bt_prediction = self.vgg16.predict(images)
        preds = self.model.predict(bt_prediction)
        pred_class = preds[0][idx]
        print(f'{label} for probability of {pred_class}') 
        return pred_class

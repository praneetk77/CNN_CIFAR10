from random import randint

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def show_image(x,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])
    plt.show()

def show_image(x,y,index, pred):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])
    plt.title("Prediction is : " + pred)
    plt.show()

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
(x_train, y_train),(x_test,y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)
x_train = x_train/255
x_test = x_test/255

# model = keras.models.Sequential([
#     #cnn
#     keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(32,32,3)),
#     keras.layers.MaxPooling2D((2,2)),
#     keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu"),
#     keras.layers.MaxPooling2D((2,2)),
#     #dense
#     keras.layers.Flatten(input_shape=(32,32,3)),
#     keras.layers.Dense(64, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
# ])
#
# model.compile(optimizer="SGD", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#
# model.fit(x_train,y_train,epochs=10)
#
# model.save("cnn.model")

model = keras.models.load_model("cnn.model")
print("Previously saved model loaded.")

loss, accuracy = model.evaluate(x_test,y_test)
print("Loss : ",loss," Accuracy : ",accuracy)
predictions = model.predict(x_test)

for i in range(5):
    n = randint(0,300)
    show_image(x_test,y_test,n,classes[np.argmax(predictions[n])])


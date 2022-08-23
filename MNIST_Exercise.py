from random import randint
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

print(train_images[0].shape)

train_images = train_images.reshape((60000,28,28,1))
test_images = test_images.reshape((10000,28,28,1))

model = keras.Sequential([
    # cnn
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    # dense
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=8)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested acc : ", test_acc)

prediction = model.predict(test_images)

for i in range(5):
    n = randint(0, 300)
    plt.imshow(test_images[n], cmap=plt.cm.binary)
    plt.xlabel("actual : " + str(test_labels[n]))
    plt.title("prediction : " + str(np.argmax(prediction[n])))
    plt.show()

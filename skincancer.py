import tensorflow as tf
# from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
# from pylab import subplot, imshow, show
import os


train = ImageDataGenerator()
test = ImageDataGenerator()
val = ImageDataGenerator()

train = 'CNN_model_dataset/data/train/'
train_data = tf.keras.utils.image_dataset_from_directory(
    train,
    validation_split = 0.25,
    image_size = (224, 224),
    seed = 50,
    batch_size = 35,
    subset = 'training',
)

val = 'CNN_model_dataset/data/train/'
val_data = tf.keras.utils.image_dataset_from_directory(
    val,
    validation_split = 0.25,
    image_size = (224, 224),
    seed = 50,
    batch_size = 35,
    subset = 'validation',
)

test = 'CNN_model_dataset/data/test/'
test_data = tf.keras.utils.image_dataset_from_directory(
    test,
    image_size = (224, 224),
    seed = 50,
    batch_size = 35,
)


# display sample of images and classification
class_names = ['Benign', 'Malignant']
train_data.class_names = class_names
val_data.class_names = class_names
plt.figure(figsize = (8, 8))
for images, labels in test_data.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(test_data.class_names[labels[i]])
        plt.axis("off")
plt.show()


# train model
model = Sequential()
model.add(Conv2D(
            8, # filters
            11, # kernel
            strides = 3, # step size
            activation = 'relu',
            input_shape = (224, 224, 3),
        )) # Output: 72 x 72 x 8 
model.add(MaxPool2D(pool_size = 2))
# Output: 36 x 36 x 8
model.add(Dropout(0.2))
model.add(Conv2D(
    8, # filters
    3, # kernel
    strides = 1,
    activation = 'relu',
)) # Output: 34 x 34 x 8
model.add(MaxPool2D(pool_size = 2))
# Output: 17 x 17 x 8
model.add(Flatten())
# Ouput length: 2312
model.add(Dense(
    1024, activation = 'relu',
))
model.add(Dense(
    512, activation = 'relu',
))
model.add(Dense(
    256, activation = 'relu'
))
model.add(Dense(
    64, activation = 'relu',
))
model.add(Dense(
    32, activation = 'relu',
))
model.add(Dense(
    2, # exactly equal to # of classes
    activation = 'softmax', # always use softmax on last layer
))
model.summary()


model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

model.fit(
    train_data,
    epochs = 10,
    validation_data = val_data,
)

model.save("cnn_model_skin")

print("\nTest data accuracy:")
model.evaluate(test_data)

# individually predict some images
print("\n\nIndividual Predictions:")
class_names = {0: "Benign", 1: "Malignant"}
for images, labels in test_data.take(1):
    for i in range(6):
        print()
        print("Actual:", test_data.class_names[labels[i]])
        x = image.img_to_array(images[i])
        x = np.expand_dims(x, axis = 0)
        p = np.argmax(model.predict(x))
        if p == 0:
            print("Predicted: Benign")
        else:
            print("Predicted: Malignant")
        print(" ")

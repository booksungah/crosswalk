import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

paths = glob.glob('./notMNIST_small/*/*.png')
print(paths)
paths = np.random.permutation(paths)
독립 = np.array([plt.imread(paths[i])for i in range(len(paths))])
종속 = np.array([paths[i].split('/')[2] for i in range(len(paths))])
print(독립.shape, 종속.shape)

import tensorflow as tf
import keras 
X = tf.keras.layers.Input(shape=[ 1 ,25,25])

H = tf. keras.layers.Conv2D(6, kernel_size=5, padding='same', activation = 'swish')(X)
H = tf.keras.layers.MaxPool2D()(H)

H = tf. keras.layers.Flatten()(H)
H = tf. keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation = 'swish')(H)
Y = tf.keras.layers.Dense(10, activation ='softmax')(H)

model = tf.keras.models.Model(X,Y)
model.compile(loss = 'categorical_crossentropy', metrics=['accuracy'])


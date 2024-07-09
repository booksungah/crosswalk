import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#glob - 파일명을 list형식으로 변환
paths = glob.glob('./notMNIST_small/*/*.png')
print(paths)
paths = np.random.permutation(paths)
# 파일 랜덤한 순서로 변경
독립 = np.array([plt.imread(paths[i])for i in range(len(paths))])
# array - 배열, plt.imread = paths값 읽기, for i in range = 모든 파일에 대해 plt.imread 실행
종속 = np.array([paths[i].split('/')[2] for i in range(len(paths))])
# numpy 식으로 배열, 
print(독립.shape, 종속.shape)

import tensorflow as tf
import keras 
X = tf.keras.layers.Input(shape=[ 1 ,25,25])
# 파일 25 * 25 파일을 불러옴

H = tf. keras.layers.Conv2D(6, kernel_size=5, padding='same', activation = 'swish')(X)
H = tf.keras.layers.MaxPool2D()(H)

H = tf. keras.layers.Flatten()(H)
H = tf. keras.layers.Dense(120, activation='swish')(H)
#처음 기본값
H = tf.keras.layers.Dense(84, activation = 'swish')(H)
#hidden layer
Y = tf.keras.layers.Dense(10, activation ='softmax')(H)
#마지막 결과값

model = tf.keras.models.Model(X,Y)
model.compile(loss = 'categorical_crossentropy', metrics=['accuracy'])


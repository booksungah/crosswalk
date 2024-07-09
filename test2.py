import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# 파일명을 리스트 형식으로 변환
paths = glob.glob('./notMNIST_small/*/*.png')
print(paths)
paths = np.random.permutation(paths)  # 파일 랜덤한 순서로 변경

# 이미지를 (25, 25) 크기로 리사이즈하고 배열로 변환
독립 = np.array([cv2.resize(plt.imread(path), (25, 25)) for path in paths])
# 종속 변수 생성
종속 = np.array([path.split('/')[2] for path in paths])

print(독립.shape, 종속.shape)

# 종속 변수 One-hot 인코딩
종속 = pd.get_dummies(종속).values

# 모델 구축
X = tf.keras.layers.Input(shape=[25, 25, 1])  # 입력 크기 수정

H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# 데이터셋을 훈련과 테스트로 분리
from sklearn.model_selection import train_test_split
독립_train, 독립_test, 종속_train, 종속_test = train_test_split(독립, 종속, test_size=0.2, random_state=42)

# 모델 훈련
model.fit(독립_train, 종속_train, epochs=10, batch_size=32, validation_data=(독립_test, 종속_test))

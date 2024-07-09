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
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

print("Ключи датасета:", iris.keys())
print("\nОписание датасета:\n", iris['DESCR'])
print("\nНазвания признаков:", iris.feature_names)
print("Названия классов:", iris.target_names)
print("Форма данных:", iris.data.shape)
print("Количество примеров каждого класса:", np.bincount(iris.target))
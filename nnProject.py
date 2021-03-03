import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

# Створюємо seed для повторюваності результатів
numpy.random.seed(42)
# Загружаємо данні і розділяємо їх на відповідні набори
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# Нормалізуємо данні
X_train = X_train.astype("float32")
Y_train = Y_train.astype("float32")
X_test = X_train.astype("float32")
Y_test = Y_test.astype("float32")
X_train /= 255
X_test /= 255
Y_test /= 255

# Змінюємо мітки класів в категорії
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)
# Створюємо послідовну модель

model = Sequential()
# перший згортковий шар

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))

# другий згортковий шар
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

# перший шар підвибірки
model.add(MaxPooling2D(pool_size=(2, 2)))

# Шар регуляризації
model.add(Dropout(0.25))

# Третій згортковий шар
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

# четвертий згортковий шар
model.add(Conv2D(64, (3, 3), activation='relu'))

# Другий шар підвибірки
model.add(MaxPooling2D(pool_size=(2, 2)))

# Шар регуляризації
model.add(Dropout(0.25))
model.add(Flatten())

# Шар для класифікації
model.add(Dense(512, activation='relu'))

# Шар регуляризації
model.add(Dropout(0.5))

# Вихідний повнозв'язний шар
model.add(Dense(10, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
model.fit(X_train, Y_train, batch_size=32, epochs=30, validation_split=0.1, shuffle=True, verbose=2)

model_json = model.to_json()

json_file = open("cifar_model.json", "w")

json_file.write(model_json)

json_file.close()

model.save_weights("cifar_model.h5")

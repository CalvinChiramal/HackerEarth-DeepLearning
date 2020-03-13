import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train = pd.read_csv("dataset/train.csv")
y = train.iloc[:,1].values
y = LabelEncoder().fit_transform(y)

print(y)


X = []
for i in range(len(train)):
    image = cv2.imread("dataset/Train Images/" + train.Image[i])
    resize = cv2.resize(image, (100,100), cv2.INTER_AREA)
    imagegray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    X.append(imagegray)

X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train.shape)
X_train = X_train.reshape((len(X_train),100,100,1))
X_test = X_test.reshape((len(X_test),100,100,1))

model = Sequential()

model.add(Conv2D(32, (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(rate = 0.5))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dense(4, activation = "softmax"))

callbacks = [
    EarlyStopping(patience = 10, verbose = 1),
    ReduceLROnPlateau(factor = 0.1, patience = 3,
    min_lr = 0.00001, verbose = 1),
    ModelCheckpoint('model.h5',verbose = 1, save_best_only = True,
    save_weights_only = True)
]

model.compile(optimizer = "Adam", metrics = ['accuracy'], loss = 'sparse_categorical_crossentropy')

model.fit(X_train, y_train, epochs = 30, validation_data = (X_test, y_test))

model.save("model.h5")
y_pred = model.predict(X_test)


print(accuracy_score(y_test, y_pred))
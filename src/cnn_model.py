import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import MaxPool2D

from src.utils.utils import load_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCH = 12


def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model_2():
    model = Sequential()

    model.add(Conv2D(4, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(8, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(16, (3, 3), padding='same', activation=relu))
    # model.add(MaxPool2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation=tanh))
    model.add(Dense(3, activation=sigmoid))
    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = load_dataset()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # model = create_cnn_model()
    model = create_cnn_model_2()

    true_values = np.argmax(y_train, axis=1)
    preds = np.argmax(model.predict(x_train), axis=1)

    print("Confusion Train Matrix Before Training")
    print(confusion_matrix(true_values, preds))

    true_values = np.argmax(y_test, axis=1)
    preds = np.argmax(model.predict(x_test), axis=1)
    print("Confusion Test Matrix Before Training")
    print(confusion_matrix(true_values, preds))

    print(f'Train Acc : {model.evaluate(x_train, y_train)[1]}')
    print(f'Test Acc : {model.evaluate(x_test, y_test)[1]}')

    logs = model.fit(x_train, y_train, batch_size=16, epochs=EPOCH, verbose=1, validation_data=(x_test, y_test),
                     callbacks=[TensorBoard()], use_multiprocessing=True)

    true_values = np.argmax(y_train, axis=1)
    preds = np.argmax(model.predict(x_train), axis=1)

    print("Confusion Train Matrix After Training")
    print(confusion_matrix(true_values, preds))

    true_values = np.argmax(y_test, axis=1)
    preds = np.argmax(model.predict(x_test), axis=1)
    print("Confusion Test Matrix After Training")
    print(confusion_matrix(true_values, preds))
    print(f'Train Acc : {model.evaluate(x_train, y_train)[1]}')
    print(f'Test Acc : {model.evaluate(x_test, y_test)[1]}')

    print(logs.history.keys())
    plt.plot(logs.history['accuracy'])
    plt.plot(logs.history['val_accuracy'])
    plt.show()

    plt.plot(logs.history['loss'])
    plt.plot(logs.history['val_loss'])
    plt.show()

    model.save('cnn_model.keras')
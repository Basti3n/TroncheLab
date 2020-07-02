import numpy as np
import os

from tensorflow.keras.callbacks import TensorBoard

from src.utils.utils import load_dataset

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

EPOCH = 300
BATCH_SIZE = 64
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_linear_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(3, activation=sigmoid))
    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_dataset()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    model = create_linear_model()

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

    logs = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, validation_data=(x_test, y_test),
                     callbacks=[TensorBoard()])

    true_values = np.argmax(y_train, axis=1)
    preds = np.argmax(model.predict(x_train), axis=1)

    print("Confusion Train Matrix After Training")
    print(confusion_matrix(true_values, preds))

    true_values = np.argmax(y_test, axis=1)
    preds = np.argmax(model.predict(x_test), axis=1)
    print("Confusion Test Matrix After Training")
    print(confusion_matrix(true_values, preds))
    train_accuracy = model.evaluate(x_train, y_train)[1]
    print(f'Train Acc : {train_accuracy}')
    print(f'Test Acc : {model.evaluate(x_test, y_test)[1]}')

    print(logs.history.keys())
    plt.plot(logs.history['accuracy'])
    plt.plot(logs.history['val_accuracy'])
    plt.show()

    plt.plot(logs.history['loss'])
    plt.plot(logs.history['val_loss'])
    plt.show()

    model.save(f'../models/linear_model_e{EPOCH}_b{BATCH_SIZE}_a{train_accuracy}.keras')
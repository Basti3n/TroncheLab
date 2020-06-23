import os

import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.activations import relu

from src.utils.utils import load_dataset, TARGET_RESOLUTION

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
EPOCH = 300

def create_dense_res_nn_model():
    input_tensor = keras.layers.Input((TARGET_RESOLUTION[0], TARGET_RESOLUTION[1], 3))

    previous_tensor = Flatten()(input_tensor)

    next_tensor = Dense(64, activation=relu)(previous_tensor)

    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])

    next_tensor = Dense(64, activation=relu)(previous_tensor)

    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])

    next_tensor = Dense(64, activation=relu)(previous_tensor)

    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])

    next_tensor = Dense(64, activation=tanh)(previous_tensor)
    next_tensor = Dense(3, activation=sigmoid)(next_tensor)

    model = keras.models.Model(input_tensor, next_tensor)

    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_dataset()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    model = create_dense_res_nn_model()

    logs = model.fit(x_train, y_train, epochs=EPOCH, batch_size=64, verbose=1, validation_data=(x_test, y_test),
                     callbacks=[TensorBoard()])

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

    model.save('resnet_model.keras')
    # plot_model(model, to_file='ResNet.png')
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))

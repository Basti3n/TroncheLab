import numpy as np
import os

from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from src.utils.utils import load_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCH = 12


def create_resnet_model():
    # model = ResNet50(weights='imagenet')
    model = ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    out = Dense(5, activation='softmax')(x)
    model_final = Model(model.input, out)
    model_final.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model_final


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = load_dataset()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    model = create_resnet_model()

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

    model.save('cnn_model.keras')
import numpy as np
import os
import logging
from dataclasses import dataclass


from tensorflow.keras.callbacks import TensorBoard
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)


@dataclass
class EssaiProcess:

    @classmethod
    def run(cls) -> None:
        logger.info("Essai process started")

        # cls.test()

        (x_train, y_train), (x_test, y_test) = cls.load_dataset()
        logger.debug(x_train.shape)
        logger.debug(y_train.shape)
        logger.debug(x_test.shape)
        logger.debug(y_test.shape)

        model = cls.create_linear_model()
        # model = create_mlp_model()

        true_values = np.argmax(y_train, axis=1)
        preds = np.argmax(model.predict(x_train), axis=1)

        logger.debug("Confusion Train Matrix Before Training")
        logger.debug(confusion_matrix(true_values, preds))

        true_values = np.argmax(y_test, axis=1)
        preds = np.argmax(model.predict(x_test), axis=1)
        logger.debug("Confusion Test Matrix Before Training")
        logger.debug(confusion_matrix(true_values, preds))

        logger.debug(f'Train Acc : {model.evaluate(x_train, y_train)[1]}')
        logger.debug(f'Test Acc : {model.evaluate(x_test, y_test)[1]}')

        logs = model.fit(x_train, y_train, batch_size=16, epochs=5000, verbose=0, validation_data=(x_test, y_test),
                         callbacks=[TensorBoard()])

        true_values = np.argmax(y_train, axis=1)
        preds = np.argmax(model.predict(x_train), axis=1)

        logger.debug("Confusion Train Matrix After Training")
        logger.debug(confusion_matrix(true_values, preds))

        true_values = np.argmax(y_test, axis=1)
        preds = np.argmax(model.predict(x_test), axis=1)
        logger.debug("Confusion Test Matrix After Training")
        logger.debug(confusion_matrix(true_values, preds))
        logger.debug(f'Train Acc : {model.evaluate(x_train, y_train)[1]}')
        logger.debug(f'Test Acc : {model.evaluate(x_test, y_test)[1]}')

        logger.debug(logs.history.keys())
        plt.plot(logs.history['accuracy'])
        plt.plot(logs.history['val_accuracy'])
        plt.show()

        plt.plot(logs.history['loss'])
        plt.plot(logs.history['val_loss'])
        plt.show()

        model.save('mlp.keras')

        logger.info("Essai process ended")

    @classmethod
    def test(cls):
        for file in os.listdir("../../../Dataset/Train/ClassA/"):
            logger.info(file)

    @classmethod
    def load_dataset(cls):
        Ximgs = []
        y_train = []

        target_resolution = (64, 64)
        for file in os.listdir("../../../Dataset/Train/ClassA/"):
            Ximgs.append(
                np.array(Image.open(f"../../../Dataset/Train/ClassA/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_train.append([1, 0, 0])
        for file in os.listdir("../../../Dataset/Train/ClassB/"):
            Ximgs.append(
                np.array(Image.open(f"../../../Dataset/Train/ClassB/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_train.append([0, 1, 0])
        for file in os.listdir("../../../Dataset/Train/ClassC/"):
            Ximgs.append(
                np.array(Image.open(f"../../../Dataset/Train/ClassC/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_train.append([0, 0, 1])
        Ximgs_test = []
        y_test = []
        for file in os.listdir("../../../Dataset/Test/ClassA/"):
            Ximgs_test.append(
                np.array(Image.open(f"../../../Dataset/Test/ClassA/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_test.append([1, 0, 0])
        for file in os.listdir("../../../Dataset/Test/ClassB/"):
            Ximgs_test.append(
                np.array(Image.open(f"../../../Dataset/Test/ClassB/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_test.append([0, 1, 0])
        for file in os.listdir("../../../Dataset/Test/ClassC/"):
            Ximgs_test.append(
                np.array(Image.open(f"../../../Dataset/Test/ClassC/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_test.append([0, 0, 1])
        x_train = np.array(Ximgs)
        y_train = np.array(y_train)
        x_test = np.array(Ximgs_test)
        y_test = np.array(y_test)
        return (x_train, y_train), (x_test, y_test)

    @classmethod
    def create_linear_model(cls):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(3, activation=sigmoid))
        model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
        return model

    @classmethod
    def create_mlp_model(cls):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(256, activation=tanh))
        model.add(Dense(256, activation=tanh))
        model.add(Dense(3, activation=sigmoid))
        model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
        return model





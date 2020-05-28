import os
from enum import Enum

import numpy as np
from PIL import Image
from keras_preprocessing import image
from tensorflow.python.keras.models import load_model

DATASET_PATH = os.environ['DATASET_PATH']
TARGET_RESOLUTION = (64, 64)


class Classes(Enum):
    CARNIVAL = 0
    FACE = 1
    MASK = 2


def load_dataset():
    Ximgs = []
    y_train = []

    for file in os.listdir(f'{DATASET_PATH}/Train/Carnaval/'):
        Ximgs.append(
            np.array(
                Image.open(f'{DATASET_PATH}/Train/Carnaval/{file}').resize(TARGET_RESOLUTION).convert('RGB')) / 255.0)
        y_train.append([1, 0, 0])
    for file in os.listdir(f'{DATASET_PATH}/Train/Face/'):
        Ximgs.append(
            np.array(Image.open(f'{DATASET_PATH}/Train/Face/{file}').resize(TARGET_RESOLUTION).convert('RGB')) / 255.0)
        y_train.append([0, 1, 0])
    for file in os.listdir(f'{DATASET_PATH}/Train/Mask/'):
        Ximgs.append(
            np.array(Image.open(f'{DATASET_PATH}/Train/Mask/{file}').resize(TARGET_RESOLUTION).convert('RGB')) / 255.0)
        y_train.append([0, 0, 1])
    Ximgs_test = []
    y_test = []
    for file in os.listdir(f'{DATASET_PATH}/Test/Carnaval/'):
        Ximgs_test.append(
            np.array(
                Image.open(f'{DATASET_PATH}./Test/Carnaval/{file}').resize(TARGET_RESOLUTION).convert('RGB')) / 255.0)
        y_test.append([1, 0, 0])
    for file in os.listdir(f'{DATASET_PATH}/Test/Face/'):
        Ximgs_test.append(
            np.array(Image.open(f'{DATASET_PATH}/Test/Face/{file}').resize(TARGET_RESOLUTION).convert('RGB')) / 255.0)
        y_test.append([0, 1, 0])
    for file in os.listdir(f'{DATASET_PATH}/Test/Mask/'):
        Ximgs_test.append(
            np.array(Image.open(f'{DATASET_PATH}/Test/Mask/{file}').resize(TARGET_RESOLUTION).convert('RGB')) / 255.0)
        y_test.append([0, 0, 1])
    x_train = np.array(Ximgs)
    y_train = np.array(y_train)
    x_test = np.array(Ximgs_test)
    y_test = np.array(y_test)
    return (x_train, y_train), (x_test, y_test)


def load_linear_model(file: str):
    model = load_model(f'./src/linear_model.keras')
    model.summary()
    img = image.load_img(f'{DATASET_PATH}./Autre/{file}', target_size=TARGET_RESOLUTION)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    print(f'Test Acc : {Classes(model.predict_classes(images, batch_size=10))}')
    # print(f'Test Acc : {model.predict_classes(images, batch_size=10)}')


def load_mlp_model(file: str):
    model = load_model(f'./src/mlp_model.keras')
    model.summary()
    img = image.load_img(f'{DATASET_PATH}./Autre/{file}', target_size=TARGET_RESOLUTION)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    print(f'Test Acc : {Classes(model.predict_classes(images, batch_size=10))}')
    # print(f'Test Acc : {model.predict_classes(images, batch_size=10)}')


def load_cnn_model(file: str):
    model = load_model(f'./src/cnn_model.keras')
    model.summary()
    img = image.load_img(f'{DATASET_PATH}./Autre/{file}', target_size=TARGET_RESOLUTION)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    print(f'Test Acc : {Classes(model.predict_classes(images, batch_size=10))}')
    # print(f'Test Acc : {model.predict_classes(images, batch_size=10)}')
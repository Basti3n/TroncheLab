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


# def identity_block(X, f, filters, stage, block):
#     # Defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     # Retrieve Filters
#     F1, F2, F3 = filters
#
#     # Save the input value
#     X_shortcut = X
#
#     # First component of main path
#     X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
#                kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
#     X = Activation('relu')(X)
#
#     # Second component of main path
#     X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
#                kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
#     X = Activation('relu')(X)
#
#     # Third component of main path
#     X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
#                kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
#
#     # Final step: Add shortcut value to main path, and pass it through a RELU activation
#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)
#
#     return X
#
# def convolutional_block(X, f, filters, stage, block, s=2):
#
#     # Defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     # Retrieve Filters
#     F1, F2, F3 = filters
#
#     # Save the input value
#     X_shortcut = X
#
#     ##### MAIN PATH #####
#     # First component of main path
#     X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
#     X = Activation('relu')(X)
#
#     # Second component of main path
#     X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
#     X = Activation('relu')(X)
#
#     # Third component of main path
#     X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
#
#     ##### SHORTCUT PATH ####
#     X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
#     X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
#
#     # Final step: Add shortcut value to main path, and pass it through a RELU activation
#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)
#
#     return X
#
#
# def ResNet50(input_shape=(64, 64, 3), classes=6):
#     """
#     CONV2D -> BATCHNORM -> RELU -> MAXPOOL
#     -> (CONVBLOCK, IDBLOCK*2) -> (CONVBLOCK, IDBLOCK*3) -> (CONVBLOCK, IDBLOCK*5)
#     -> (CONVBLOCK, IDBLOCK*2) -> AVGPOOL -> TOPLAYER
#
#     Arguments:
#     input_shape -- shape of the images of the dataset
#     classes -- integer, 라벨 수
#
#     Returns: model -- a Model() instance in Keras
#     """
#
#     # Define the input as a tensor with shape input_shape
#     X_input = Input(input_shape)
#     # Zero-Padding
#     X = ZeroPadding2D((3, 3))(X_input)
#
#     # Stage 1
#     X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1',
#                kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name='bn_conv1')(X)
#     X = Activation('relu')(X)
#     X = MaxPooling2D((3, 3), strides=(2, 2))(X)
#
#     # Stage 2
#     X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
#     X = identity_block(X, 3, filters=[64, 64, 256], stage=2, block='b')
#     X = identity_block(X, 3, filters=[64, 64, 256], stage=2, block='c')
#
#     # Stage 3
#     X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
#     X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block='b')
#     X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block='c')
#     X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block='d')
#
#     # Stage 4
#     X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
#     X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='b')
#     X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='c')
#     X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='d')
#     X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='e')
#     X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='f')
#
#     # Stage 5
#     X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
#     X = identity_block(X, 3, filters=[512, 512, 2048], stage=5, block='b')
#     X = identity_block(X, 3, filters=[512, 512, 2048], stage=5, block='c')
#
#     # AVGPOOL: don't use padding in pooling layer
#     X = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', name='avg_pool')(X)
#
#     # output layer
#     X = Flatten()(X)
#     X = Dense(classes, activation='softmax',
#               name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
#
#     # Create model
#     model = Model(inputs=X_input, outputs=X, name='ResNet50')
#     return model
#
#
# def create_resnet_model():
#     model = ResNet50(input_shape=(64, 64, 3), classes=3)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

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

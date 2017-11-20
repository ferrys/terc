# -*- coding: utf-8 -*-

import dataset
import numpy as np

from keras.preprocessing import image
from keras.applications import resnet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

from sklearn.metrics import log_loss
from sklearn.metrics import fbeta_score

from keras import backend as K


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


def resnet50_model(img_rows, img_cols, color_type=1, num_classes=None):
    """
    Resnet 50 Model for Keras

    Model Schema is based on
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

    ImageNet Pretrained Weights
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """

    input = Input(shape=(img_rows, img_cols, color_type), name='image_input')


    model = resnet50.ResNet50(weights='imagenet', include_top=True)

    x = Dense(num_classes, activation='sigmoid', name='predictions')(model.layers[-2].output)

    model = Model(input=model.input, output=x)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy', f1])
  
    return model

if __name__ == '__main__':

    train_path = 'Terc_Images\\processed_images'

    validation_size = 0.2
    test_size = 0.2
    img_rows, img_cols, img_size = 224, 224, 224


    channel = 3
    batch_size = 16
    nb_epoch = 10

    # Prepare input data
    classes = ['Volcano', 'Sunrise Sunset', 'ISS Structure', 'Stars', 'Night', 'Aurora', 'Movie', 'Day', 'Moon',
               'Inside ISS', 'Dock Undock', 'Cupola']
    num_classes = len(classes)


    X_train, y_train, X_valid, y_valid = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size, test_size=test_size)


    # Load our model
    model = resnet50_model(img_rows, img_cols, channel, num_classes)

    # Start Fine-tuning
    model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, y_valid),
              )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    predictions = [[1 if predictions_valid[i][j] > 0.5 else 0 for j in range(predictions_valid.shape[1])] for i in range(predictions_valid.shape[0])]

    predictions = np.asarray(predictions)
    np.savetxt('predictions.csv', predictions, delimiter=",")


    # Cross-entropy loss score
    score = log_loss(y_valid, predictions_valid)


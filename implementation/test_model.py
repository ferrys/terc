# -*- coding: utf-8 -*-

import dataset
import numpy as np

from keras.optimizers import SGD
from keras import backend as K

from sklearn.metrics import log_loss
from sklearn.metrics import fbeta_score

from keras import backend as K

from keras.models import model_from_json


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




# load test data
train_path = 'Terc_Images\\processed_images'
validation_size = 0.2
test_size = 0.2
batch_size = 10
img_rows, img_cols, img_size = 224, 224, 224

# Prepare input data
classes = ['Volcano', 'Sunrise Sunset', 'ISS Structure', 'Stars', 'Night', 'Aurora', 'Movie', 'Day', 'Moon',
           'Inside ISS', 'Dock Undock', 'Cupola']

# We shall load all the training and validation images and labels into memory using openCV and use that during training
_, _, _, _, X_test, y_test = dataset.read_train_sets(train_path, img_size, classes,
                                                                   validation_size=validation_size, test_size=test_size)


# load model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()


loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
# loaded_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy', f1, recall, precision])
predictions_test = loaded_model.predict(X_test, batch_size=batch_size, verbose=1)
predictions = [[1 if predictions_test[i][j] > 0.5 else 0 for j in range(predictions_test.shape[1])] for i in range(predictions_test.shape[0])]

predictions = np.asarray(predictions)
np.savetxt('predictions_test.csv', predictions, delimiter=",")




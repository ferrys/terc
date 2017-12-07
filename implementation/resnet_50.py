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

import pandas as pd
from matplotlib import pyplot as plt
import accuracy
import predictions



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
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy', f1, recall, precision])
  
    return model

if __name__ == '__main__':
    # Example to fine-tune on 3000 samples from Cifar10
    lr = 1e-2
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


    # We shall load all the training and validation images and labels into memory using openCV and use that during training
    X_train, y_train, X_valid, y_valid, _, _ = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size, test_size=test_size)


    # Load our model
    model = resnet50_model(img_rows, img_cols, channel, num_classes)

    # Start Fine-tuning
    hist = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, y_valid),
              )

    # export model and weights
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')
    print("Saved model to disk!")

    # get values from training
    train_loss = hist.history['loss']
    train_f1 = hist.history['f1']
    train_acc = hist.history['acc']
    train_precision = hist.history['precision']
    train_recall = hist.history['recall']

    val_loss = hist.history['val_loss']
    val_acc = hist.history['val_acc']
    val_precision = hist.history['val_precision']
    val_f1 = hist.history['val_f1']
    val_recall =hist.history['val_recall']

    # export to csv
    train_df = pd.DataFrame({'acc':train_acc, 'precision': train_precision, 'recall': train_recall, 'f1':train_f1, 'loss':train_loss})
    print(train_df)
    train_df.to_csv("training_output_"+str(lr)+".csv")

    val_df = pd.DataFrame({'val_acc':val_acc, 'val_precision': val_precision, 'val_recall': val_recall, 'val_f1':val_f1, 'val_loss':val_loss})
    print(val_df)
    val_df.to_csv("validation_output_"+str(lr)+".csv")


    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    predictions = [[1 if predictions_valid[i][j] > 0.5 else 0 for j in range(predictions_valid.shape[1])] for i in range(predictions_valid.shape[0])]

    validation_images = 'validation_data.csv'
    ids = predictions.get_ids(validation_images)

    predictions.save_predictions(ids, predictions)


    #Display accuracy
    print("Accuracy for {}".format(lr))
    valid_category_accuracy = accuracy.get_category_accuracy('Terc_Images\\processed_images\\validation_data.csv','predictions_' + str(lr) + '.csv')
    valid_overall_accuracy = accuracy.get_overall_accuracy('Terc_Images\\processed_images\\validation_data.csv','predictions_' + str(lr) + '.csv')
    print(valid_overall_accuracy)
    
    # Display cross-entropy loss 
    print("Cross-entropy loss for {}".format(lr))
    loss_score = log_loss(y_valid, predictions_valid)
    print(loss_score)

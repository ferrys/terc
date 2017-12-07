# -*- coding: utf-8 -*-

import dataset
import numpy as np

from keras.optimizers import SGD
from keras import backend as K

from sklearn.metrics import log_loss
from sklearn.metrics import fbeta_score

from keras import backend as K

from keras.models import model_from_json

import accuracy
import predictions

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

# Make predictions
predictions_test = loaded_model.predict(X_test, batch_size=batch_size, verbose=1)
predictions_test = [[1 if predictions_test[i][j] > 0.5 else 0 for j in range(predictions_test.shape[1])] for i in range(predictions_test.shape[0])]
predictions_test = np.array(predictions_test)

test_images = 'Terc_Images\\processed_images\\testing_data.csv'
ids = predictions.get_ids(test_images)

predictions.save_predictions(ids, predictions_test, 'testing')

pred_path = 'predictions/predictions_testing.csv'

# Display accuracy
print("Testing accuracy")
test_category_accuracy = accuracy.get_category_accuracy('Terc_Images\\processed_images\\testing_data.csv', pred_path, 'testing')
test_overall_accuracy = accuracy.get_overall_accuracy('Terc_Images\\processed_images\\testing_data.csv', pred_path)
print(test_overall_accuracy)
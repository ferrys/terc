# -*- coding: utf-8 -*-

import csv
import numpy as np
import glob
import cv2

from keras.optimizers import SGD
from keras import backend as K

from sklearn.metrics import log_loss
from sklearn.metrics import fbeta_score

from keras import backend as K

from keras.models import model_from_json

import accuracy
import predictions

# load test data
image_path = 'Terc_Images\\processed_images'
X_test_files = []
X_test = []

with open(image_path + '/testing_data.csv', 'rt') as f:
	reader = csv.reader(f, delimiter = ',')
	next(reader, None)
	for row in reader:
		X_test_files.append(image_path + "\\" + row[0])

# all_images = glob.glob(image_path + '/*.jpg')

for file in X_test_files:
	img = file + '/*.jpg'
	image = cv2.imread(img, cv2.INTER_LINEAR)
	image = image.astype(np.float)
	image = np.multiply(image, 1.0 / 255.0)
	X_test.append(image)

# for file in all_images:
# 	if file in X_test_files:
# 	  image = cv2.imread(file, cv2.INTER_LINEAR)
# 	  image = image.astype(np.float)
# 	  image = np.multiply(image, 1.0 / 255.0)
# 	  X_test.append(image)

X_test = np.array(X_test)

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
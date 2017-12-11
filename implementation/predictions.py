import os
import numpy as np
import csv

def get_ids(images_csv):

	ids = []
	with open(images_csv, 'rt') as f:
		reader = csv.reader(f, delimiter = ',')
		for row in reader:
			ids.append(row[0])
	f.close()
	return ids


def save_predictions(ids, predictions, phase, lr=''):
	tags = np.array(['Volcano', 'Sunrise Sunset', 'ISS Structure', 'Stars', 'Night', 'Aurora', 'Movie', 'Day', 'Moon', 'Inside ISS', 'Dock Undock', 'Cupola']).reshape(1,12)

	predictions = np.concatenate((tags, predictions))

	if not os.path.exists('predictions'):
		os.mkdir('predictions')

	with open('predictions/predictions_' + phase + lr + '.csv', 'w') as f:
		for i, image in enumerate(predictions):
			f.write(ids[i] + ',')
			for pred in image:
				f.write(pred + ',')
			f.write('\n')
	f.close()


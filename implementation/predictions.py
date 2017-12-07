import numpy as np
import csv

predictions_valid = np.array([[.56,.02,.73],[.01,.84,.51],[.02,.1,.9]])

predictions = [[1 if predictions_valid[i][j] > 0.5 else 0 for j in range(predictions_valid.shape[1])] for i in range(predictions_valid.shape[0])]
predictions = np.asarray(predictions) 

ids = []
with open('validation_data.csv', 'rt') as f:
	reader = csv.reader(f, delimiter = ',')
	for row in reader:
		ids.append(row[0])
f.close()

tags = 'Volcano,Sunrise Sunset,ISS Structure,Stars,Night,Aurora,Movie,Day,Moon,Inside ISS,Dock Undock,Cupola'

np.savetxt('predictions.csv', predictions, fmt='%.0f', delimiter=',', header=tags, comments='')

predictions = []
with open('predictions.csv', 'rt') as f:
	reader = csv.reader(f, delimiter = ',')
	for row in reader:
		predictions.append(row)
f.close()


with open('predictions.csv', 'w') as f:
	for i in range(len(ids)):
		f.write(ids[i] + ',')
		for j in range(len(predictions[0])):
			f.write(predictions[i][j] + ',')
		f.write('\n')
f.close()



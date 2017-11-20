import csv
import pandas as pd 
import numpy as np

valid_real = pd.DataFrame.from_csv('validation_data.csv').as_matrix()
valid_pred = pd.DataFrame.from_csv('predictions.csv',header=None).reset_index().as_matrix()

## PER CATEGORY ACCURACY

#accuracy = np.zeros(12)

# for j in range(valid_real.shape[1]):
# 	total_correct = 0
# 	for i in range(valid_real.shape[0]):
# 		if valid_real[i][j] == valid_pred[i][j]:
# 			total_correct += 1
# 	accuracy[j] = total_correct/valid_real.shape[0]

# tags = ['Volcano', 'Sunrise Sunset', 'ISS Structure', 'Stars', 'Night', 'Aurora', 'Movie', 'Day', 'Moon', 'Inside ISS', 'Dock Undock', 'Cupola']

# with open('accuracy.csv', 'w') as f:
# 	for tag in tags:
# 		f.write(tag + ',')
# 	f.write('\n')
# 	for tag_accuracy in accuracy:
# 		f.write(str(tag_accuracy) + ',')
# 	f.close()


## OVERALL ACCURACY

total_correct = 0
for i in range(valid_real.shape[0]):
	locally_correct = 0
	for j in range(valid_real.shape[1]):
		if valid_real[i][j] == valid_pred[i][j]:
			locally_correct += 1
	if locally_correct == valid_real.shape[1]:
		total_correct += 1
accuracy = total_correct/valid_real.shape[0]

print(accuracy)
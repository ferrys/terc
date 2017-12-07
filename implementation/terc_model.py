import numpy as np
from keras import backend as K

from keras.models import model_from_json


batch_size = 10

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()


loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')

predictions_terc = loaded_model.predict(X_test, batch_size=batch_size, verbose=1)
predictions = [[1 if predictions_terc[i][j] > 0.5 else 0 for j in range(predictions_terc.shape[1])] for i in range(predictions_terc.shape[0])]

predictions = np.asarray(predictions)

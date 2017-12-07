import numpy as np
from keras import backend as K
import resize
from keras.models import model_from_json
import sys
import os
import glob
import cv2
import predictions
import insert_tags

def get_images(image_directory):
    images = []
    img_names = ["id"]
    resized_path = resize.resize_images(image_directory)
    resized_files = glob.glob(resized_path + '/*.jpg')
    
    for file in resized_files:
        if os.name == 'nt':
            img_names.append(file[file.rindex("\\")+1:]) ## CHANGE THIS TO RELATIVE FILE NAME
        elif os.name == 'posix':
            img_names.append(file[file.rindex("/")+1:])
        else:
            print("Unsupported operating system. Please use Windows or Mac.")
        image = cv2.imread(file, cv2.INTER_LINEAR)
        image = image.astype(np.float)
        image = np.multiply(image, 1.0 / 255.0)
        images.append(image)
    images = np.array(images)
    img_names = np.array(img_names)
        
    return img_names, images

def predict(image_directory):
    
    img_names, images = get_images(image_directory)
    print(img_names)
    batch_size = 10
    
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model.h5')
    
    predictions_terc = loaded_model.predict(images, batch_size=batch_size, verbose=1)
    predictions_array = [[1 if predictions_terc[i][j] > 0.5 else 0 for j in range(predictions_terc.shape[1])] for i in range(predictions_terc.shape[0])]
    
    predictions_array = np.asarray(predictions_array)
    print(predictions_array)
    predictions.save_predictions(img_names, predictions_array)
    
    predictions_path = "predictions/predictions.csv"
    insert_tags.insert_tags(predictions_path, image_directory)
    


if __name__ == "__main__":
    program_name = sys.argv[0]
    image_directory = sys.argv[1]
    predict(image_directory)
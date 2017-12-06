import cv2
import pandas as pd
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []

    print('Going to read training images')

    resized_files = glob.glob(train_path + '/*.jpg')

    for file in resized_files:
      img_names.append(file[file.rindex("\\")+1:]) ## CHANGE THIS TO RELATIVE FILE NAME
      image = cv2.imread(file, cv2.INTER_LINEAR)
      image = image.astype(np.float)
      image = np.multiply(image, 1.0 / 255.0)
      images.append(image)

    tagsEncoding = pd.read_csv('extracted_tags.csv').fillna('n/a')

    tagsEncoding = tagsEncoding[tagsEncoding['id'].isin(img_names)].reset_index()

    tagsDf = tagsEncoding.copy().drop(['id'], axis=1)

    for c in classes:
      tagsEncoding[c] = 0  

    for row in range(len(tagsDf)):
      for col in range(len(tagsDf.columns)):
          if tagsDf.iloc[row,col] != 'n/a':
            tag = tagsDf.iloc[row,col]
            tagsEncoding.set_value(row, tag, 1)

    tagsEncoding = tagsEncoding[['Volcano', 'Sunrise Sunset', 'ISS Structure', 'Stars', 'Night', 'Aurora', 'Movie', 'Day', 'Moon', 'Inside ISS', 'Dock Undock', 'Cupola']]
    labels = tagsEncoding.as_matrix()

    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)

    return images, labels, img_names

class DataSet(object):

  def __init__(self, images, labels, img_names):
    self._num_examples = labels.shape[0]
    self._images = images
    self._labels = labels
    self._img_names = img_names

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names


def read_train_sets(train_path, image_size, classes, validation_size, test_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names = load_train(train_path, image_size, classes)
  images, labels, img_names  = shuffle(images, labels, img_names)

  if isinstance(validation_size, float):
    validation_size = int(validation_size * labels.shape[0])

  if isinstance(test_size, float):
    test_size = int(test_size * labels.shape[0])

  X_test = images[:test_size]
  y_test = labels[:test_size]
  test_img_names = img_names[:test_size]

  X_valid = images[test_size:test_size+validation_size]
  y_valid = labels[test_size:test_size+validation_size]
  validation_img_names = img_names[test_size:test_size+validation_size]

  X_train = images[test_size+validation_size:]
  y_train = labels[test_size+validation_size:]
  train_img_names = img_names[test_size+validation_size:]

  data_sets.train = DataSet(X_train, y_train, train_img_names)
  data_sets.valid = DataSet(X_valid, y_valid, validation_img_names)
  data_sets.test = DataSet(X_test, y_test, test_img_names)

  with open(train_path + '/training_data.csv', 'w') as f:
    f.write('id, Volcano, Sunrise Sunset, ISS Structure, Stars, Night, Aurora, Movie, Day, Moon, Inside ISS, Dock Undock, Cupola')
    f.write('\n')
    for index, name in enumerate(data_sets.train._img_names):
      f.write(name + ',')
      for tag in data_sets.train._labels[index]:
        f.write(str(tag) + ',')
      f.write('\n')
    f.close()

  with open(train_path + '/validation_data.csv', 'w') as f:
    f.write('id, Volcano, Sunrise Sunset, ISS Structure, Stars, Night, Aurora, Movie, Day, Moon, Inside ISS, Dock Undock, Cupola')
    f.write('\n')
    for index, name in enumerate(data_sets.valid._img_names):
      f.write(name + ',')
      for tag in data_sets.valid._labels[index]:
        f.write(str(tag) + ',')
      f.write('\n')
    f.close()

  with open(train_path + '/testing_data.csv', 'w') as f:
    f.write('id, Volcano, Sunrise Sunset, ISS Structure, Stars, Night, Aurora, Movie, Day, Moon, Inside ISS, Dock Undock, Cupola')
    f.write('\n')
    for index, name in enumerate(data_sets.test._img_names):
      f.write(name + ',')
      for tag in data_sets.test._labels[index]:
        f.write(str(tag) + ',')
      f.write('\n')
    f.close()

  return X_train, y_train, X_valid, y_valid, X_test, y_test


# if __name__ == "__main__":
#     from tensorflow import set_random_seed
#
#     set_random_seed(2)
#
#     batch_size = 32
#
#     # Prepare input data
#     classes = ['Volcano', 'Sunrise Sunset', 'ISS Structure', 'Stars', 'Night', 'Aurora', 'Movie', 'Day', 'Moon',
#                'Inside ISS', 'Dock Undock', 'Cupola']
#     num_classes = len(classes)
#
#     # 20% of the data will automatically be used for validation
#     validation_size = 0.2
#     test_size = 0.2
#     img_size = 224
#     num_channels = 3
#     train_path = 'Terc_Images\\processed_images'
#
#     # We shall load all the training and validation images and labels into memory using openCV and use that during training
#     X_train, y_train, X_valid, y_valid = read_train_sets(train_path, img_size, classes, validation_size=validation_size, test_size=test_size)
#
#     print(X_train.shape)
#     print(y_train.shape)
#     print(X_valid.shape)
#     print(y_valid.shape)
#
#     # print('Complete reading input data. Will Now print a snippet of it')
#     # print('Number of files in Training-set:\t\t{}'.format(len(data.train.labels)))
#     # print('Number of files in Validation-set:\t{}'.format(len(data.valid.labels)))
#     # print('Number of files in Test-set:\t{}'.format(len(data.test.labels)))
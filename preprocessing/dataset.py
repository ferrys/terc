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
    cls = []


    print('Going to read training images')

    files = glob.glob(train_path + '/*.jpg')
    for file in files:
      image = cv2.imread(file)
      image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
      image = image.astype(np.float)
      image = np.multiply(image, 1.0 / 255.0)
      images.append(image)
      img_names.append(file[file.rindex("\\")+1:]) ## CHANGE THIS TO RELATIVE FILE NAME
      cls.append(['Volcano', 'Sunrise Sunset', 'ISS Structure', 'Stars', 'Night', 'Aurora', 'Movie', 'Day', 'Moon', 'Inside ISS', 'Dock Undock', 'Cupola'])
    
    tagsEncoding = pd.read_csv('extracted_tags.csv', error_bad_lines=False).fillna('n/a')

    tagsEncoding = tagsEncoding[tagsEncoding['id'].isin(img_names)]

    print(len(img_names))
    print(len(tagsEncoding))

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
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls

class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  print('Training data:')
  print(data_sets.train._img_names)

  print('Validation data')
  print(data_sets.valid._img_names)

  return data_sets

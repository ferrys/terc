import cv2
import pandas as pd
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_train(train_path, image_size, classes):
    #images = []
    labels = []
    img_names = []
    cls = []

    processed_dir = train_path + '/processed_images'
    if not os.path.isdir(processed_dir):
      os.mkdir(processed_dir)

    print('Going to read training images')

    files = glob.glob(train_path + '/*.jpg')
    processed_files = glob.glob(processed_dir + '/*.jpg')

    for file in files:
      img_names.append(file[file.rindex("\\")+1:]) ## CHANGE THIS TO RELATIVE FILE NAME
      image = cv2.imread(file)
      #image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
      image = cv2.resize(image, (image_size, image_size),0,0)
      cv2.imwrite(processed_dir + '/' + file[file.rindex("\\")+1:], image)
      os.remove(file)

      # image = image.astype(np.float)
      # image = np.multiply(image, 1.0 / 255.0)
      # images.append(image)
      # cls.append(['Volcano', 'Sunrise Sunset', 'ISS Structure', 'Stars', 'Night', 'Aurora', 'Movie', 'Day', 'Moon', 'Inside ISS', 'Dock Undock', 'Cupola'])
    
    for file in processed_files:
      img_names.append(file[file.rindex('\\')+1:])

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

    #images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    # cls = np.array(cls)
    
    return labels, img_names
    #return images, labels, img_names, cls

class DataSet(object):

  #def __init__(self, images, labels, img_names, cls):
  def __init__(self, labels, img_names):
    self._num_examples = labels.shape[0]

    #self._images = images
    self._labels = labels
    self._img_names = img_names
    #self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  # @property
  # def images(self):
  #   return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  # @property
  # def cls(self):
  #   return self._cls

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

    return self._labels[start:end], self._img_names[start:end]
    #return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size, test_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  labels, img_names = load_train(train_path, image_size, classes)
  labels, img_names = shuffle(labels, img_names) 
  
  #images, labels, img_names, cls = load_train(train_path, image_size, classes)
  #images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  

  if isinstance(validation_size, float):
    validation_size = int(validation_size * labels.shape[0])

  if isinstance(test_size, float):
    test_size = int(test_size * labels.shape[0])

  #test_images = images[:test_size]
  test_labels = labels[:test_size]
  test_img_names = img_names[:test_size]
  # test_cls = cls[:test_size]

  #validation_images = images[test_size:validation_size]
  validation_labels = labels[test_size:test_size+validation_size]
  validation_img_names = img_names[test_size:test_size+validation_size]
  # validation_cls = cls[test_size:validation_size]

  #train_images = images[validation_size:]
  train_labels = labels[test_size+validation_size:]
  train_img_names = img_names[test_size+validation_size:]
  # train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_labels, train_img_names)
  data_sets.valid = DataSet(validation_labels, validation_img_names)
  data_sets.test = DataSet(test_labels, test_img_names)

  # data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  # data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)
  # data_sets.test = DataSet(test_images, test_labels, test_img_names, test_cls)

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

  # print('Training data:')
  # print(data_sets.train._img_names)

  # print('Validation data')
  # print(data_sets.valid._img_names)

  # print('Test data')
  # print(data_sets.test._img_names)

  return data_sets

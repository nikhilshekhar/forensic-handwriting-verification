from tensorflow.python.framework import dtypes
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
import gzip


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_images(f):
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def extract_labels(f, one_hot=False, num_classes=2):
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

class DataSet(object):

    def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
          raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                          dtype)
        if fake_data:
          self._num_examples = 10000
          self.one_hot = one_hot
        else:
          assert images.shape[0] == labels.shape[0], (
              'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
          self._num_examples = images.shape[0]

          # Convert shape from [num examples, rows, columns, depth]
          # to [num examples, rows*columns] (assuming depth == 1)
          if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
          if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size=100):
        
        current_permutation = np.random.permutation(range(len(self._images)))
        epoch_images = self._images[current_permutation, ...]
        if self._labels is not None:
            epoch_labels = self._labels[current_permutation, ...]

        # Then iterate over the epoch
        self.current_batch_idx = 0
        #print(len(self._images))
        while self.current_batch_idx < len(self._images):
            end_idx = min(
                self.current_batch_idx + batch_size, len(self._images))
            #print(self.current_batch_idx)
            #print("end_idx",end_idx)
            this_batch = {
                'images': epoch_images[self.current_batch_idx:end_idx],
                'labels': epoch_labels[self.current_batch_idx:end_idx]
                if self.labels is not None else None
            }
            self.current_batch_idx += batch_size
            yield this_batch['images'], this_batch['labels']
    
    '''def next_batch(self, batch_size, fake_data=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._images = self._images[perm]
          self._labels = self._labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]
    '''
def read_data_sets(train_dir,
                     fake_data=False,
                     one_hot=False,
                     dtype=dtypes.float32,
                     reshape=True,
                     validation_size=5000):
      if fake_data:
          def fake():
              return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

          train = fake()
          validation = fake()
          test = fake()
          return base.Datasets(train=train, validation=validation, test=test)

      TRAIN_IMAGES = 'training-images-and-ubyte_4.gz'
      TRAIN_LABELS = 'training-labels-and-ubyte_4.gz'
      VALIDATION_IMAGES = 'validation-images-and-ubyte_4.gz'
      VALIDATION_LABELS = 'validation-labels-and_ubyte_4.gz'
      TEST_IMAGES = 'testing-images-and-ubyte_4.gz'
      TEST_LABELS = 'testing-labels-and-ubyte_4.gz'


      with open(train_dir+ TRAIN_IMAGES, 'rb') as f:
          train_images = extract_images(f)

      with open(train_dir+TRAIN_LABELS, 'rb') as f:
          train_labels = extract_labels(f, one_hot=one_hot)

      #train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
      #return train

      with open(train_dir + VALIDATION_IMAGES, 'rb') as f:
          validation_images = extract_images(f)

      with open(train_dir + VALIDATION_LABELS, 'rb') as f:
          validation_labels = extract_labels(f, one_hot=one_hot)

      with open(train_dir + TEST_IMAGES , 'rb') as f:
	  test_images = extract_images(f)
	
      with open(train_dir + TEST_LABELS , 'rb') as f:
	  test_labels = extract_labels(f , one_hot = one_hot)	
      #validation_images = train_images[:validation_size]
      #validation_labels = train_labels[:validation_size]
      #train_images = train_images[validation_size:]
      #train_labels = train_labels[validation_size:]

      train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
      validation = DataSet(validation_images,
                           validation_labels,
                           dtype=dtype,
                           reshape=reshape)
      test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

      return base.Datasets(train=train, validation=validation , test=test)

import datetime
import logging
import pickle
import sys
from os import path

import h5py
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from constants import DATA_ROOT


def randomize_split(data, split_ratio=0.8):
    # shuffle the dataset
    np.random.shuffle(data)

    # divide training and testing dataset
    training_count = int(len(data) * split_ratio)

    training_data = data[:training_count]
    testing_data = data[training_count:]
    return training_data, testing_data


def get_dataset(filename='signal-dataset.pkl'):
    if not path.exists(DATA_ROOT + filename):
        download(filename)

    with open(DATA_ROOT + filename, 'rb') as f:
        data = pickle.load(f)
        return data


def download(filename, base_url='https://s3-ap-southeast-1.amazonaws.com/usq.iothealth/iemocap/'):
    import urllib.request

    url = base_url + filename

    print('Beginning file download {}'.format(url))

    store_file = DATA_ROOT + filename
    urllib.request.urlretrieve(url, store_file)

    print("Downloaded and saved to file: {}".format(store_file))


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(
            f"There is no such a hdf5 file ({hdf5_name}). \nDownload from here: https://s3.ap-southeast-2.amazonaws.com/usq.iothealth.sidney/iemocap/{hdf5_name}")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def train(model, x, y, epochs, batch_size=4, log_base_dir='./logs'):
    print("Start Training")
    log_dir = log_base_dir + "/" + model.name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    callback_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=log_base_dir + "/" + model.name + '.h5',
            monitor='val_accuracy',
            save_best_only='True',
            verbose=1,
            mode='max'
        ), tensorboard_callback]

    history = model.fit(x, y,
                        batch_size=batch_size, epochs=epochs,
                        validation_split=0.2,
                        verbose=True,
                        callbacks=callback_list)
    return history, model

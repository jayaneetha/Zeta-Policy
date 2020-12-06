import pickle
from os import path

import numpy as np

from constants import PKL_ROOT


def randomize_split(data, split_ratio=0.8):
    # shuffle the dataset
    np.random.shuffle(data)

    # divide training and testing dataset
    training_count = int(len(data) * split_ratio)

    training_data = data[:training_count]
    testing_data = data[training_count:]
    return training_data, testing_data


def get_dataset(filename='signal-dataset.pkl'):
    if not path.exists(PKL_ROOT + filename):
        download(filename)

    with open(PKL_ROOT + filename, 'rb') as f:
        data = pickle.load(f)
        return data


def download(filename, base_url='https://s3-ap-southeast-1.amazonaws.com/usq.iothealth/iemocap/'):
    import urllib.request

    url = base_url + filename

    print('Beginning file download {}'.format(url))

    store_file = PKL_ROOT + filename
    urllib.request.urlretrieve(url, store_file)

    print("Downloaded and saved to file: {}".format(store_file))

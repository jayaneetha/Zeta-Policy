import numpy as np

from datastore import Datastore
from feature_type import FeatureType
from framework import get_dataset, randomize_split


class SaveeDatastore(Datastore):
    data_pkl = None
    data = []
    pre_train_data = []

    def __init__(self, feature_type: FeatureType):
        if not (FeatureType.MFCC == feature_type):
            raise Exception("Only supports {}".format(FeatureType.MFCC.name))

        self.data_pkl = get_dataset("savee_sr_22k_2sec_4-classes.pkl")

        rl_data, pre_train_data = randomize_split(self.data_pkl, split_ratio=0.7)

        self.data = rl_data
        self.pre_train_data = pre_train_data

    def get_data(self):
        np.random.shuffle(self.data)
        x_train_mfcc = np.array([d[FeatureType.MFCC.name] for d in self.data])
        y_train_emo = np.array([d['y_emo'] for d in self.data])

        return (x_train_mfcc, y_train_emo, None), (None, None, None)

    def get_pre_train_data(self):
        x_train_mfcc = np.array([d[FeatureType.MFCC.name] for d in self.pre_train_data])
        y_train_emo = np.array([d['y_emo'] for d in self.pre_train_data])

        return x_train_mfcc, y_train_emo, None

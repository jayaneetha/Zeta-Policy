import librosa
import numpy as np

from constants import SR, NUM_MFCC
from datastore import Datastore
from feature_type import FeatureType
from framework import get_dataset, randomize_split


class IemocapDatastore(Datastore):
    data_pkl = None
    data = []
    pre_train_data = []

    def __init__(self, feature_type: FeatureType) -> None:
        self.data_pkl = get_dataset("signal-no-silent-4-class-dataset-2sec_sr_22k.pkl")
        for d in self.data_pkl:
            single_file = {}
            feature = librosa.feature.mfcc(d['x'], sr=SR, n_mfcc=NUM_MFCC)
            single_file[feature_type.name] = feature
            single_file['y_emo'] = d['emo']
            single_file['y_gen'] = d['gen']
            self.data.append(single_file)

        rl_data, pre_train_data = randomize_split(self.data, split_ratio=0.7)

        self.data = rl_data
        self.pre_train_data = pre_train_data

    def get_data(self):
        training_data, testing_data = randomize_split(self.data)

        x_train_mfcc = np.array([d[FeatureType.MFCC.name] for d in training_data])
        y_train_emo = np.array([d['y_emo'] for d in training_data])
        y_train_gen = np.array([d['y_gen'] for d in training_data])

        x_test_mfcc = np.array([d[FeatureType.MFCC.name] for d in testing_data])
        y_test_emo = np.array([d['y_emo'] for d in testing_data])
        y_test_gen = np.array([d['y_gen'] for d in testing_data])

        return (x_train_mfcc, y_train_emo, y_train_gen), (x_test_mfcc, y_test_emo, y_test_gen)

    def get_pre_train_data(self):
        training_data = self.pre_train_data

        x_train_mfcc = np.array([d[FeatureType.MFCC.name] for d in training_data])
        y_train_emo = np.array([d['y_emo'] for d in training_data])
        y_train_gen = np.array([d['y_gen'] for d in training_data])

        return x_train_mfcc, y_train_emo, y_train_gen

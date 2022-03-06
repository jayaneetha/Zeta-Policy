from constants import DATA_ROOT
from datastore import Datastore
from feature_type import FeatureType
from framework import read_hdf5


class ESDDatastore(Datastore):
    def __init__(self, feature_type: FeatureType):
        if not (FeatureType.MFCC == feature_type):
            raise Exception("Only supports {}".format(FeatureType.MFCC.name))

        self.train_mfcc = read_hdf5(f"{DATA_ROOT}/train_vltp_noised_balanced_esd.h5", "mfcc")
        self.train_emotion = read_hdf5(f"{DATA_ROOT}/train_vltp_noised_balanced_esd.h5", "emotion_one_hot")

    def get_data(self):
        return (self.train_mfcc, self.train_emotion, None), (None, None, None)

    def get_pre_train_data(self):
        mfcc = read_hdf5(f"{DATA_ROOT}/valid_vltp_noised_balanced_esd.h5", "mfcc")
        emotion = read_hdf5(f"{DATA_ROOT}/valid_vltp_noised_balanced_esd.h5", "emotion_one_hot")

        return mfcc, emotion, None

    def get_eval_data(self):
        mfcc = read_hdf5(f"{DATA_ROOT}/test_vltp_noised_balanced_esd.h5", "mfcc")
        emotion = read_hdf5(f"{DATA_ROOT}/test_vltp_noised_balanced_esd.h5", "emotion_one_hot")

        return mfcc, emotion, None

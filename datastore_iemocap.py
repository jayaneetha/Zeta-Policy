from constants import DATA_ROOT
from datastore import Datastore
from feature_type import FeatureType
from framework import read_hdf5


class IemocapDatastore(Datastore):
    data_pkl = None
    data = []
    pre_train_data = []

    def __init__(self, feature_type: FeatureType, custom_split: float = None) -> None:
        if not (FeatureType.MFCC == feature_type):
            raise Exception("Only supports {}".format(FeatureType.MFCC.name))

        base_h5_file = "vltp_noised_balanced_iemocap.h5"

        if custom_split is None:
            self.train_mfcc = read_hdf5(f"{DATA_ROOT}/train_{base_h5_file}", "mfcc")
            self.train_emotion = read_hdf5(f"{DATA_ROOT}/train_{base_h5_file}", "emotion_one_hot")
            self.test_mfcc = read_hdf5(f"{DATA_ROOT}/test_{base_h5_file}", "mfcc")
            self.test_emotion = read_hdf5(f"{DATA_ROOT}/test_{base_h5_file}", "emotion_one_hot")
        else:
            assert 0 < custom_split < 1
            mfcc = read_hdf5(f"{DATA_ROOT}/{base_h5_file}", "mfcc")
            emotion_one_hot = read_hdf5(f"{DATA_ROOT}/{base_h5_file}", "emotion_one_hot")
            # emotion = read_hdf5(f"{DATA_ROOT}/{base_h5_file}", "emotion")

            training_count = int((len(mfcc) * custom_split))
            self.train_mfcc = mfcc[:training_count]
            self.train_emotion = emotion_one_hot[:training_count]
            self.test_mfcc = mfcc[training_count:]
            self.test_emotion = emotion_one_hot[training_count:]

        assert len(self.train_mfcc) == len(self.train_emotion)
        assert len(self.test_mfcc) == len(self.test_emotion)

    def get_data(self):
        return (self.train_mfcc, self.train_emotion, None), (None, None, None)

    # def get_pre_train_data(self):
    #     mfcc = read_hdf5(f"{DATA_ROOT}/valid_vltp_noised_balanced_iemocap.h5", "mfcc")
    #     emotion = read_hdf5(f"{DATA_ROOT}/valid_vltp_noised_balanced_iemocap.h5", "emotion_one_hot")
    #
    #     return mfcc, emotion, None

    def get_testing_data(self):
        # mfcc = read_hdf5(f"{DATA_ROOT}/test_vltp_noised_balanced_iemocap.h5", "mfcc")
        # emotion = read_hdf5(f"{DATA_ROOT}/test_vltp_noised_balanced_iemocap.h5", "emotion")

        return self.test_mfcc, self.test_emotion, None

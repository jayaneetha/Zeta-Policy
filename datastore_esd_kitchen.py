from constants import DATA_ROOT
from datastore import Datastore
from feature_type import FeatureType
from framework import read_hdf5


class KitchenESDDatastore(Datastore):
    def __init__(self, feature_type: FeatureType, custom_split: float = None, background_noise_db=None):
        if not (FeatureType.MFCC == feature_type):
            raise Exception("Only supports {}".format(FeatureType.MFCC.name))
        if background_noise_db is None:
            base_h5_file = "kitchen_bg_vltp_noised_balanced_esd.h5"
        else:
            base_h5_file = f"kitchen_bg_db{background_noise_db}_vltp_noised_balanced_esd.h5"

        if custom_split is None:
            self.train_mfcc = read_hdf5(f"{DATA_ROOT}/train_{base_h5_file}", "mfcc")
            self.train_emotion = read_hdf5(f"{DATA_ROOT}/train_{base_h5_file}", "emotion_one_hot")
            self.target_mfcc = read_hdf5(f"{DATA_ROOT}/test_{base_h5_file}", "mfcc")
            self.target_emotion = read_hdf5(f"{DATA_ROOT}/test_{base_h5_file}", "emotion_one_hot")
        else:
            assert 0 < custom_split < 1
            mfcc = read_hdf5(f"{DATA_ROOT}/{base_h5_file}", "mfcc")
            emotion_one_hot = read_hdf5(f"{DATA_ROOT}/{base_h5_file}", "emotion_one_hot")
            # emotion = read_hdf5(f"{DATA_ROOT}/{base_h5_file}", "emotion")

            training_count = int((len(mfcc) * custom_split))
            self.train_mfcc = mfcc[:training_count]
            self.train_emotion = emotion_one_hot[:training_count]
            self.target_mfcc = mfcc[training_count:]
            self.target_emotion = emotion_one_hot[training_count:]

        assert len(self.train_mfcc) == len(self.train_emotion)
        assert len(self.target_mfcc) == len(self.target_emotion)

    def get_data(self):
        return (self.train_mfcc, self.train_emotion, None), (None, None, None)

    def get_testing_data(self):
        return self.target_mfcc, self.target_emotion, None

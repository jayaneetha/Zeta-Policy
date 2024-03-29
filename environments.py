import gym
import numpy as np

from constants import EMOTIONS, NUM_MFCC, NO_features
from datastore import CombinedDatastore
from datastore_emodb import EmoDBDatastore
from datastore_emodb_kitchen import KitchenEmoDBDatastore
from datastore_esd import ESDDatastore
from datastore_esd_kitchen import KitchenESDDatastore
from datastore_iemocap import IemocapDatastore
from datastore_improv import ImprovDatastore
from datastore_savee import SaveeDatastore
from feature_type import FeatureType


class AbstractEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore) -> None:
        super().__init__()
        self.itr = 0

        self.X = []
        self.Y = []
        self.num_classes = len(EMOTIONS)

        self.data_version = data_version

        self.datastore = datastore

        self.set_data()

        self.action_space = gym.spaces.Discrete(self.num_classes)
        self.observation_space = gym.spaces.Box(-1, 1, [NUM_MFCC, NO_features])

    def step(self, action):
        assert self.action_space.contains(action)
        reward = -0.1 + int(action == np.argmax(self.Y[self.itr]))
        # reward = 1 if action == self.Y[self.itr] else -1

        done = (len(self.X) - 2 <= self.itr)

        next_state = self.X[self.itr + 1]

        info = {
            "ground_truth": np.argmax(self.Y[self.itr]),
            "itr": self.itr,
            "correct_inference": int(action == np.argmax(self.Y[self.itr]))
        }
        self.itr += 1

        return next_state, reward, done, info

    def get_next_state(self):
        # TODO: Fix after try-out
        #  Trying to remove outlier happy utterances
        if EMOTIONS[np.argmax(self.Y[self.itr + 1])] == 'hap':
            if (-495.6675 < np.mean(self.X[self.itr + 1], axis=1)[0]) or (
                    np.mean(self.X[self.itr + 1], axis=1)[0] < -262.93038):
                return self.X[self.itr + 1]
            else:
                self.itr += 1
                return self.get_next_state()
        else:
            return self.X[self.itr + 1]

    def render(self, mode='human'):
        print("Not implemented \t i: {}".format(self.itr))

    def reset(self):
        self.itr = 0
        self.set_data()
        return self.X[self.itr]

    def set_data(self):
        self.X = []
        self.Y = []

        (x_train, y_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = self.datastore.get_data()

        assert len(x_train) == len(y_train)
        self.X = x_train
        self.Y = y_train


class IemocapEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore: IemocapDatastore = None, custom_split: float = None) -> None:
        if datastore is None:
            datastore = IemocapDatastore(FeatureType.MFCC, custom_split)
        super().__init__(data_version=data_version, datastore=datastore)


class ImprovEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore: ImprovDatastore = None, custom_split: float = None) -> None:
        if datastore is None:
            datastore = ImprovDatastore(FeatureType.MFCC, custom_split)
        super().__init__(data_version=data_version, datastore=datastore)


class SaveeEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version) -> None:
        super().__init__(data_version=data_version, datastore=SaveeDatastore(FeatureType.MFCC))


class ESDEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore: ESDDatastore = None, custom_split: float = None) -> None:
        if datastore is None:
            datastore = ESDDatastore(FeatureType.MFCC, custom_split)
        super().__init__(data_version=data_version, datastore=datastore)


class KitchenESDEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore: KitchenESDDatastore = None, custom_split: float = None) -> None:
        if datastore is None:
            datastore = KitchenESDDatastore(FeatureType.MFCC, custom_split)
        super().__init__(data_version=data_version, datastore=datastore)


class EmoDBEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore: EmoDBDatastore = None, custom_split: float = None) -> None:
        if datastore is None:
            datastore = EmoDBDatastore(FeatureType.MFCC, custom_split)
        super().__init__(data_version=data_version, datastore=datastore)


class KitchenEmoDBEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore: KitchenEmoDBDatastore = None, custom_split: float = None) -> None:
        if datastore is None:
            datastore = KitchenEmoDBDatastore(FeatureType.MFCC, custom_split)
        super().__init__(data_version=data_version, datastore=datastore)


class CombinedEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore: CombinedDatastore = None, custom_split: float = None) -> None:
        if datastore is None:
            datastore = CombinedDatastore(FeatureType.MFCC, custom_split)
        super().__init__(data_version=data_version, datastore=datastore)

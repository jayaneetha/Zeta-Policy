from enum import Enum


class FeatureType(Enum):
    MFCC = 1
    MEL = 2
    LOG_MEL = 3
    STFT = 4
    RAW = 5
    PITCH = 6

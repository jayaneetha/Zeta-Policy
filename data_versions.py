from enum import Enum


class DataVersions(Enum):
    IEMOCAP = 1,
    SAVEE = 2,  # SAVEE dataset
    IMPROV = 3,  # MSP-IMPROV Dataset
    ESD = 4,  # ESD Dataset
    COMBINED = 5,  # Combined Dataset
    EMODB = 6,  # EmoDB Dataset
    KITCHEN_EMODB = 7,  # EmoDB Dataset with Kitchen background sound
    KITCHEN_ESD = 8,  # ESD Dataset with Kitchen background sound
    KITCHEN_ESD_DB0 = 9,  # ESD Dataset with Kitchen background sound SNR db 0
    KITCHEN_ESD_DBn5 = 10,  # ESD Dataset with Kitchen background sound SNR db -5
    KITCHEN_ESD_DBn10 = 11,  # ESD Dataset with Kitchen background sound SNR db -10
    KITCHEN_ESD_DBp5 = 12,  # ESD Dataset with Kitchen background sound SNR db +5
    KITCHEN_ESD_DBp10 = 13,  # ESD Dataset with Kitchen background sound SNR db +10

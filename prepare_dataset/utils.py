import glob
import re

import numpy as np

from constants import EMOTIONS


def randomize_split(data, split_ratio=0.8):
    # shuffle the dataset
    np.random.shuffle(data)

    # divide training and testing dataset
    training_count = int(len(data) * split_ratio)

    training_data = data[:training_count]
    testing_data = data[training_count:]
    return training_data, testing_data


def remove_none(lst):
    res = []
    for val in lst:
        if val is not None:
            res.append(val)

    return res


# --------------------- IEMOCAP ---------------------
def get_details_of_path(path):
    sections = path.split('/')
    details = {
        'SESSION_ID': sections[-5],
        'DIALOG_ID': sections[-2],
        'SENTENCE_ID': sections[-1].split(".")[0]
    }
    return details


def get_emotion_of_sentence(session_id, dialog_id, sentence_id, dataset_base_path):
    emo_evaluation_file = dataset_base_path + '/' + session_id + '/dialog/EmoEvaluation/' + dialog_id + '.txt'
    with open(emo_evaluation_file, 'r') as f:
        targets = [line for line in f if sentence_id in line]
        emo = targets[0].split('\t')[2]
        if emo in EMOTIONS:
            return EMOTIONS.index(emo)
        else:
            return -1


def get_iemocap_files_list(dataset_base_path, session_id='*'):
    file_list = []  # final file list
    r = re.compile(".*_impro\d{2}")

    if session_id == '*':
        session_dirs = glob.glob(dataset_base_path + "/Session*/")
        for session_dir in session_dirs:
            wavs = glob.glob(session_dir + "sentences/wav/*/*.wav")
            filtered_wavs = list(filter(r.match, wavs))
            file_list.extend(filtered_wavs)
    else:
        session_dir = dataset_base_path + "/" + session_id
        wavs = glob.glob(session_dir + "/sentences/wav/*/*.wav")
        filtered_wavs = list(filter(r.match, wavs))
        file_list.extend(filtered_wavs)

    return file_list


# --------------------- ESD ---------------------
def get_esd_emotion(file_path):
    emo_map = {
        'Sad': 'sad',
        'Happy': 'hap',
        'Angry': 'ang',
        'Neutral': 'neu'
    }
    e = file_path.split('/')[-3]
    emo_i = emo_map[e]
    if emo_i in EMOTIONS:
        return EMOTIONS.index(emo_i)
    else:
        return -1


def get_esd_files_list(dataset_base_path):
    ESD_SPEAKERS = ['0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']
    ESD_SUBSET = 'train'
    Es = ['Angry', 'Happy', 'Neutral', 'Sad']
    Ss = ESD_SPEAKERS
    file_list = []

    for s in Ss:
        for e in Es:
            files = glob.glob(f"{dataset_base_path}/EmotionalSpeechDataset/{s}/{e}/{ESD_SUBSET}/*.wav")
            file_list = file_list + files

    return file_list


# --------------------- EmoDB ---------------------
def get_emodb_emotion(file_path):
    emo_map = {
        'T': 'sad',
        'F': 'hap',
        'W': 'ang',
        'N': 'neu',
        'L': '',
        'E': '',
        'A': ''
    }
    e = file_path.split('/')[-1][5]
    emo_i = emo_map[e]
    if emo_i in EMOTIONS:
        return EMOTIONS.index(emo_i)
    else:
        return -1


def get_emodb_files_list(dataset_base_path):
    files = glob.glob(f"{dataset_base_path}/*.wav")
    return files


# --------------------- MSP-IMPROV ---------------------
def get_improv_emotion(file_path):
    emo_map = {
        'S': 'sad',
        'H': 'hap',
        'A': 'ang',
        'N': 'neu'
    }
    sections = file_path.split('/')
    e = sections[-1].split('-')[-4][-1]
    emo_i = emo_map[e]
    if emo_i in EMOTIONS:
        return EMOTIONS.index(emo_i)
    else:
        return -1


def get_improv_files_list(dataset_base_path, session_id='*'):
    file_list = glob.glob(f"{dataset_base_path}/{session_id}/*/R/*.wav")
    return file_list

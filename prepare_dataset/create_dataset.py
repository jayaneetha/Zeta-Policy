import random

import librosa
import nlpaug.augmenter.audio as naa
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm

from audio_utils import load_wav, add_noise, remove_silent, split_audio, \
    _get_mfcc, get_noise_from_sound
from constants import SR, DURATION, EMOTIONS, NUM_MFCC
from data_versions import DataVersions
from hdf5_utils import write_hdf5
from prepare_dataset.utils import randomize_split, get_details_of_path, get_emotion_of_sentence, get_esd_emotion, \
    get_emodb_emotion, get_improv_emotion, remove_none, get_iemocap_files_list, get_esd_files_list, \
    get_emodb_files_list, get_improv_files_list

aug = naa.VtlpAug(sampling_rate=SR)

BACKGROUND_FILENAME = ''  # path of the background audio file

IEMOCAP_DIR = ''
ESD_DIR = ''
MSP_DIR = ''
EMODB_DIR = ''


def process_file(filename, sampling_rate, noise=False, add_background=False, bg_splits=None, snr=None,
                 augment=False):
    audio, sr = load_wav(filename, sr=sampling_rate)

    if noise:
        audio = add_noise(audio)

    if augment:
        audio = aug.augment(audio)

    audio = remove_silent(audio)

    audio = split_audio(audio, sr, DURATION)[0]

    if add_background:
        audio_bg = bg_splits[random.randint(0, len(bg_splits) - 1)]
        audio_bg_noise = get_noise_from_sound(audio, audio_bg, snr)
        audio = (audio + audio_bg_noise) - np.mean(audio_bg_noise)  # normalize to 0

    mfcc = _get_mfcc(audio, sr, n_mfcc=NUM_MFCC)

    return audio, mfcc


def process_files(files, dataset_filename, data_version: DataVersions, noise=False, add_background_audio=False,
                  balance_classes=False):
    sampling_rate = SR

    utterance_count = [0, 0, 0, 0]
    utterance_max_count = 90000

    emotion_files = {
        'hap': [],
        'sad': [],
        'ang': [],
        'neu': []
    }

    audios = []
    mfccs = []

    emotions = []

    background_audio, sr = librosa.load(BACKGROUND_FILENAME, sr=sampling_rate)
    splits = split_audio(background_audio, sr, DURATION)

    for f in tqdm(files):
        emo = -1

        if data_version == DataVersions.IEMOCAP:
            file_details = get_details_of_path(f)
            emo = get_emotion_of_sentence(file_details['SESSION_ID'], file_details['DIALOG_ID'],
                                          file_details['SENTENCE_ID'], IEMOCAP_DIR)

        if data_version == DataVersions.ESD:
            emo = get_esd_emotion(f)

        if data_version == DataVersions.EMODB:
            emo = get_emodb_emotion(f)

        if data_version == DataVersions.IMPROV:
            emo = get_improv_emotion(f)

        if emo > -1 and utterance_count[emo] < utterance_max_count:
            emotion_files[EMOTIONS[emo]].append(f)
            audio, mfcc = process_file(filename=f, sampling_rate=sampling_rate, noise=noise,
                                       add_background=add_background_audio, bg_splits=splits, snr=-5, augment=False)

            audios.append(audio)
            mfccs.append(mfcc)
            emotions.append(emo)
            utterance_count[emo] += 1

    if balance_classes:
        for i in range(EMOTIONS):
            balance_utterances = max(utterance_count) - utterance_count[i]
            _e = EMOTIONS[i]
            print(f"Balancing dataset for emotion: {_e} with {balance_utterances} utterances")
            for _i in tqdm(range(balance_utterances)):
                _f = random.choice(emotion_files[_e])
                audio, mfcc = process_file(filename=_f, sampling_rate=sampling_rate, noise=noise,
                                           add_background=add_background_audio, bg_splits=splits, augment=False)

                audios.append(audio)
                mfccs.append(mfcc)
                emotions.append(_e)
                utterance_count[_e] += 1

    audios = remove_none(audios)
    print(f"No. Records: {len(audios)}")
    assert len(audios) > 0
    mfccs = remove_none(mfccs)
    emotions = remove_none(emotions)

    dataset_filename = f'../pkl/{dataset_filename}'
    write_hdf5(dataset_filename, "audio", audios)
    write_hdf5(dataset_filename, "mfcc", mfccs)
    write_hdf5(dataset_filename, "emotion_one_hot", to_categorical(emotions, num_classes=len(EMOTIONS)))
    write_hdf5(dataset_filename, "emotion", emotions)
    print(f"Files write to: {dataset_filename}")


def run(data_version: DataVersions, base_filename: str, noise=False, add_background_audio=False, train=True,
        valid=True, test=True, full_dataset=False):
    files = []

    if data_version == DataVersions.IEMOCAP:
        files = get_iemocap_files_list(IEMOCAP_DIR)

    if data_version == DataVersions.ESD:
        files = get_esd_files_list(ESD_DIR)

    if data_version == DataVersions.EMODB:
        files = get_emodb_files_list(ESD_DIR)

    if data_version == DataVersions.IMPROV:
        files = get_improv_files_list(MSP_DIR)

    training_files, testing_files = randomize_split(files)
    training_files, validation_files = randomize_split(training_files)

    if full_dataset:
        process_files(files, base_filename, data_version, noise, add_background_audio=add_background_audio,
                      balance_classes=True)

    if valid:
        process_files(validation_files, 'valid_' + base_filename, data_version, noise, add_background_audio,
                      balance_classes=False)
    if train:
        process_files(training_files, 'train_' + base_filename, data_version, noise, add_background_audio,
                      balance_classes=True)
    if test:
        process_files(testing_files, 'test_' + base_filename, data_version, noise, add_background_audio,
                      balance_classes=False)


if __name__ == '__main__':
    run(DataVersions.IEMOCAP, 'vltp_noised_balanced_iemocap.h5', noise=True, add_background_audio=False,
        full_dataset=True)
    run(DataVersions.IMPROV, 'vltp_noised_balanced_mspimprov.h5', noise=True, add_background_audio=False,
        full_dataset=True)
    run(DataVersions.ESD, 'vltp_noised_balanced_esd.h5', noise=True, add_background_audio=False, full_dataset=True)
    run(DataVersions.ESD, 'kitchen_bg_vltp_noised_balanced_esd.h5', noise=True, add_background_audio=True,
        full_dataset=True)
    run(DataVersions.EMODB, 'vltp_noised_balanced_emodb.h5', noise=True, add_background_audio=False, full_dataset=True)
    run(DataVersions.EMODB, 'kitchen_bg_vltp_noised_balanced_emodb.h5', noise=True, add_background_audio=True,
        full_dataset=True)

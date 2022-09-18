import math

import librosa
import numpy as np

from constants import NUM_MFCC


def load_wav(filename, sr=None):
    audio, sr = librosa.load(filename, sr=sr)
    return audio, sr


def add_noise(signal, noise_intensity=0.005):
    y_noise = signal.copy()
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = noise_intensity * np.random.uniform() * np.amax(y_noise)
    y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])

    return y_noise


def get_noise_from_sound(signal, noise, snr):
    RMS_s = math.sqrt(np.mean(signal ** 2))
    # required RMS of noise
    RMS_n = math.sqrt(RMS_s ** 2 / (pow(10, snr / 10)))

    # current RMS of noise
    RMS_n_current = math.sqrt(np.mean(noise ** 2))
    noise = noise * (RMS_n / RMS_n_current)

    return noise


def remove_silent(signal, top_db=10):
    audio, index = librosa.effects.trim(signal, top_db=top_db)
    return audio


def add_missing_padding(audio, sr, duration):
    signal_length = duration * sr
    audio_length = audio.shape[0]
    padding_length = signal_length - audio_length
    if padding_length > 0:
        padding = np.zeros(padding_length)
        signal = np.hstack((audio, padding))
        return signal
    return audio


def split_audio(signal, sr, split_duration):
    length = split_duration * sr

    if length < len(signal):
        frames = librosa.util.frame(signal, frame_length=length, hop_length=length).T
        return frames
    else:
        audio = add_missing_padding(signal, sr, split_duration)
        frames = [audio]
        return np.array(frames)


def _get_mfcc(signal, sr, n_mfcc=NUM_MFCC):
    return librosa.feature.mfcc(signal, sr, n_mfcc=n_mfcc)

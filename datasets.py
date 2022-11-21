import os
import random

import h5py
import librosa
import numpy as np
from intervaltree import Interval, IntervalTree

import utils

MUSIC_NET_PATH = r"../Datasets/MusicNet/musicnet.npz"
MODAL_PATH = r"../Datasets/modal/onsets1.1.hdf5"
BOCK_SET_PATH = r"../Datasets/Boeck/"


class MusicNet(object):
    r"""
    https://homes.cs.washington.edu/~thickstn/musicnet.html
    A large classical music dataset with music note labeled as MIDI-like format.
    Onset are represented in samples.
    """

    SAMPLE_RATE = 44100

    def __init__(self):
        self.file = np.load(MUSIC_NET_PATH, 'rb', encoding='latin1', allow_pickle=True)
        self.keys = list(self.file.keys())
        self._pieces_cache = {}

    def get_keys(self) -> list:
        return self.keys

    def get_piece(self, key) -> 'MusicNet.Piece':
        if key not in self._pieces_cache:
            self._pieces_cache[key] = MusicNet.Piece(self.file[key])
        return self._pieces_cache[key]

    class Piece(object):
        def __init__(self, t):
            (wave, labels) = t
            self.wave = wave
            self.labels = labels
            self._onsets = None

        def get_wave(self) -> list:
            return self.wave

        def get_onsets(self) -> list:
            r"""
            Onsets are stored in a sorted list, elements are onset times in sample number.
            Note that simultaneous onsets are not merged.
            """
            # cache onsets
            if self._onsets is None:
                onsets = []
                for label in self.labels:
                    # Interval "label" has its 0th data being the start time(sample) of the note
                    onsets.append(label[0])
                onsets.sort()
                # cache onsets for repeated read
                self._onsets = onsets
            return self._onsets

        def get_sr(self) -> int:
            r"""
            This always returns 44100
            """
            return MusicNet.SAMPLE_RATE

        def get_data(self) -> (list, list, int):
            r"""
            :return: (wave, onsets, sample_rate)
            """
            return self.get_wave(), self.get_onsets(), self.get_sr()

        def get_raw_labels(self):
            return self.labels


class Modal(object):
    r"""
    https://github.com/johnglover/modal
    A rather small and mixed dataset for onset detection.
    onset are represented in samples.
    """

    def __init__(self):
        self.file = h5py.File(MODAL_PATH, 'r+')
        self.keys = list(self.file.keys())

    def get_keys(self) -> list:
        return self.keys

    def get_piece(self, key) -> 'Modal.Piece':
        return Modal.Piece(self.file[key])

    class Piece(object):
        def __init__(self, data):
            self.data = data

        def get_attr(self, attr_key):
            return self.data.attrs.get(attr_key)

        def get_wave(self) -> list:
            return self.data[...]

        def get_onsets(self) -> list:
            return self.get_attr("onsets")

        def get_sr(self) -> int:
            return self.get_attr("sampling_rate")

        def get_data(self) -> (list, list, int):
            return self.get_wave(), self.get_onsets(), self.get_sr()


class BockSet(object):
    ANNOTATIONS_PATH = BOCK_SET_PATH + "annotations/all/"
    AUDIO_PATH = BOCK_SET_PATH + "audio/"
    SPLITS_PATH = BOCK_SET_PATH + "splits/"
    AUDIO_EXTENSION = ".flac"
    ONSETS_EXTENSION = ".onsets"

    def __init__(self, sr=44100):
        self.keys = []
        self.sr = sr
        for filename in os.listdir(BockSet.AUDIO_PATH):
            if filename.endswith(".flac"):
                self.keys.append(os.path.splitext(filename)[0])
        # all 8 splits as a list
        self.splits = []
        for filename in os.listdir(BockSet.SPLITS_PATH):
            with open(BockSet.SPLITS_PATH + filename) as file:
                # all keys in a file as a list
                elements = []
                for line in file:
                    if line.strip():
                        elements.append(line.strip())
                self.splits.append(elements)

    def get_all_keys(self):
        return self.keys
        pass

    def get_piece(self, key):
        # TODO cache object?
        return BockSet.Piece(key, sr=self.sr)

    def get_splits(self):
        r"""
        Splits used by S. Boeck
        Notice that the Modal dataset is excluded from all splits
        :return: a list of splits, which each is a list of keys
        """
        return self.splits

    def get_split(self, index):
        r"""
        Splits used by S. Boeck
        Notice that the Modal dataset is excluded from all splits
        -----
        :param index: must be in range(0, 8)
        :return: a list of keys in the split
        """
        return self.splits[index]

    def generate_splits(self, test_split_index):
        r"""
        :return: (training_set_keys, validation_set_keys, test_set_keys)
        """
        splits = self.splits.copy()
        # setup training set, validation set, test set
        test_set_keys = splits.pop(test_split_index)
        # set the next one as validation set
        validation_set_keys = splits.pop(test_split_index % len(splits))
        # flatten the training set
        training_set_keys = np.concatenate(splits)
        return training_set_keys, validation_set_keys, test_set_keys

    class Piece(object):
        onsets = []
        onset_in_seconds = []
        wave = []
        sr = 0

        def __init__(self, key, sr=None):
            self.key = key
            self.sr = sr
            # audio must be loaded first (onset conversion relies on sampling rate)
            self._load_audio()
            self._load_onsets()

        # audio
        def _load_audio(self):
            self.wave = []

        def _load_onsets(self):
            self.onsets_in_seconds = []
            with open(BockSet.ANNOTATIONS_PATH + self.key + BockSet.ONSETS_EXTENSION) as onset_file:
                for line in onset_file:
                    # skip empty lines
                    if line.strip():
                        self.onsets_in_seconds.append(float(line))
            self.onsets = np.multiply(self.onsets_in_seconds, self.sr).astype(int)

        def get_wave(self):
            self.wave, _ = librosa.load(BockSet.AUDIO_PATH + self.key + BockSet.AUDIO_EXTENSION, sr=self.sr)
            return self.wave

        def seconds2samples(self, seconds) -> int:
            return int(seconds * self.sr)

        def samples2seconds(self, samples) -> float:
            return float(samples) / self.sr

        def get_onsets(self):
            r"""
            :return: onsets in sample numbers, as numpy 1darray
            """
            return self.onsets

        def get_onsets_seconds(self):
            r"""
            :return: onsets in seconds
            """
            return self.onsets_in_seconds

        def get_sr(self):
            return self.sr

        def get_data(self):
            r"""

            :return: (wave, onset_in_samples, sampling_rate)
            """
            return self.get_wave(), self.get_onsets(), self.get_sr()


def test_MusicNet():
    music_net = MusicNet()
    keys = music_net.get_keys()
    example_key = keys[0]
    piece = music_net.get_piece(example_key)
    wave, onsets, sr = piece.get_data()
    utils.plot_wave(wave[0:30 * sr])
    utils.plot_music_net_roll(piece.get_raw_labels(), 44100)


def test_modal():
    modal = Modal()
    keys = modal.get_keys()
    example_key = keys[1]
    piece = modal.get_piece(example_key)
    wave, onsets, sr = piece.get_data()
    utils.plot_wave(wave[0:30 * sr])


def test_BockSet():
    bock = BockSet()
    piece = bock.get_piece(bock.get_split(0)[0])
    print("Length of audio:")
    print((len(piece.get_wave()) / float(piece.sr)), "sec")
    print(piece.get_sr())
    print(type(piece.get_onsets()))
    print(piece.get_onsets())
    keys = bock.get_all_keys()
    for key in keys:
        try:
            onsets = bock.get_piece(key).get_onsets_seconds()
            if len(onsets) == 0:
                print(f"Onset length = 0 for {key}")
        except FileNotFoundError:
            pass


def count_boeck_split_info():
    import datetime
    boeck_set = BockSet()
    train, valid, test = boeck_set.generate_splits(0)
    splits = {"training": train, "validation": valid, "test": test}
    # count time, onsets of each split
    for split_name, split in splits.items():
        total_time = 0.
        total_onsets = 0
        for key in split:
            piece = boeck_set.get_piece(key)
            onsets = piece.get_onsets_seconds()
            length = len(piece.get_wave()) / piece.get_sr()
            total_onsets += len(onsets)
            total_time += length
        print(f"Split info of {split_name}:")
        print(f"\tTotal time: {str(datetime.timedelta(seconds=total_time))}")
        print(f"\tTotal onset annotations: {total_onsets}")


if __name__ == '__main__':
    # test_BockSet()
    count_boeck_split_info()

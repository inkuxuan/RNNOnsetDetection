import numpy as np
import sys
import math


def merge_onsets(onsets, time_interval):
    r"""
    Merge onsets that are close to each other. This only preserve the first onset.
    For merging using arithmetic mean, refer to boeck's code

    -----
    :param onsets: list of onsets
    :param time_interval: threshold (inclusive) within which onsets are merged (only the first one is remained)
    :return: list of merged onsets
    """
    if len(onsets) < 1:
        return []
    onsets_original = onsets.copy()
    onsets_original.sort()
    # initialize with the first onset
    onsets = [onsets_original[0]]
    for i in range(1, len(onsets_original)):
        # insert one onset that is at least time_interval away from the last one
        if onsets_original[i] - onsets[len(onsets) - 1] > time_interval:
            onsets.append(onsets_original[i])
    return onsets


def onset_list_to_target(onsets, sample_per_frame, length, mode='single', key=None):
    r"""
    Convert an onset list to a target list for training

    -----
    :param onsets: List of onsets in sample number
    :param sample_per_frame: Number of samples per frame
    :param length: Length (in frames)
    :param mode: one of `single`, `continuous`, `precise`
    :param output: one of 'int', 'boolean'
    :return: np array of length `length`, each representing the onset strength
    """
    # TODO continuous and precise target conversion

    target = np.zeros(length, dtype=np.int)
    for onset in onsets:
        frame = int(math.floor(float(onset) / sample_per_frame))
        if mode == 'single':
            if frame >= length:
                frame = length-1
            # IndexError: out of bounds
            target[frame] = 1
    return target


def test():
    onsets = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
    m = merge_onsets(onsets, 2)
    print(m)
    merge_onsets(onsets, 2, in_place=True)
    print(onsets)


if __name__ == '__main__':
    test()

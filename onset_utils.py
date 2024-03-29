import numpy as np
import math

import scipy.ndimage
import scipy.signal


def combine_onsets(onsets, time_interval, key=None):
    r"""
    Merge onsets that are close to each other. This only preserve the first onset.

    -----
    :param key: for information when printing warning
    :param onsets: list of onsets
    :param time_interval: threshold (inclusive) within which onsets are merged (only the first one is remained)
    :return: list of merged onsets
    """
    if len(onsets) < 1:
        if key:
            print(f"[WARNING] Empty onset list! key={key}")
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


def combine_onsets_avg(onsets, time_interval):
    r"""
    Merge onsets that are close to each other replacing with arithmetic mean.

    -----
    :param key: for information when printing warning
    :param onsets: list of onsets
    :param time_interval: threshold (inclusive) within which onsets are merged
    :return: list of merged onsets
    """
    if time_interval == 0:
        return onsets
    onsets_original = onsets.copy()
    onsets_original.sort()
    # the final output(combined), initialize with the first onset
    onsets = []
    index = 0
    while index < len(onsets_original):
        first_index = index
        first = onsets_original[first_index]
        while index < len(onsets_original) and (onsets_original[index] - first <= time_interval):
            index += 1
        onsets.append(np.mean(onsets_original[first_index:index]))
    return onsets


def onset_list_to_target(onsets, sample_per_frame, length, delta, mode='linear'):
    r"""
    Convert an onset list to a target list for training

    -----
    :param delta: if mode set to continuous or precise, number of frames to allow non-zero values
    to appear around an onset, in each side
    :param onsets: List of onsets in sample number
    :param sample_per_frame: Number of samples per frame
    :param length: Length (in frames)
    :param mode: one of `single`, `linear`, `precise`
    :return: np array of length `length`, each representing the onset strength
    """
    # TODO continuous and precise target conversion

    fill_len = int(delta + 1)
    # rising does not include 1
    rising = np.linspace(0, 1, fill_len, endpoint=False)
    # falling includes 1
    falling = np.linspace(1, 0, fill_len, endpoint=False)

    target = np.zeros(length, dtype='float32')
    for onset in onsets:
        frame = int(math.floor(float(onset) / sample_per_frame))
        # IndexError: out of bounds
        if frame >= length:
            frame = length - 1
        if mode == 'single':
            target[frame] = 1
        if mode == 'linear':
            target[frame] = 1

            # rising edge
            n_out_of_range = np.max([fill_len - frame, 0])
            # actual applied length
            d = fill_len - n_out_of_range
            # applying maximum to avoid overwriting close onsets
            target[frame - d: frame] = np.maximum(rising[n_out_of_range:], target[frame - d:frame])

            # falling edge
            n_out_of_range = np.max([(frame + fill_len) - length + 1, 0])
            # actual applied length
            d = fill_len - n_out_of_range
            # applying maximum to avoid overwriting close onsets
            target[frame: frame + d] = np.maximum(falling[:d], target[frame: frame + d])
    return target


def peak_pick_dynamic(signal, lambda_=1.0, min_threshold=0.1, max_threshold=0.5, smooth_window=7):
    r"""
    Find peaks in the signal according to the paper
    Eyben "Universal onset detection with bidirectional long short-term memory neural networks" ISMIR 2010

    -----
    :param signal: input signal
    :param lambda_: lambda parameter
    :param min_threshold: minimum threshold
    :param max_threshold: maximum threshold
    :return: list of peaks
    """
    signal = np.array(signal)
    # smooth the signal using hamming window
    if smooth_window and smooth_window > 1:
        signal = np.convolve(signal, np.hamming(smooth_window), mode='same')
    threshold = np.median(signal)
    threshold = lambda_ * max(min_threshold, min(max_threshold, threshold))
    signal = signal * (signal > threshold)
    local_maxima = scipy.signal.argrelextrema(signal, np.greater)
    return local_maxima


def peak_pick_static(signal, threshold=0.35, smooth_window=5):
    """returns (peak_index, )"""
    signal = np.array(signal)
    # smooth the signal using hamming window
    if smooth_window and smooth_window > 1:
        signal = np.convolve(signal, np.hamming(smooth_window), mode='same')
    threshold = threshold
    signal = signal * (signal > threshold)
    local_maxima = scipy.signal.argrelextrema(signal, np.greater)
    return local_maxima


def test():
    onsets = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
    m = combine_onsets(onsets, 2)
    print(m)
    onsets = combine_onsets(onsets, 2)
    print(onsets)


if __name__ == '__main__':
    test()

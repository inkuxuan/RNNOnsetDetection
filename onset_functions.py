import numpy as np
import librosa

import datasets
import utils


def complex_domain_odf(stft, aggregate=np.mean, rectify=True):
    r"""
    Calculate onset and offset strength array from the stft frames
    This implementation is derived from Boeck's

    ------
    :param rectify: apply rectification that only preserve rising magnitude
    :param stft: stft frames, must having complex values in each element
    :param aggregate: aggregate function, default mean()
    :return: (onset_strength, offset_strength)
    """
    phase = np.arctan2(np.imag(stft), np.real(stft))
    magnitude = np.abs(stft)
    cd_result = np.zeros_like(stft)
    # expected X
    cd_target = np.zeros_like(phase)
    # assume constant phase change
    cd_target[:, 1:] = 2 * phase[:, 1:] - phase[:, :-1]
    # take cd_target(phase) and add magnitude
    # note that target[n] is the expected stft of stft[n+1]
    cd_target = magnitude * np.exp(1j * cd_target)
    # complex spectrogram
    # note that cd_target[0] == 0, so only [2:] is calculated
    cd_result[:, 2:] = stft[:, 2:] - cd_target[:, 1:-1]
    # rectify so that only onsets remain
    if rectify:
        cd_result[:, 1:] = cd_result[:, 1:] * (magnitude[:, 1:] > magnitude[:, :-1])
    # take norm and aggregate
    return aggregate(np.abs(cd_result), axis=0)


# noinspection DuplicatedCode
def _tone_frequencies(band_per_octave, f_min=27.5, f_max=17000, initial_freq=440):
    r"""
    Used to generate a frequency array include frequencies for each tone evenly placed in octaves.

    ------
    :param band_per_octave: the number of frequencies per octave
    :param f_min: minimum frequency, default A0 = 27.5
    :param f_max: maximum frequency, default 17000
    :param initial_freq: frequency to start with, default A4 = 440
    :return: Array of frequencies in Hertz
    """
    assert f_min > 0 and f_max > 0 and initial_freq > 0, "Illegal arguments"
    # a factor to be applied to an arbitrary frequency to raise or lower the tone
    tone_factor = 2.0 ** (1.0 / band_per_octave)
    freq = initial_freq
    frequencies = [freq]
    # from A4 go up until f_max
    # the while loop will give a final frequency above f_max which will be the stopband freq
    while freq <= f_max:
        freq *= tone_factor
        frequencies.append(freq)
    freq = initial_freq
    # from A4 go down until f_min
    while freq >= f_min:
        freq /= tone_factor
        frequencies.append(freq)

    frequencies.sort()
    return frequencies


def _triangular_filter(length, center, normalize=False):
    r"""
    Create a triangular filter.
    Note that the unit here are all frequency bins.

    ------
    :param length: number of frequency bins of the filter
    :param center: frequency bin# of the center frequency
    :param normalize: True meaning to normalize the filter's area to be 1. This is useful if one wants to normalize
        amplitudes in each bin.
    :return: a 1d-array representing a triangular filter,
        with the values being the "gain" of each frequency bin.

    """

    # the "gain" value in center frequency
    height = 2.0 / length if normalize else 1
    triangular_filter = np.empty(length)
    # rising
    # filter[0(inclusive : center(exclusive)] = (0(inclusive) ~ height(exclusive))
    triangular_filter[:center] = np.linspace(0, height, center, endpoint=False)
    # falling
    # filter[center(inclusive) : length(exclusive)] = (height(inclusive) ~ 0(exclusive))
    triangular_filter[center:] = np.linspace(height, 0, length - center, endpoint=False)
    return triangular_filter


def _tonal_filter_bank(
        n_fft_bins,
        sr,
        bands_per_octave=24,
        f_min=27.5,
        f_max=17000,
        normalized_filters=False):
    r"""
    Create a filterbank for spectrogram matrix calculation, filters evenly placed in equal temperament.

    ------
    :param n_fft_bins: the height of a stft
    :param sr: sample rate of the audio (used to cut out unnecessarily high frequencies)
    :param bands_per_octave: number of filters per octave, default 24 (one per quarter-tone)
    :param f_min: minimum frequency
    :param f_max: maximum frequency
    :param normalized_filters: True: to normalize the filter area to be 1
    :return: a filterbank in numpy 2d-array
    """
    # cut off f_max above fft range
    if f_max > sr / 2:
        f_max = sr / 2
    frequencies = _tone_frequencies(bands_per_octave, f_min, f_max)
    # frequency bin to frequency factor
    f_bin_factor = (sr / 2.0) / n_fft_bins
    # round target frequencies to frequency bins
    f_bins = np.round(np.asarray(frequencies) / f_bin_factor).astype(int)
    # discard any repeated bins
    f_bins = np.unique(f_bins)
    # cut out any bins that are above fft bins
    f_bins = [f for f in f_bins if f < n_fft_bins]
    # f_bins contains 2 extra stop band frequencies
    n_bands = len(f_bins) - 2
    assert n_bands >= 3, "Too few filters in the filterbank. Check frequency range."
    # create filter bank from each center frequency
    filterbank = np.zeros([n_bands, n_fft_bins], dtype=float)
    for i in range(n_bands):
        # the center freq of the next band is the stop band of the previous one
        start, center, stop = f_bins[i:i + 3]
        triangular_filter = _triangular_filter(stop - start, center - start, normalize=normalized_filters)
        filterbank[i, start:stop] = triangular_filter
    return filterbank


def tonal_spectrogram(stft, filter_bank=None, power_spectrum=False):
    r"""
    Calculate spectrogram based on a given filterbank
    :param stft: stft frames
    :param filter_bank: filterbank used to do the dot product
    :param power_spectrum: if set to True, a power spectrogram (stft**2) is used instead before applying filter bank
        , and finally taken decibel scale.
    :return: the spectrogram in 2d-array
    """
    spectrogram = abs(stft)
    if power_spectrum:
        spectrogram = spectrogram ** 2
    if filter_bank is not None:
        # n_frame = stft.shape[1]
        spectrogram = np.dot(filter_bank, spectrogram)
    if power_spectrum:
        spectrogram = librosa.core.power_to_db(spectrogram)
    return spectrogram


def super_flux_odf(
        stft,
        sr,
        n_fft,
        hop_size,
        center=False,
        lag=1,
        filter_bank=None,
        bands_per_octave=24,
        f_min=27.5,
        f_max=17000,
        normalized_filters=False,
        max_size=3,
        power_spectrum=False
):
    r"""
    Compute a Super Flux onset strength function.

    This includes computing a tone based spectrogram using a filter bank,
    applying maximum filter on the reference spectrogram,
    taking the difference in the specified lag, and finally pad the onset envelop
    to compensate the stft hop-size effect.

    .. [#] Böck, Sebastian, and Gerhard Widmer.
           "Maximum filter vibrato suppression for onset detection."
           16th International Conference on Digital Audio Effects,
           Maynooth, Ireland. 2013.

    ------
    :param stft: the stft frames
    :param sr: sample rate of the audio (used to calculate filter banks)
    :param n_fft: the stft window size (for frame compensation)
    :param hop_size: the stft hop length (for frame compensation)
    :param center: to pad the odf by ``n_fft // (2 * hop_size)`` frames
    :param lag: to which the different is taken in the spectrogram
    :param filter_bank: used to calculate the spectrogram if specified
    :param bands_per_octave: number of filter bank sub-band per octave
    :param f_min: minimum frequency for the spectrogram
    :param f_max: maximum frequency for the spectrogram
    :param normalized_filters: set to True if you need to normalize the filters having the area to be 1
    :param max_size: the size of the maximum filter
    :param power_spectrum: if set to True, the stft frame is powered first,
        and the spectrogram is taken decibel rather than magnitude
    :return: (onset_strength, filter_bank, tonal_spectrogram), onset strength function in a 1d-array
    """
    if filter_bank is None:
        # create filter banks for the stft to spectrogram transfer
        filter_bank = _tonal_filter_bank(
            len(stft),
            sr=sr,
            bands_per_octave=bands_per_octave,
            f_min=f_min,
            f_max=f_max,
            normalized_filters=normalized_filters)
    spectrogram = tonal_spectrogram(stft, filter_bank=filter_bank, power_spectrum=power_spectrum)
    # librosa here use differential operation to compute onset envelope,
    # apply the maximum filter,
    # if center=True, it will pad the onset envelope
    # by ``n_fft // (2 * hop_size)`` frames
    # to compensate the sftf and hop-size effect (here default is set to False)
    onset_strength = librosa.onset.onset_strength(
        sr=sr, S=spectrogram, lag=lag, max_size=max_size, center=center,
        n_fft=n_fft, hop_length=hop_size)
    return onset_strength, filter_bank, spectrogram


def mel_filter_onset_strength(stft, sr, n_fft, hop_size):
    r"""
    This is equivalent to calling librosa.onset.onset_strength(), but can make use of the stft frames

    :param hop_size: stft hop length
    :param stft: stft frames
    :param sr: sample rate of audio
    :param n_fft: fft window size
    :return: onset strength as 1-d array
    """
    mel_basis = librosa.filters.mel(sr, n_fft)
    S = abs(stft) ** 2
    S = np.dot(mel_basis, S)
    S = librosa.power_to_db(S)
    mel_onset_strength = librosa.onset.onset_strength(S=S, sr=sr, n_fft=n_fft, hop_length=hop_size)
    return mel_onset_strength


def _main():
    import matplotlib.pyplot as plt
    import librosa.display as display

    n_fft = 2048
    hop_length = 512

    x, sr = librosa.load('music/Paganini.m4a', sr=22050)
    idx = slice(*list(librosa.time_to_samples([10, 30], sr=sr)))
    x = x[idx]
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    odf = complex_domain_odf(stft)
    sf, filter_bank, tonal_spec = super_flux_odf(stft, sr, n_fft, hop_length)

    mel_onset = mel_filter_onset_strength(stft, sr, n_fft, hop_length)

    plt.plot(mel_onset)
    plt.show()

    segment = slice(200, 400)
    fig, axs = plt.subplots(3)
    fig.tight_layout(pad=1.5)
    axs[0].plot(odf[segment])
    axs[0].set_title('Complex Domain')
    axs[1].plot(mel_onset[segment])
    axs[1].set_title('Mel-scale SF')
    axs[2].plot(sf[segment])
    axs[2].set_title('SuperFlux')
    plt.show()

    fig, ax = plt.subplots(3)
    fig.tight_layout(pad=4.0)
    fig.set_size_inches(22, 11)
    display.specshow(librosa.amplitude_to_db(abs(stft)), y_axis='hz', x_axis='time', sr=sr, ax=ax[0])
    ax[0].set_title('Mel-scale')
    display.specshow(librosa.amplitude_to_db(abs(stft)), y_axis='fft_note', x_axis='time', sr=sr, ax=ax[1])
    ax[1].set_title('Log-scale')
    display.specshow(librosa.amplitude_to_db(tonal_spec), y_axis='off', x_axis='time', ax=ax[2])
    ax[2].set_title('Tonal-scale')
    plt.show()

    fft_freqs = librosa.fft_frequencies(sr, n_fft)
    tone_freqs = _tone_frequencies(24, f_max=sr / 2.0)
    print(f"fft frequencies ({len(fft_freqs)})")
    print(fft_freqs)
    print(f"tonal spectrogram frequencies ({len(tone_freqs)})")
    print(tone_freqs)
    print(f"sub band count: {len(filter_bank)}")

    plt.plot(filter_bank[20:35, :])
    plt.show()


def test_cd():
    boeck = datasets.BockSet()
    splits = boeck.splits
    key = splits[0][0]
    piece = boeck.get_piece(key)
    wave, onsets, sr = piece.get_data()
    # wave, sr = librosa.load("music/yoake.wav", sr=44100)

    stft = librosa.stft(wave, n_fft=2048, hop_length=441, center=False)
    cd = complex_domain_odf(stft, rectify=False)
    rcd = complex_domain_odf(stft, rectify=True)
    onset_frames = librosa.samples_to_frames(onsets, 441, n_fft=2048)
    utils.plot_odf(cd[200:400], title="CD")
    utils.plot_odf(rcd[200:400], title="RCD")


if __name__ == '__main__':
    test_cd()

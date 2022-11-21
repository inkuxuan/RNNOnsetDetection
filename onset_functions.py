from concurrent.futures import ThreadPoolExecutor

import numpy as np
import librosa
from scipy.ndimage import maximum_filter1d
from scipy.ndimage import uniform_filter1d

import boeck.onset_evaluation
import datasets
import utils
from boeck.onset_evaluation import Counter


def complex_domain_odf(stft, aggregate=np.mean, rectify=True, log=True):
    r"""
    Calculate onset and offset strength array from the stft frames

    ------
    :param rectify: apply rectification that only preserve rising magnitude
    :param stft: stft frames, must having complex values in each element
    :param aggregate: aggregate function, default mean()
    :param log: True to apply logarithm to the result
    :return: onset_strength
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
    # take norm
    cd_result = np.abs(cd_result)
    # take log
    if log:
        cd_result = np.log10(cd_result + 1)
    return aggregate(cd_result, axis=0)


def _wrap_to_pi(angle):
    """
    Wrap an angle the range [-π,π)

    """
    return np.mod(angle + np.pi, 2.0 * np.pi) - np.pi


def phase_deviation(stft, aggregate=np.mean, weighted=True):
    r"""
    (Weighted) Phase Deviation

    ------
    :param stft: stft frames, must having complex values in each element
    :param aggregate: aggregate function, default mean()
    :param weighted: if set to True, outputs Weighted Phase Deviation
    :return: onset_strength
    """
    phase = np.arctan2(np.imag(stft), np.real(stft))
    magnitude = np.abs(stft)
    pd = np.zeros_like(phase)
    # 2nd derivative of Phi (phase)
    pd[:, 2:] = phase[:, 2:] - 2 * phase[:, 1:-1] + phase[:, :-2]
    pd = _wrap_to_pi(pd)
    if weighted:
        pd = aggregate(np.abs(pd * magnitude), axis=0)
    else:
        pd = aggregate(np.abs(pd), axis=0)
    return pd

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
    "Maximum Filter Vibrato Suppression for Onset Detection"
    Sebastian Böck and Gerhard Widmer.
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), Maynooth, Ireland, September 2013.

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


def tonal_spectrogram(stft, filter_bank=None, power_spectrum=False, log=False):
    r"""
    Calculate spectrogram based on a given filterbank
    :param stft: stft frames
    :param filter_bank: filterbank used to do the dot product
    :param power_spectrum: if set to True, a power spectrogram (stft**2) is used instead before applying filter bank
        , and finally taken decibel scale.
    :param log: if set to True, the spectrogram is taken logarithmically
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
    if log:
        spectrogram = np.log10(spectrogram + 1)
    return spectrogram


def super_flux_odf(
        stft,
        sr,
        lag=2,
        filter_bank=None,
        bands_per_octave=24,
        f_min=27.5,
        f_max=17000,
        normalized_filters=False,
        max_size=3,
        power_spectrum=False,
        log=True,
        keep_dims=True,
        aggregate=np.mean
):
    r"""
    Compute a SuperFlux onset strength function.

    This includes computing a western scale based spectrogram using a filter bank,
    applying maximum filter on the reference spectrogram,
    taking the difference in the specified lag, and finally pad the onset envelop
    to compensate the stft hop-size effect.

    .. [#] Böck, Sebastian, and Gerhard Widmer.
           "Maximum filter vibrato suppression for onset detection."
           16th International Conference on Digital Audio Effects,
           Maynooth, Ireland. 2013.

    ------
    :param stft: the stft frames with shape (frequency_bin, time_frame)
    :param sr: sample rate of the audio (used to calculate filter banks)
    :param lag: to which the different is taken in the spectrogram
    :param filter_bank: used to calculate the spectrogram if specified
    :param bands_per_octave: number of filter bank sub-band per octave
    :param f_min: minimum frequency for the spectrogram
    :param f_max: maximum frequency for the spectrogram
    :param normalized_filters: set to True if you need to normalize the filters having the area to be 1
    :param max_size: the size of the maximum filter
    :param power_spectrum: if set to True, the stft frame is powered first,
        and the spectrogram is taken decibel rather than magnitude
    :param log: if set to True, the spectrogram is taken logarithmically
    :param keep_dims: if set to True, the returned odf keeps the same length (in time) as the input stft
    :param aggregate: the aggregation function to be applied to the final difference
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
    spectrogram = tonal_spectrogram(stft, filter_bank=filter_bank, power_spectrum=power_spectrum, log=log)

    # apply maximum filter on the reference spectrogram
    if max_size > 1:
        # apply max filter
        from scipy.ndimage.filters import maximum_filter
        size = (max_size, 1)
        ref_spectrogram = maximum_filter(spectrogram, size=size)
    else:
        ref_spectrogram = spectrogram

    # calculate the difference between the reference spectrogram and the original spectrogram
    if keep_dims:
        diff = np.zeros_like(spectrogram)
        # the first frame has to be zero in order to keep the same dimensions as the input spectrogram
        diff[:, lag:] = spectrogram[:, lag:] - ref_spectrogram[:, :-lag]
    else:
        diff = spectrogram[:, lag:] - ref_spectrogram[:, :-lag]

    # keep only the positive differences
    diff = np.maximum(diff, 0)
    onset_strength = aggregate(diff, axis=0)

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
    sf, filter_bank, tonal_spec = super_flux_odf(stft, sr, )

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
    sf, _, _ = super_flux_odf(stft, sr)
    onset_frames = librosa.samples_to_frames(onsets, 441, n_fft=2048)
    utils.plot_odf(cd[200:400], title="CD")
    utils.plot_odf(rcd[200:400], title="RCD")
    utils.plot_odf(sf[200:400], title="SuperFlux")


def sf_peak_picking(odf, sr, hop_length, pre_max=0.03, post_max=0.03,
                    pre_avg=0.1, post_avg=0.07, delta=0.1,
                    combine_width=0.03):
    # Implementation by Boeck https://github.com/CPJKU/SuperFlux/
    # Maximum Filter Vibrato Suppression for Onset Detection (Boeck et al., 2013)
    frame_rate = float(sr) / hop_length
    # convert timing information to frames
    pre_avg = int(round(frame_rate * pre_avg))
    pre_max = int(round(frame_rate * pre_max))
    post_max = int(round(frame_rate * post_max))
    post_avg = int(round(frame_rate * post_avg))
    # init detections
    detections = []
    # moving maximum
    max_length = pre_max + post_max + 1
    max_origin = int(np.floor((pre_max - post_max) / 2))
    mov_max = maximum_filter1d(odf, max_length,
                               mode='constant', origin=max_origin)
    # moving average
    avg_length = pre_avg + post_avg + 1
    avg_origin = int(np.floor((pre_avg - post_avg) / 2))
    mov_avg = uniform_filter1d(odf, avg_length,
                               mode='constant', origin=avg_origin)
    # detections are activation equal to the moving maximum
    detections = odf * (odf == mov_max)
    # detections must be greater or equal than the mov. average + threshold
    detections *= (detections >= mov_avg + delta)
    # convert detected onsets to a list of timestamps
    detections = np.nonzero(detections)[0].astype('float') / frame_rate
    # always use the first detection and all others if none was reported
    # within the last `combine` seconds
    if detections.size > 1:
        # filter all detections which occur within `combine` seconds
        detections = detections[1:][np.diff(detections) > combine_width]
    return detections


def test_superflux_f_score(n_fft=2048, hop_length=220.5, lag=2,
                           pre_max=0.03, post_max=0.03,
                           pre_avg=0.1, post_avg=0.07, delta=0.1,
                           combine_width=0.03, offset=0.0,
                           eval_window=0.25):
    r"""
    test f-score on the boeck dataset using SuperFlux ODF
    """
    # load data
    boeck_set = datasets.BockSet()
    splits = boeck_set.get_splits()
    # load all splits into a single list
    keys = splits[0] + splits[1] + splits[2] + splits[3] + splits[4] + splits[5] + splits[6] + splits[7]
    count = Counter()
    for key in keys:
        onsets, detections = superflux_detections_by_key(key, n_fft=n_fft, hop_length=hop_length, lag=lag,
                                                         pre_max=pre_max, post_max=post_max,
                                                         pre_avg=pre_avg, post_avg=post_avg, delta=delta,
                                                         combine_width=combine_width, offset=offset)
        # merge close onsets
        onsets = boeck.onset_evaluation.combine_events(onsets, combine_width)
        # calculate f-score
        count += boeck.onset_evaluation.count_errors(detections, onsets, eval_window)
    print(f"delta: {delta}. f-score: {count.fmeasure}")
    print(f"\tTP: {count.tp}, FP:{count.fp}, FN:{count.fn}")


def superflux_detections_by_key(key, n_fft=2048, hop_length=220.5, lag=2,
                                pre_max=0.03, post_max=0.03,
                                pre_avg=0.1, post_avg=0.07, delta=.1,
                                combine_width=0.03, offset=0.):
    boeck_set = datasets.BockSet()
    track = boeck_set.get_piece(key)
    wave = track.get_wave()
    # wave = utils.normalize_wav(wave, 'float')
    ground_truth = track.get_onsets_seconds()
    sr = track.get_sr()
    stft_f = stft(wave, n_fft=n_fft, hop_length=hop_length)
    sf, _, _ = super_flux_odf(stft_f, sr, lag=lag, aggregate=np.mean)
    # peak-picking
    detections = sf_peak_picking(sf, sr, hop_length, pre_max=pre_max, post_max=post_max, pre_avg=pre_avg,
                                 post_avg=post_avg, delta=delta, combine_width=combine_width)
    if offset != 0:
        detections += offset
    return ground_truth, detections


def stft(wave, n_fft=2048, hop_length=220.5, online=False, include_dc=False, use_incorrect_bins=False):
    r"""
    This is implemented to ensure the same performance as

    .. [#] Böck, Sebastian, and Gerhard Widmer.
           "Maximum filter vibrato suppression for onset detection."
           16th International Conference on Digital Audio Effects,
           Maynooth, Ireland. 2013.

    Refer to https://github.com/CPJKU/SuperFlux/blob/master/SuperFlux.py

    :param wave: input wave
    :param n_fft: the length of the FFT window (hanning window)
    :param hop_length: the hop size, can be a float in order to match an integer fps
    :param online: online mode (but see comments for explanations on why this is NOT online)
    :param include_dc: whether to include the DC component in the STFT (to include frequency bin #0)
    :param use_incorrect_bins: See comments on `stft_frames[:, frame] = fft.rfft(signal)[1:]`
    :return: the STFT frames with shape (frequency_bin, frame)
    """
    import scipy.fft as fft
    n_samples = len(wave)
    n_frames = int(np.ceil(len(wave) / hop_length))
    # NOTE that here n_fft_bins is not `n_fft // 2 + 1`
    # because only this many bins are used in the referred paper (i.e. not include the DC bin)
    # However see below the `FFT` part for a difference in the implementation
    n_fft_bins = int(n_fft / 2)
    # If include DC bin
    if include_dc:
        n_fft_bins += 1
    window = np.hanning(n_fft)
    # in case the wave is in int
    try:
        int_max = np.iinfo(wave.dtype).max
        window /= int_max
    except ValueError:
        pass
    # init frames
    stft_frames = np.empty((n_fft_bins, n_frames), dtype='complex')
    for frame in range(n_frames):
        # seek position
        if online:
            # move 1 hop_length forward, and then step one n_fft back
            # this result in the frame position being at the start of the "new" contents of the frame
            # NOTE that this does NOT actually make future information unavailable (by stepping one hop_length forward)
            # Therefore the referred paper is not accurate on the description of "online"
            seek = int((frame + 1) * hop_length - n_fft)
        else:
            # move 1 hop_length back, and then step half n_fft back
            # so that the frame position represents the center of a frame
            seek = int(frame * hop_length - n_fft / 2)
        if seek >= n_samples:
            # EOF
            break
        elif seek + n_fft > n_samples:
            # signal too short for the frame, append zeros
            zeros = np.zeros(seek + n_fft - n_samples)
            signal = wave[seek:]
            signal = np.append(signal, zeros)
        elif seek < 0:
            # start before the signal, pad zeros
            zeros = np.zeros(-seek)
            signal = wave[:seek + n_fft]
            signal = np.append(zeros, signal)
        else:
            signal = wave[seek:seek + n_fft]
        # apply window
        signal = signal * window
        # FFT
        if include_dc:
            # bin [0:n_fft_bins+1]
            stft_frames[:, frame] = fft.rfft(signal)[:]
        else:
            # NOTE that here the referred paper used `fft.fft(signal)[:n_fft_bins]` which includes the DC bin
            # which is not correct
            # Here we use the RFFT and discard the DC bin
            # [1:n_fft_bins+1]
            if not use_incorrect_bins:
                stft_frames[:, frame] = fft.rfft(signal)[1:]
            else:
                stft_frames[:, frame] = fft.fft(signal)[:n_fft_bins]
    return stft_frames


def test_superflux_detections(delta=0.006):
    boeck = datasets.BockSet()
    splits = boeck.splits
    key = splits[0][0]
    piece = boeck.get_piece(key)
    wave, onsets, sr = piece.get_data()
    onsets = piece.get_onsets_seconds()

    n_fft = 2048
    hop_length = 441
    fps = sr / hop_length

    # stft_frames = librosa.stft(wave, n_fft=n_fft, hop_length=hop_length, center=False)
    stft_frames = stft(wave, n_fft=n_fft, hop_length=hop_length)
    sf, _, _ = super_flux_odf(stft_frames, sr, aggregate=np.mean)
    onset_frames = np.multiply(onsets, fps)
    utils.plot_odf(sf, title="SuperFlux(Ground-truth)", onsets=onset_frames)
    _, detections = superflux_detections_by_key(key, delta=delta)
    detections_frames = np.multiply(detections, fps)
    utils.plot_odf(sf, title="SuperFlux(Detections)", onsets=detections_frames)


def test_stft():
    sr = 44100
    hop = 220
    n_fft = 2048
    wave, _ = librosa.load("./demo_audio/original.wav", sr=sr)
    stft_frames = stft(wave, n_fft=n_fft, hop_length=hop, include_dc=True)
    print(stft_frames.shape)
    librosa_stft = librosa.stft(wave, n_fft=n_fft, hop_length=hop, center=True)
    print(librosa_stft.shape)
    from matplotlib import pyplot as plt
    mid_point = stft_frames.shape[1] // 2
    plt.figure()
    plt.plot(stft_frames[:200, mid_point])
    plt.show()
    plt.figure()
    plt.plot(librosa_stft[:200, mid_point])
    plt.show()
    print(np.allclose(stft_frames, librosa_stft))


def get_example_superflux_output():
    boeck = datasets.BockSet()
    splits = boeck.splits
    key = splits[0][0]
    piece = boeck.get_piece(key)
    wave, onsets, sr = piece.get_data()

    n_fft = 2048
    hop_length = 220.5
    stft_frames = stft(wave, n_fft=n_fft, hop_length=hop_length)
    sf, _, _ = super_flux_odf(stft_frames, sr)
    return sf


def get_example_wpd_output():
    boeck = datasets.BockSet()
    splits = boeck.splits
    key = splits[0][0]
    piece = boeck.get_piece(key)
    wave, onsets, sr = piece.get_data()

    n_fft = 2048
    hop_length = 220.5
    stft_frames = stft(wave, n_fft=n_fft, hop_length=hop_length)
    wpd = phase_deviation(stft_frames)
    return wpd


def plot_example_superflux():
    sf = get_example_superflux_output()
    utils.plot_odf(sf)


def plot_example_wpd():
    sf = get_example_wpd_output()
    utils.plot_odf(sf)


def test_save_superflux_output():
    sf = get_example_superflux_output()
    np.save("./superflux_array", sf)
    print("test file saved")


def test_load_superflux_output():
    sf_computed = get_example_superflux_output()
    sf_loaded = np.load("./superflux_array.npy")
    print(f"allclose: {np.allclose(sf_computed, sf_loaded)}")
    print(f"equal: {np.equal(sf_computed, sf_loaded)}")
    print("test end")


if __name__ == '__main__':
    plot_example_wpd()
    # test_stft()
    # test_superflux_detections(delta=0.2)
    # test_load_superflux_output()
    # test_superflux_f_score(delta=0.006, hop_length=441, lag=1)
    # plot_example_superflux()
    # with ThreadPoolExecutor(max_workers=6) as executor:
    #     for delta in np.arange(0.002, 0.022, 0.002):
    #         future = executor.submit(test_superflux_f_score, delta=delta, hop_length=441, lag=1)

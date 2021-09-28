import random

import boeck.onset_program
import datasets
import networks
import onsets
import odf
import utils

from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import librosa
import numpy as np
from torch import nn
import torch.optim
import boeck.onset_evaluation
from boeck.onset_evaluation import Counter

N_FFT = 2048
HOP_SIZE = 441
ONSET_DELTA = 0.030
# Number of cores in CPU, used to multithreading in Onset Detection Function calculations
CPU_CORES = 6


def get_sets(splits, test_split_index) -> tuple[list, list, list]:
    splits = splits.copy()
    # setup training set, validation set, test set
    test_set_keys = splits.pop(test_split_index)
    # set the first one as validation set
    validation_set_keys = splits.pop(0)
    # flatten the training set
    training_set_keys = []
    for sublist in splits:
        for item in sublist:
            training_set_keys.append(item)
    return training_set_keys, validation_set_keys, test_set_keys


def get_features(wave, features=None, n_fft=2048, hop_size=440, sr=44100, center=False) -> np.ndarray:
    r"""
    :param wave: audio wave
    :param features: list of features (str), in ['cd', 'rcd', 'superflux']
    :return: ndarray, axis 0 being feature type, axis 1 being the length of a sequence
    """
    if features is None:
        features = ['complex_domain', 'super_flux']
    stft = librosa.stft(wave, n_fft=n_fft, hop_length=hop_size, center=center)
    f = []
    for feature in features:
        if feature in ['complex_domain', 'cd']:
            onset_strength = odf.complex_domain_odf(stft)
            f.append(onset_strength)
        elif feature in ['rectified_complex_domain', 'rcd']:
            onset_strength = odf.complex_domain_odf(stft, rectify=True)
            f.append(onset_strength)
        elif feature in ['super_flux', 'superflux']:
            onset_strength, _, _ = odf.super_flux_odf(stft, sr, n_fft, hop_size, center)
            f.append(onset_strength)
    return np.asarray(f)


def validation(boeck_set, validation_set, model, features) -> Counter:
    with torch.no_grad():
        for key in validation_set:
            predictions = get_model_output(boeck_set, key, model, features)
            # TODO peak picking for prediction function
            # boeck.onset_evaluation.count_errors()
    return None


def get_model_output(boeck_set, key, model, features, verbose=False):
    r"""
    NOTICE RETURN VALUE

    :return: (prediction, hidden_state)
    """
    with torch.no_grad():
        t0 = time.perf_counter()
        length, odfs, target, onset_seconds, audio_len = prepare_data(boeck_set, features, key)
        t1 = time.perf_counter()
        if verbose:
            print(f"Data preparation ({t1 - t0})")
        input_array = np.expand_dims(odfs, axis=0)
        input_array = torch.from_numpy(input_array).to(device=networks.device).type(model.dtype)
        t2 = time.perf_counter()
        if verbose:
            print(f"Tensor conversion ({t2 - t1})")
        out = model(input_array)
        t3 = time.perf_counter()
        if verbose:
            print(f"Prediction ({t3 - t2})")
            print(f"Audio {audio_len:.1f}sec, speed {audio_len / (t3 - t0):.1f}x")
    return out


class BoeckDataLoader(object):

    def __init__(self, boeck_set, training_set_keys, batch_size, shuffle=True, features=None):
        self.boeck_set = boeck_set
        self.training_set = training_set_keys
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.features = features

    def _shuffle(self):
        random.shuffle(self.training_set)

    def generate_data(self):
        r"""
        Generator that yields (input_ndarray, target_ndarray, total_audio_length)
        Note that total length is in seconds.
        Input array shape: (batch_size, max_length, n_features)
        Target shape: (batch_size, max_length)
        max_length is the maximum length of all sequences in a batch
        """
        if self.shuffle:
            self._shuffle()
        index = 0
        data_size = len(self.training_set)
        while index < data_size:
            # current batch size
            b_size = self.batch_size
            if index + b_size >= data_size:
                b_size = data_size - index - 1
            end_index = index + b_size
            keys = self.training_set[index:end_index]

            # concurrent, prepare ODFs for every piece in the batch
            with ThreadPoolExecutor(max_workers=max([CPU_CORES, b_size])) as executor:
                results = executor.map(prepare_data, repeat(self.boeck_set), repeat(self.features), keys)
                results = list(zip(*results))
                lengths = results[0]
                odfs_list = results[1]
                target_list = results[2]
                audio_in_seconds = results[4]
            maxlen=np.max(lengths)
            total_audio_length = np.sum(audio_in_seconds)

            # resize (pad zeros) ndarrays to form a batch
            # input shape: (batch_size, max_length_frames, features)
            input_np = np.array([np.resize(odfs, (maxlen, odfs.shape[1])) for odfs in odfs_list])
            # target shape: (batch_size, max_length_frames)
            target_np = np.array([np.resize(target, maxlen) for target in target_list])
            yield input_np, target_np, total_audio_length
            index = end_index


def train_network(boeck_set, training_set, validation_set, model, loss_fn, optimizer,
                  features=None):
    r"""


    ------
    :param features: Feature functions
    :return: Counter in validation set
    """

    # epoch
    continue_epoch = True
    i = 0
    while continue_epoch:
        i += 1
        for key_index in range(len(training_set)):
            t0 = time.perf_counter()

            key = training_set[key_index]
            # preparing data
            length, odfs, target, _, audio_len = prepare_data(boeck_set, features, key)
            # training model
            # input shape: (batch_size, length, n_feature)
            # output shape: (batch_size, length)
            input_array = np.expand_dims(odfs, axis=0)

            input_array = torch.from_numpy(input_array).to(device=networks.device).type(model.dtype)
            target = torch.from_numpy(target).to(device=networks.device).type(model.dtype)

            prediction, _ = model(input_array)

            loss = loss_fn(prediction.squeeze(), target)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.perf_counter()
            print(
                f"{key} ({audio_len:.1f}s), speed {(audio_len / (t1 - t0)):.1f}x, {key_index + 1}/{len(training_set)}, "
                f"epoch {i}")
            print(f"loss: {loss.item():>7f}")
            if loss.item() <= 0.05 or i >= 10:
                continue_epoch = False

    # validation
    # TODO validation
    # validation(boeck_set, validation_set, model, features)
    continue_epoch = False


def prepare_data(boeck_set, features, key, normalize=True):
    r"""
    Notice that odfs is in shape (length, n_features), target is in shape (length)

    :return: length(frames), onset_detection_function_values, target_values, onset_list_in_seconds, audio_length_sec
    """
    piece = boeck_set.get_piece(key)
    wave, onsets_list, sr = piece.get_data()
    onset_seconds = boeck.onset_evaluation.combine_events(piece.get_onsets_seconds(), ONSET_DELTA)
    onsets_list = np.asarray(onset_seconds) * sr
    odfs = get_features(wave, n_fft=N_FFT, hop_size=HOP_SIZE, sr=sr, center=False, features=features)
    length = odfs.shape[1]
    # Normalize the odfs so that they range from 0 to 1
    if normalize:
        for i in range(len(odfs)):
            max_value = np.max(odfs[i, :])
            odfs[i, :] = odfs[i, :] / max_value

    target = onsets.onset_list_to_target(onsets_list, HOP_SIZE, length, ONSET_DELTA * sr / HOP_SIZE, key=key,
                                         mode='linear')

    # arrange dimensions so that the model can accept (shape==[seq_len, n_feature])
    odfs = odfs.T

    return length, odfs, target, onset_seconds, len(wave) / sr


def train_and_test(boeck_set, splits, test_split_index,
                   num_layers=2, num_layer_unit=4,
                   strategy='softmax',
                   learning_rate=0.1,
                   features=None):
    if features is None:
        features = ['complex_domain', 'super_flux']
    counter = Counter()
    training_set, validation_set, test_set = get_sets(splits, test_split_index)
    # TODO implementing other strategies
    assert strategy == 'softmax'
    # input: superflux + cd (for now)
    rnn = networks.SingleOutRNN(len(features), num_layer_unit, num_layers).to(networks.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    train_network(boeck_set, training_set, validation_set, rnn, loss_fn, optimizer,
                  features=features)
    # TODO test network
    return rnn


def main():
    boeck_set = datasets.BockSet()
    splits = boeck_set.splits
    for i in range(len(splits)):
        # test once treating each split as a test set
        train_and_test(boeck_set, splits, i)


def test_network_training():
    print(f"device: {networks.device}")
    boeck_set = datasets.BockSet()
    splits = boeck_set.splits
    features = ['super_flux', 'complex_domain']
    model = networks.SingleOutRNN(len(features), 4, 2, sigmoid=False, bidirectional=True).to(networks.device)

    frame_rate = 44100 / HOP_SIZE
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)

    output, hidden = get_model_output(boeck_set, splits[0][0], model, features)
    output = torch.sigmoid(output)
    output = output.squeeze()
    output = output.T
    output = output.cpu()
    utils.plot_odf(output, title="Network(RAW)")

    train_network(boeck_set, splits[0], None, model, loss_fn, optimizer,
                  features=features)

    # piece = boeck_set.get_piece(splits[0][0])
    # wave, _, sr = piece.get_data()
    # stft = librosa.stft(wave, N_FFT, HOP_SIZE, center=False)
    # super_flux, _, _ = odf.super_flux_odf(stft, sr, N_FFT, HOP_SIZE)

    output, _ = get_model_output(boeck_set, splits[0][0], model, features)
    output = torch.sigmoid(output)
    output = output.squeeze()
    output = output.T
    output = output.cpu()
    utils.plot_odf(output, title="Network(Trained)")

    length, odfs, target, onset_seconds, audio_len = prepare_data(boeck_set, features, splits[0][0])
    utils.plot_odf(odfs, title="SuperFlux")
    utils.plot_odf(target, title="Target")

    now = datetime.now()
    dstr = now.strftime("%Y-%m-%d %H%M%S")
    torch.save(model, 'model-' + dstr + '.pt')

    # print(hidden)


def test_output():
    boeck_set = datasets.BockSet()
    t0 = time.perf_counter()
    model = networks.SingleOutRNN(2, 4, 2).to(networks.device)
    t1 = time.perf_counter()
    print(f"Network initialized ({t1 - t0})")
    prediction, _ = get_model_output(boeck_set, boeck_set.get_split(0)[0], model,
                                     ['complex_domain', 'super_flux'],
                                     verbose=True)
    print(prediction.shape)
    print(next(model.parameters()).is_cuda)
    print(prediction.is_cuda)


def test_prepare_data():
    boeck_set = datasets.BockSet()
    splits = boeck_set.splits
    features = ['super_flux', 'complex_domain']
    for i in range(len(splits[0])):
        t0 = time.perf_counter()
        length, odfs, target, onset_seconds, audio_len = prepare_data(boeck_set, features, splits[0][i])
        t1 = time.perf_counter()
        print(f"{audio_len:.2f}s, elapsed {t1 - t0:.2f}, {audio_len / (t1 - t0):.1f}x speed")


def test_data_loader():
    batch_size = 8
    boeck = datasets.BockSet()
    training_set, _, _ = get_sets(boeck.splits, 0)
    loader = BoeckDataLoader(boeck, training_set, batch_size)
    t0 = time.perf_counter()
    v_in, target, total_len = next(loader.generate_data())
    t1 = time.perf_counter()
    print("Input array shape:", v_in.shape)
    print("Target array shape: ", target.shape)
    print("Time elapsed: ", (t1-t0), ", Speed: ", total_len/(t1-t0), "x")
    assert v_in.shape[0] == batch_size
    assert target.shape[0] == batch_size


if __name__ == '__main__':
    test_data_loader()

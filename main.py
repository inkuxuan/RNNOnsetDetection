import librosa
import torch.optim

import boeck.onset_program
import datasets
import networks
import onsets
import odf
import numpy as np
import boeck.onset_evaluation
import utils
from boeck.onset_evaluation import Counter
from torch import nn

import time

N_FFT = 2048
HOP_SIZE = 441
ONSET_DELTA = 0.030


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
    :return: ndarray, axis 0 being feature type, axis 1 being the length of a sequence
    """
    if features is None:
        features = ['complex_domain', 'super_flux']
    stft = librosa.stft(wave, n_fft=n_fft, hop_length=hop_size, center=center)
    f = []
    for feature in features:
        if feature == 'complex_domain':
            onset_strength = odf.complex_domain_odf(stft)
            f.append(onset_strength)
        elif feature == 'super_flux':
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
        input_array = torch.from_numpy(input_array).to(device=networks.device).float()
        t2 = time.perf_counter()
        if verbose:
            print(f"Tensor conversion ({t2 - t1})")
        out = model(input_array)
        t3 = time.perf_counter()
        if verbose:
            print(f"Prediction ({t3 - t2})")
            print(f"Audio {audio_len:.1f}sec, speed {audio_len / (t3 - t0):.1f}x")
    return out


def train_network(boeck_set, training_set, validation_set, model, loss_fn, optimizer,
                  features=None):
    r"""


    ------
    :param features: Feature functions
    :return: Counter in validation set
    """
    if features is None:
        features = ['complex_domain', 'super_flux']

    # epoch
    continue_epoch = True
    for i in range(4):
        for key_index in range(len(training_set)):
            t0 = time.perf_counter()

            key = training_set[key_index]
            # preparing data
            length, odfs, target, _, audio_len = prepare_data(boeck_set, features, key)
            # training model
            # input shape: (batch_size, length, n_feature)
            # output shape: (batch_size, length, target_dimension)
            input_array = np.expand_dims(odfs, axis=0)

            input_array = torch.from_numpy(input_array).to(device=networks.device).float()
            target = torch.from_numpy(target).to(device=networks.device).long()

            prediction, _ = model(input_array)

            loss = loss_fn(prediction.squeeze(), target)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.perf_counter()
            print(
                f"{key} ({audio_len:.1f}s), speed {(audio_len / (t1 - t0)):.1f}x, {key_index + 1}/{len(training_set)}, "
                f"epoch {i + 1}")
            print(f"loss: {loss.item():>7f}")

    # validation
    # TODO validation
    # validation(boeck_set, validation_set, model, features)
    continue_epoch = False


def prepare_data(boeck_set, features, key):
    r"""
    Notice that odfs is in shape (length, n_features), target is in shape (length)

    :return: length, onset_detection_function_values, target_values, onset_list_in_seconds, audio_length_sec
    """
    piece = boeck_set.get_piece(key)
    wave, onsets_list, sr = piece.get_data()
    onset_seconds = boeck.onset_evaluation.combine_events(piece.get_onsets_seconds(), ONSET_DELTA)
    onsets_list = np.asarray(onset_seconds) * sr
    odfs = get_features(wave, n_fft=N_FFT, hop_size=HOP_SIZE, sr=sr, center=False, features=features)
    length = odfs.shape[1]
    target = onsets.onset_list_to_target(onsets_list, HOP_SIZE, length, key=key)

    # arrange dimensions so that the model can accept
    odfs = odfs.T

    return length, odfs, target, onset_seconds, len(wave) / sr


def train_and_test(boeck_set, splits, test_split_index,
                   num_layers=2, num_layer_unit=4,
                   strategy='softmax',
                   learning_rate=0.05,
                   features=None):
    if features is None:
        features = ['complex_domain', 'super_flux']
    counter = Counter()
    training_set, validation_set, test_set = get_sets(splits, test_split_index)
    # TODO implementing other strategies
    assert strategy == 'softmax'
    # input: superflux + cd (for now)
    rnn = networks.OneHotRNN(len(features), num_layer_unit, num_layers).to(networks.device)
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
    features = ['super_flux']
    model = networks.OneHotRNN(len(features), 4, 2).to(networks.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    output, hidden = get_model_output(boeck_set, splits[0][0], model, features)
    output = output.squeeze()
    output = torch.log_softmax(output, 1)
    output = output.T
    output = output.cpu()
    utils.plot_odf(output[1, :], title="Network(RAW)")

    train_network(boeck_set, splits[1], None, model, loss_fn, optimizer,
                  features=features)

    piece = boeck_set.get_piece(splits[0][0])
    wave, _, sr = piece.get_data()
    stft = librosa.stft(wave, N_FFT, HOP_SIZE, center=False)
    super_flux, _, _ = odf.super_flux_odf(stft, sr, N_FFT, HOP_SIZE)

    output, _ = get_model_output(boeck_set, splits[0][0], model, features)
    output = output.squeeze()
    output = torch.log_softmax(output, 1)
    output = output.T
    output = output.cpu()
    utils.plot_odf(output[1, :], title="Network(Trained)")
    utils.plot_odf(super_flux, title="SuperFlux")

    print(hidden)


def test_output():
    boeck_set = datasets.BockSet()
    t0 = time.perf_counter()
    model = networks.OneHotRNN(2, 4, 2).to(networks.device)
    t1 = time.perf_counter()
    print(f"Network initialized ({t1 - t0})")
    prediction, _ = get_model_output(boeck_set, boeck_set.get_split(0)[0], model,
                                     ['complex_domain', 'super_flux'],
                                     verbose=True)
    print(prediction.shape)
    print(next(model.parameters()).is_cuda)
    print(prediction.is_cuda)


if __name__ == '__main__':
    test_network_training()

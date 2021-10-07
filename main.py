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
import multiprocessing
import gc

import librosa
import numpy as np
from torch import nn
import torch.optim
import boeck.onset_evaluation
from boeck.onset_evaluation import Counter

# ---------------CONFIGURATIONS----------------

# Number of cores in CPU
CPU_CORES = multiprocessing.cpu_count()
# If enabled, all the odf preprocessing result will be cached in MEMORY
# Recommend turning on if more than 8GB RAM is AVAILABLE
# If not enough RAM is available, the program will likely just crash
CACHED_PREPROCESSING = True

# ---------------META PARAMETERS----------------

N_FFT = 2048
HOP_SIZE = 441
ONSET_DELTA = 0.030
TARGET_MODE = 'linear'
FEATURES = ['rcd', 'superflux']


def get_features(wave, features=None, n_fft=2048, hop_size=440, sr=44100, center=False) -> np.ndarray:
    r"""
    :param wave: audio wave
    :param features: list of features (str), in ['cd', 'rcd', 'superflux']
    :return: ndarray, axis 0 being feature type, axis 1 being the length of a sequence
    """
    if features is None:
        features = ['rcd', 'superflux']
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


odfs_cache = {}
features_cache = {}


def prepare_data(boeck_set, features, key):
    r"""
    Notice that odfs is in shape (length, n_features), target is in shape (length)

    :return: length(frames), onset_detection_function_values, target_values, onset_list_in_seconds, audio_length_sec
    """
    piece = boeck_set.get_piece(key)
    wave, onsets_list, sr = piece.get_data()
    onset_seconds = boeck.onset_evaluation.combine_events(piece.get_onsets_seconds(), ONSET_DELTA)
    onsets_list = np.asarray(onset_seconds) * sr
    # load from cache if available (check if feature type matches)
    if CACHED_PREPROCESSING and (odfs_cache.get(key) is not None) and (features_cache.get(key) == features):
        # retrieve from cached (already transposed and normalised)
        odfs = odfs_cache[key]
    else:
        odfs = get_features(wave, n_fft=N_FFT, hop_size=HOP_SIZE, sr=sr, center=False, features=features)
        # arrange dimensions so that the model can accept (shape==[seq_len, n_feature])
        odfs = odfs.T
        # Normalize the odfs along each feature so that they range from 0 to 1
        for i in range(odfs.shape[1]):
            max_value = np.max(odfs[:, i])
            odfs[:, i] = odfs[:, i] / max_value
        if CACHED_PREPROCESSING:
            # save transposed odfs
            odfs_cache[key] = odfs
            features_cache[key] = features
            # Prevent from memory overflowing
            gc.collect()
    length = odfs.shape[0]
    target = onsets.onset_list_to_target(onsets_list, HOP_SIZE, length, ONSET_DELTA * sr / HOP_SIZE, key=key,
                                         mode=TARGET_MODE)
    return length, odfs, target, onset_seconds, len(wave) / sr


class BoeckDataLoader(object):
    r"""
    DataLoader (Not PyTorch DataLoader) for Boeck Dataset.
    Shuffle and minibatch implemented, concurrent ODF calculation within a batch.
    No. of threads correspond to CPU core count.
    """

    def __init__(self, boeck_set, training_set_keys, batch_size, shuffle=True, features=None, target_mode='single'):
        r"""

        :param batch_size: size of each minibatch, default is the available core count of CPU
        :param shuffle: bool, whether to shuffle for each epoch
        """
        self.boeck_set = boeck_set
        self.training_set = training_set_keys
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.features = features
        self.target_mode = target_mode

    def _shuffle(self):
        random.shuffle(self.training_set)

    def generate_data(self):
        r"""
        Generator that yields (input_ndarray, target_ndarray, total_audio_length)
        Note that total length is in seconds.
        Input array shape: (batch_size, max_length, n_features)
        Target shape: (batch_size, max_length)
        max_length is the maximum length (frames) of all sequences in a batch
        """
        if self.shuffle:
            self._shuffle()
        index = 0
        data_size = len(self.training_set)
        while index < data_size:
            # current batch size
            b_size = self.batch_size
            if index + b_size > data_size:
                b_size = data_size - index
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
            maxlen = np.max(lengths)
            total_audio_length = np.sum(audio_in_seconds)

            # resize (pad zeros) ndarrays to form a batch
            # input shape: (batch_size, max_length_frames, features)
            input_np = np.array([np.resize(odfs, (maxlen, odfs.shape[1])) for odfs in odfs_list])
            # target shape: (batch_size, max_length_frames)
            target_np = np.array([np.resize(target, maxlen) for target in target_list])
            yield input_np, target_np, total_audio_length
            index = end_index


class ModelManager(object):
    def __init__(self,
                 boeck_set: datasets.BockSet,
                 features=None,
                 num_layer_unit=4,
                 num_layers=2,
                 nonlinearity='tanh',
                 bidirectional=True,
                 loss_fn=None,
                 optimizer=None,
                 scheduler=None):
        r"""

        :param features: list, element in ['rcd', 'cd', 'superflux']; default ['rcd', 'superflux']
        :param num_layer_unit: Number of units in one hidden layer
        :param num_layers: Number of hidden layers
        :param loss_fn: default BCEWithLogitsLoss
        :param optimizer: default SGD
        :param scheduler: default milestone scheduler. Set to False to disable scheduler
        """
        self.boeck_set = boeck_set
        self.features = features
        if self.features is None:
            self.features = ['rcd', 'superflux']
        self.model = networks.SingleOutRNN(
            len(self.features),
            num_layer_unit,
            num_layers,
            nonlinearity=nonlinearity,
            bidirectional=bidirectional,
            sigmoid=False
        ).to(networks.device)
        self.loss_fn = loss_fn
        if self.loss_fn is None:
            self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.6)
        self.scheduler = scheduler
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 40, 50], gamma=0.15)

    def save(self, filename=None):
        if filename is None:
            now = datetime.now()
            dstr = now.strftime("%Y%m%d %H%M%S")
            filename = 'mdl_' + dstr
            filename += '_' + str(self.model.recurrent.nonlinearity)
            filename += '_' + str(self.model.num_layers) + 'x' + str(self.model.hidden_size)
            if self.model.recurrent.bidirectional:
                filename += '(bi)'
            filename += '.pt'
        torch.save(self.model, filename)

    def predict(self,
                key=None,
                wave=None,
                hop_size=441,
                n_fft=2048,
                sr=441000,
                verbose=False,
                sigmoid=True):
        r"""
        Specify either key or wave
        if wave is specified, hop_size, n_fft, sr is used

        :return: raw network output if sigmoid=False
        """
        with torch.no_grad():
            t0 = time.perf_counter()
            if key is not None:
                _, odfs, _, _, audio_len = prepare_data(self.boeck_set, self.features, key)
            elif wave is not None:
                odfs = get_features(wave, self.features, hop_size=hop_size, n_fft=n_fft, sr=sr)
                audio_len = len(wave) / float(sr)
            else:
                raise ValueError("either a key or a wave array should be specified")
            t1 = time.perf_counter()
            if verbose:
                print(f"Data preparation ({t1 - t0})")
            input_array = np.expand_dims(odfs, axis=0)
            input_array = torch.from_numpy(input_array).to(device=networks.device).type(self.model.dtype)
            t2 = time.perf_counter()
            if verbose:
                print(f"Tensor conversion ({t2 - t1})")
            out, _ = self.model(input_array)
            if sigmoid:
                out = torch.sigmoid(out)
            t3 = time.perf_counter()
            if verbose:
                print(f"Prediction ({t3 - t2})")
                print(f"Audio {audio_len:.1f}sec, speed {audio_len / (t3 - t0):.1f}x")
        return out

    def generate_splits(self, test_split_index) -> tuple[list, list, list]:
        r"""
        :return: (training_set_keys, validation_set_keys, test_set_keys)
        """
        splits = self.boeck_set.splits.copy()
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

    def train_on_split(self,
                       training_keys: list,
                       validation_keys: list,
                       batch_size=CPU_CORES,
                       verbose=False,
                       debug_max_epoch_count=None,
                       debug_min_loss=None):
        r"""
        Train the network on a specific training set and validation set

        -----
        :param batch_size: How many pieces at a time to train the network.
        Training process uses concurrency for both pre-processing and training. This is default the cores count of CPU.
        :param verbose: bool, Print debug outputs
        :return:
        """
        loader = BoeckDataLoader(self.boeck_set, training_keys, batch_size, features=self.features)
        continue_epoch = True
        # TODO validation
        epochs_without_improvement = 0
        epoch = 0
        loss_record = []
        while continue_epoch:
            epoch += 1
            print(f"===EPOCH {epoch}===")
            if self.scheduler:
                print(f"lr={self.scheduler.get_lr()}")
            if verbose:
                last_time = time.perf_counter()
            for v_in, target, total_len in loader.generate_data():
                v_in = torch.from_numpy(v_in).to(device=networks.device).type(self.model.dtype)
                target = torch.from_numpy(target).to(device=networks.device).type(self.model.dtype)
                prediction, _ = self.model(v_in)
                loss = self.loss_fn(prediction.squeeze(dim=2), target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_record.append(loss.item())

                if verbose:
                    now = time.perf_counter()
                    # noinspection PyUnboundLocalVariable
                    print(f"{now - last_time:.1f}s {total_len / (now - last_time):.1f}x in epoch {epoch}")
                    print(f"loss: {loss.item():>7f}")
                    last_time = now
                if ((debug_min_loss and loss.item() <= debug_min_loss)
                        or (debug_max_epoch_count and epoch >= debug_max_epoch_count)):
                    continue_epoch = False
            if self.scheduler:
                self.scheduler.step()
        return loss_record

    def train_and_test(self, test_split_index, verbose=False, **kwargs):
        training_keys, validation_keys, test_keys = self.generate_splits(test_split_index)
        return self.train_on_split(training_keys, validation_keys, verbose=verbose, **kwargs)
        # TODO test (evaluation codes)


def test_network_training():
    print(f"device: {networks.device}")
    print(f"cpu {CPU_CORES} cores")

    # configure
    boeck_set = datasets.BockSet()
    trainer = ModelManager(boeck_set, bidirectional=True)
    splits = trainer.boeck_set.splits
    # initialize
    trainer.features = FEATURES
    trainer.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3))
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.5)
    trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        trainer.optimizer,
        milestones=[30, 230, 430], gamma=0.2)

    # training
    loss = trainer.train_and_test(0, debug_max_epoch_count=700, verbose=True, batch_size=64)
    # loss = trainer.train_on_split(splits[1], [], debug_max_epoch_count=430, verbose=True, batch_size=64)

    # testing
    test_key = splits[0][0]
    piece = boeck_set.get_piece(test_key)
    onset_seconds = boeck.onset_evaluation.combine_events(piece.get_onsets_seconds(), ONSET_DELTA)
    onsets_list = np.asarray(onset_seconds) * 44100
    onsets_list = onsets_list / HOP_SIZE

    output = trainer.predict(key=test_key)
    output = output.squeeze()
    output = output.T
    output = output.cpu()
    utils.plot_odf(output, title="Network(Trained)", onsets=onsets_list)

    length, odfs, target, onset_seconds, audio_len = prepare_data(boeck_set, trainer.features, test_key)
    utils.plot_odf(odfs, title="SuperFlux")
    utils.plot_odf(target, title="Target")
    utils.plot_loss(loss)

    trainer.save()

    # print(hidden)


def test_output():
    boeck_set = datasets.BockSet()

    t0 = time.perf_counter()
    mgr = ModelManager(boeck_set)
    t1 = time.perf_counter()
    print(f"Network initialized ({t1 - t0})")
    prediction = mgr.predict(boeck_set.splits[0][0], verbose=True)

    print(prediction.shape)
    print("CUDA ", prediction.is_cuda)


def test_prepare_data():
    boeck_set = datasets.BockSet()
    splits = boeck_set.splits
    for i in range(len(splits[0])):
        t0 = time.perf_counter()
        length, odfs, target, onset_seconds, audio_len = prepare_data(boeck_set, None, splits[0][i])
        t1 = time.perf_counter()
        print(f"{audio_len:.2f}s, elapsed {t1 - t0:.2f}, {audio_len / (t1 - t0):.1f}x speed")


def test_data_loader():
    batch_size = 8
    boeck = datasets.BockSet()
    mgr = ModelManager(boeck)
    training_set, _, _ = mgr.generate_splits(2)
    loader = BoeckDataLoader(boeck, training_set, batch_size)
    t0 = time.perf_counter()
    audio_length = 0
    v_in = None
    target = None
    for v_in, target, total_len in loader.generate_data():
        audio_length += total_len
        break
    t1 = time.perf_counter()
    print("Input array shape:", v_in.shape)
    print("Target array shape: ", target.shape)
    print("Time elapsed: ", (t1 - t0), ", Speed: ", audio_length / (t1 - t0), "x")
    odf = v_in[0, :, :]
    target = target[0, :]
    if odf.shape[0] > 500:
        odf = odf[:500, :]
        target = target[:500]
    print(f"shape plot: {odf.shape}, {target.shape}")
    utils.plot_odf(odf, onset_target=target)


if __name__ == '__main__':
    test_network_training()

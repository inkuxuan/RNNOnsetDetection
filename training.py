import random
import sys

import boeck.onset_program
import datasets
import networks
import onset_utils
import onset_functions
import utils

from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import multiprocessing
import gc
import pathlib

import librosa
import scipy.signal
import numpy as np
from torch import nn
import torch.optim
import torch.backends.cudnn
import boeck.onset_evaluation
from boeck.onset_evaluation import Counter

# ---------------REPRODUCIBILITY---------------

RANDOM_SEED = 0x39C5BB


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(RANDOM_SEED)

# ---------------CONFIGURATIONS---------------

# Number of cores in CPU
CPU_CORES = multiprocessing.cpu_count()
# If enabled, all the odf preprocessing result will be cached in MEMORY
# Recommend turning on if more than 8GB RAM is AVAILABLE
# If not enough RAM is available, the program will likely just crash
CACHED_PREPROCESSING = True

# ---------------META PARAMETERS----------------

SAMPLING_RATE = 44100
N_FFT = 2048
HOP_SIZE = 441
# lag setting for SuperFlux
SPF_LAG = 1
# onsets within what time is merged into one
ONSET_DELTA = 0.030
# in ['single', 'linear']
TARGET_MODE = 'single'
# how many non-zero target frames should be generated around an onset (each side)
# only used when TARGET_MODE == 'linear'
TARGET_DELTA = 0
DEFAULT_FEATURES = ['rcd', 'superflux']
# in ['static', 'dynamic']
PEAK_PICK_MODE = 'static'

# Keep this True unless you need to run a previously saved model
DEBUG_TAKE_LOG = True

# function reference for onset combiner
# in [onsets.combine_onsets_avg, onsets.merge_onsets, None]
# stands for combining onsets by average, by the first onset, and no combination
COMBINE_ONSETS = onset_utils.combine_onsets_avg
COMBINE_ONSETS_DETECTION = True

now = datetime.now()
dstr = now.strftime("%Y%m%d %H%M%S")
RUN_DIR = f'run {dstr}/'

PEAK_PICK_FUNCTION = onset_utils.peak_pick_static if PEAK_PICK_MODE == 'static' else onset_utils.peak_pick_dynamic


# ---------------HELPER FUNCTIONS----------------


def get_features(wave, features=None, n_fft=N_FFT, hop_size=HOP_SIZE, sr=SAMPLING_RATE) -> np.ndarray:
    r"""
    :param sr: sampling rate
    :param hop_size: FFT hop length
    :param n_fft: FFT window size
    :param wave: audio wave
    :param features: list of features (str), in ['cd', 'rcd', 'superflux']
    :return: ndarray, axis 0 being feature type, axis 1 being the length of a sequence
    """
    if features is None:
        features = DEFAULT_FEATURES
    stft = onset_functions.stft(wave, n_fft=n_fft, hop_length=hop_size, online=False)
    f = []
    for feature in features:
        if feature in ['complex_domain', 'cd']:
            onset_strength = onset_functions.complex_domain_odf(stft, rectify=False, log=DEBUG_TAKE_LOG)
            f.append(onset_strength)
        elif feature in ['rectified_complex_domain', 'rcd']:
            onset_strength = onset_functions.complex_domain_odf(stft, rectify=True, log=DEBUG_TAKE_LOG)
            f.append(onset_strength)
        elif feature in ['super_flux', 'superflux']:
            onset_strength, _, _ = onset_functions.super_flux_odf(stft, sr, lag=SPF_LAG, log=DEBUG_TAKE_LOG)
            f.append(onset_strength)
    return np.asarray(f)


odfs_cache = {}
features_cache = {}


def normalize_odf(odfs):
    r"""
    This normalize the second dimension in an nd-array to range [0,1]
    """
    for i in range(odfs.shape[1]):
        max_value = np.max(odfs[:, i])
        odfs[:, i] = odfs[:, i] / max_value
    return odfs


def prepare_data(boeck_set, features, key, normalize=True):
    r"""
    Notice that odfs is in shape (length, n_features), target is in shape (length)

    :return: length(frames), onset_detection_function_values, target_values, onset_list_in_seconds, audio_length_sec
    """
    piece = boeck_set.get_piece(key)
    wave, onsets_list, sr = piece.get_data()
    if normalize:
        wave = utils.normalize_wav(wave, type='float')
    if COMBINE_ONSETS:
        onsets_list = COMBINE_ONSETS(piece.get_onsets_seconds(), ONSET_DELTA)
    # convert from second to sample
    onsets_list = np.asarray(onsets_list) * sr
    # load from cache if available (check if feature type matches)
    if CACHED_PREPROCESSING and (odfs_cache.get(key) is not None) and (features_cache.get(key) == features):
        # retrieve from cached (already transposed and normalised)
        odfs = odfs_cache[key]
    else:
        odfs = get_features(wave, n_fft=N_FFT, hop_size=HOP_SIZE, sr=sr, features=features)
        # arrange dimensions so that the model can accept (shape==[seq_len, n_feature])
        odfs = odfs.T
        # Normalize the odfs along each feature so that they range from 0 to 1
        odfs = normalize_odf(odfs)
        if CACHED_PREPROCESSING:
            # save transposed odfs
            odfs_cache[key] = odfs
            features_cache[key] = features
            # Prevent from memory overflowing
            gc.collect()
    length = odfs.shape[0]
    target = onset_utils.onset_list_to_target(onsets_list, HOP_SIZE, length, TARGET_DELTA, key=key,
                                              mode=TARGET_MODE)
    return length, odfs, target, onsets_list, len(wave) / sr


# ---------------CLASSES----------------

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
        if self.shuffle and self.batch_size < len(self.training_set):
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


class ModelConfig(object):
    def __init__(self,
                 features=None,
                 num_layer_unit=4,
                 num_layers=2,
                 nonlinearity='tanh',
                 bidirectional=True):
        r"""
        :param features: list, element in ['rcd', 'cd', 'superflux']; default ['rcd', 'superflux']
        :param num_layer_unit: Number of units in one hidden layer
        :param num_layers: Number of hidden layers
        :param nonlinearity: either 'tanh' or 'relu'
        :param bidirectional: True for BiRNN
        """
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity
        self.num_layers = num_layers
        self.num_layer_unit = num_layer_unit
        self.features = features
        if self.features is None:
            self.features = DEFAULT_FEATURES


class TrainingConfig(object):
    def __init__(self,
                 weight=1,
                 optimizer_constructor=None,
                 optimizer_args=None,
                 optimizer_kwargs=None,
                 scheduler_constructor=None,
                 scheduler_args=None,
                 scheduler_kwargs=None,
                 epoch=5000,
                 batch_size=64,
                 loss_fn=None,
                 early_stop_patience=50):
        r"""

        :param weight: weight for positive class
        :param optimizer_constructor: optimizer class, default Adam with a lr of 1e-3.
        if supplied, optimizer_params must be provided along
        :param optimizer_args: arguements for the optimizer constructor
        :param optimizer_kwargs: keywords arguements for the optimizer constructor
        :param epoch: maximum no of epochs
        :param batch_size: minibatch size
        :param loss_fn: loss function (instance)
        :param early_stop_patience: epochs to wait until early stop. If set to 0, early stopping won't be performed
        """
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_args = scheduler_args
        self.scheduler_constructor = scheduler_constructor
        self.early_stop_patience = early_stop_patience
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.epoch = epoch
        self.optimizer = optimizer_constructor
        self.optimizer_args = optimizer_args
        self.optimizer_kwargs = optimizer_kwargs
        self.weight = weight
        if self.loss_fn is None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.weight))
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam
            self.optimizer_args = None
            self.optimizer_kwargs = {'lr': 1e-3}


# noinspection PyUnusedLocal
class ModelManager(object):

    def __init__(self,
                 boeck_set: datasets.BockSet,
                 model_config: ModelConfig,
                 training_config: TrainingConfig,
                 load_file=None):
        r"""
        :param load_file: if a path is specified, the model is loaded from a file
        """
        self.model_config = model_config
        self.training_config = training_config
        self.boeck_set = boeck_set
        self.features = self.model_config.features
        if load_file:
            self.model = torch.load(load_file, map_location=networks.device)
        else:
            self.model = networks.SingleOutRNN(
                len(self.model_config.features),
                self.model_config.num_layer_unit,
                self.model_config.num_layers,
                nonlinearity=self.model_config.nonlinearity,
                bidirectional=self.model_config.bidirectional,
                sigmoid=False
            ).to(networks.device)
        self.loss_fn = self.training_config.loss_fn
        if self.loss_fn is None:
            self.loss_fn = nn.BCEWithLogitsLoss()
        if self.training_config.optimizer_args and self.training_config.optimizer_kwargs:
            self.optimizer = self.training_config.optimizer(self.model.parameters(),
                                                            *self.training_config.optimizer_args,
                                                            **self.training_config.optimizer_kwargs)
        elif self.training_config.optimizer_kwargs:
            self.optimizer = self.training_config.optimizer(self.model.parameters(),
                                                            **self.training_config.optimizer_kwargs)
        else:
            self.optimizer = self.training_config.optimizer(self.model.parameters())
        self.scheduler = None
        if self.training_config.scheduler_constructor:
            if self.training_config.scheduler_args and self.training_config.scheduler_kwargs:
                self.scheduler = self.training_config.scheduler_constructor(self.optimizer,
                                                                            *self.training_config.scheduler_args,
                                                                            **self.training_config.scheduler_kwargs)
            elif self.training_config.scheduler_kwargs:
                self.scheduler = self.training_config.scheduler_constructor(self.optimizer,
                                                                            **self.training_config.scheduler_kwargs)

    def initialize_model(self):
        self.model.init_normal()

    def save_model(self, filename=None):
        if not self.model:
            raise RuntimeError("Model not initialized")
        if filename is None:
            now = datetime.now()
            dstr = now.strftime("%Y%m%d %H%M%S")
            filename = 'mdl_' + dstr
            filename += '_' + str(self.model.recurrent.nonlinearity)
            filename += '_' + str(self.model.num_layers) + 'x' + str(self.model.hidden_size)
            if self.model.recurrent.bidirectional:
                filename += '(bi)'
            filename += '.pt'
        pathlib.Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
        torch.save(self.model, RUN_DIR + filename)

    def save_cp(self, filename):
        if filename == None:
            now = datetime.now()
            dstr = now.strftime("%Y%m%d %H%M%S")
            filename = 'mdl_' + dstr
            filename += '_' + str(self.model.recurrent.nonlinearity)
            filename += '_' + str(self.model.num_layers) + 'x' + str(self.model.hidden_size)
            if self.model.recurrent.bidirectional:
                filename += '(bi)'
            filename += '.cp.pt'
        checkpoint = {
            'model': self.model.state_dict() if self.model else None,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }
        pathlib.Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, RUN_DIR + filename)

    def load(self, path):
        r"""
        This WILL NOT re-initialize the optimizer and the scheduler.
        Use the constructor instead if wish to train the network.
        """
        self.model = torch.load(path, map_location=networks.device)

    def load_model_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def load_cp(self, filename):
        checkpoint = torch.load(filename)
        if self.model:
            self.model.load_state_dict(checkpoint['model'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def predict(self,
                key=None,
                wave=None,
                hop_size=HOP_SIZE,
                n_fft=N_FFT,
                sr=SAMPLING_RATE,
                verbose=False,
                sigmoid=True,
                **kwargs):
        r"""
        either `key` or `wave` is needed
        if wave is specified, hop_size, n_fft, sr is used

        :return: raw network output if sigmoid=False
        """
        self.model.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            if key is not None:
                _, odfs, _, _, audio_len = prepare_data(self.boeck_set, self.features, key)
            elif wave is not None:
                wave = utils.normalize_wav(wave)
                odfs = get_features(wave, features=self.features,
                                    hop_size=hop_size, n_fft=n_fft, sr=sr, center=False)
                audio_len = len(wave) / float(sr)
                odfs = odfs.T
                odfs = normalize_odf(odfs)
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

    def predict_onsets_offline(self, key=None, wave=None, lambda_=0.35, smooth=0.05, **kwargs):
        r"""
        either `key` or `wave` is needed

        :param wave: a wave audio in 1-d array
        :param key: key of the track
        :param lambda_: see onset_utils.peak_pick()
        :param smooth: smooth window size in seconds
        :return: onsets in seconds, not combined
        """
        out = self.predict(key=key, wave=wave, **kwargs)
        out = out.squeeze()
        out = out.cpu()
        if TARGET_MODE != 'precise':
            peaks = PEAK_PICK_FUNCTION(out, lambda_=lambda_,
                                       smooth_window=int(smooth * SAMPLING_RATE / HOP_SIZE))
            frame_rate = SAMPLING_RATE / HOP_SIZE
            onsets_sec = np.array(peaks[0]) / frame_rate
            return onsets_sec
        # TODO precise mode decoding
        return None

    def test_learning_rates(self, lrs=None):
        r"""
        test the learning rate on the validation set

        ----

        :returns: list of lrs, list of losses
        """
        if lrs is None:
            lrs = np.concatenate([np.arange(1e-5, 1e-3, 1e-5), np.arange(1e-3, 1e-1, 1e-3)])
        training_set_keys, valid_set_kets, _ = self.boeck_set.generate_splits(0)
        self.training_config.batch_size = 256
        loader = BoeckDataLoader(self.boeck_set, training_set_keys,
                                 self.training_config.batch_size, features=self.features)
        v_loader = BoeckDataLoader(self.boeck_set, valid_set_kets,
                                   len(valid_set_kets), features=self.features)
        loss_list = []
        self.model.train()
        for i in range(len(lrs)):
            lr = lrs[i]
            print(f"Testing learning rate {lrs[i]}")
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
            loss = self._fit(i + 1, loader, True)
            valid_loss = self._validate_loss(i + 1, v_loader)
            print(f"Validation loss: {valid_loss:.4f}")
            loss_list.append(valid_loss)
        return lrs, loss_list

    def train_on_split(self,
                       training_keys,
                       validation_keys,
                       verbose=True,
                       save_checkpoint=None,
                       min_epoch=1000,
                       callback=None,
                       start_epoch=0,
                       **kwargs):
        r"""
        Train the network on a specific training set and validation set

        -----
        :param save_checkpoint: if being True or a str, constantly saves the model to a file
        :param validation_keys: list of keys in validation set
        :param training_keys: list of keys in training set
        Training process uses concurrency for both pre-processing and training. This is default the cores count of CPU.
        :param verbose: bool, Print debug outputs
        :param min_epoch: int, minimum epochs to train (for early stop)
        :param callback: function, callback function to be called after each epoch
            callback(model_manager=, epoch=, valid_loss_record=, best_state_dict=, best_valid_loss=, **kwargs)
        :return: dict containing {loss_record, valid_loss_record, epoch_now, lr, best_state_dict}
        """
        loader = BoeckDataLoader(self.boeck_set, training_keys, self.training_config.batch_size, features=self.features)
        valid_loader = BoeckDataLoader(self.boeck_set, validation_keys, len(validation_keys), features=self.features,
                                       shuffle=False)
        continue_epoch = True
        epoch_now = start_epoch
        loss_record = []
        valid_loss_record = []
        best_valid_loss = float('inf')
        best_state_dict = self.model.state_dict()
        early_stopping = False
        if self.training_config.early_stop_patience > 0:
            early_stopping = utils.EarlyStopping(patience=self.training_config.early_stop_patience,
                                                 min_epoch=min_epoch)
        while continue_epoch:
            epoch_now += 1
            if self.training_config.epoch and epoch_now >= self.training_config.epoch:
                continue_epoch = False
            print(f"===EPOCH {epoch_now}===")
            print(f"lr={self.optimizer.param_groups[0]['lr']}")
            # FIT THE MODEL
            avg_loss = self._fit(epoch_now, loader, verbose)
            loss_record.append(avg_loss)
            # VALIDATION
            valid_loss = self._validate_loss(epoch_now, valid_loader)
            valid_loss_record.append(valid_loss)
            # LEARNING RATE UPDATE
            if self.scheduler:
                self.scheduler.step()
            # Checkpoint saving
            if valid_loss < best_valid_loss:
                print("Current Best")
                best_valid_loss = valid_loss
                best_state_dict = self.model.state_dict()
                if save_checkpoint:
                    if isinstance(save_checkpoint, str):
                        self.save_cp(save_checkpoint)
                    else:
                        self.save_cp("checkpoint.pt")
            # CALLBACK
            if callback:
                callback(model_manager=self, epoch=epoch_now, valid_loss_record=valid_loss_record, **kwargs)
            # EARLY STOPPING
            if early_stopping:
                early_stopping(valid_loss)
            if early_stopping.early_stop:
                continue_epoch = False
        return {'loss_record': loss_record,
                'valid_loss_record': valid_loss_record,
                'epoch_now': epoch_now,
                'lr': self.optimizer.param_groups[0]['lr'],
                'best_state_dict': best_state_dict,
                'best_valid_loss': best_valid_loss}

    # noinspection DuplicatedCode
    def _fit(self, epoch_now, loader, verbose):
        self.model.train()
        if verbose:
            last_time = time.perf_counter()
        track_count = 0
        avg_loss = 0.
        for v_in, target, total_len in loader.generate_data():
            v_in = torch.from_numpy(v_in).to(device=networks.device).type(self.model.dtype)
            target = torch.from_numpy(target).to(device=networks.device).type(self.model.dtype)
            prediction, _ = self.model(v_in)
            loss = self.loss_fn(prediction.squeeze(dim=2), target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # loss calculation (weighted)
            track_count += len(v_in)
            avg_loss += loss.item() * len(v_in)

            if verbose:
                now = time.perf_counter()
                # noinspection PyUnboundLocalVariable
                print(f"{now - last_time:.1f}s {total_len / (now - last_time):.1f}x in epoch {epoch_now}")
                print(f"loss: {loss.item():>7f}")
                last_time = now
        return avg_loss / track_count

    # noinspection DuplicatedCode
    def _validate_loss(self, epoch_now, loader):
        self.model.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            track_count = 0
            avg_loss = 0.
            for v_in, target, total_len in loader.generate_data():
                v_in = torch.from_numpy(v_in).to(device=networks.device).type(self.model.dtype)
                target = torch.from_numpy(target).to(device=networks.device).type(self.model.dtype)
                prediction, _ = self.model(v_in)
                loss = self.loss_fn(prediction.squeeze(dim=2), target)
                # loss calculation (weighted)
                track_count += len(v_in)
                avg_loss += loss.item() * len(v_in)

                t1 = time.perf_counter()
                # noinspection PyUnboundLocalVariable
                print(f"Validate: {t1 - t0:.1f}s {total_len / (t1 - t0):.1f}x in epoch {epoch_now}")
                print(f"Valid_loss: {loss.item():>7f}")
            return avg_loss / track_count

    def train_only(self, test_split_index, verbose=True, **kwargs):
        r"""
        :return: dict containing {loss_record, valid_loss_record, epoch_now, lr, best_state_dict}
        """
        training_keys, validation_keys, test_keys = self.boeck_set.generate_splits(test_split_index)
        info = self.train_on_split(training_keys, validation_keys, verbose=verbose, **kwargs)
        return info

    def train_and_test(self, test_split_index, verbose=True, lambda_=.35, window=0.025, delay=0,
                       **kwargs):
        r"""

        :param verbose: whether to print a message every minibatch
        :param test_split_index: the split index of test set
        :param window: evaluation window radius, in seconds
        :param delay: time delay for detections, in seconds
        :return: counter, training_info (dict)
        """
        info = self.train_only(test_split_index, verbose=verbose, **kwargs)
        counter = self.test_only(test_split_index, lambda_=lambda_, window=window, delay=delay, **kwargs)
        return counter, info

    def test_only(self, test_split_index,
                  combine_output_onsets=COMBINE_ONSETS_DETECTION, lambda_=.35, window=0.025, delay=0, **kwargs):
        r"""

        :param window: evaluation window radius, in seconds
        :param delay: time delay for detections, in seconds
        :return: counter
        """
        test_keys = self.boeck_set.get_split(test_split_index)
        return self.test_on_keys(test_keys, combine_output_onsets=combine_output_onsets,
                                 lambda_=lambda_, window=window, delay=delay, **kwargs)

    def test_on_keys(self, keys, combine_output_onsets=COMBINE_ONSETS_DETECTION,
                     lambda_=.35, window=0.025, delay=0,
                     concurrent=True, **kwargs):
        r"""

        :param keys: keys of the dataset to test
        :param window: evaluation window radius, in seconds
        :param delay: time delay for detections, in seconds
        :return: counter
        """
        count = Counter()
        if concurrent:
            with ThreadPoolExecutor(max_workers=max([CPU_CORES, len(keys)])) as executor:
                futures = []
                for key in keys:
                    future = executor.submit(self.test_on_key, key, lambda_=lambda_, window=window,
                                             combine_output_onsets=combine_output_onsets,
                                             delay=delay, **kwargs)
                    futures.append(future)
                for future in futures:
                    count += future.result()
        else:
            for key in keys:
                count1 = self.test_on_key(key,
                                          combine_output_onsets=combine_output_onsets, lambda_=lambda_, window=window,
                                          delay=delay, **kwargs)
                count += count1
        return count

    def test_on_key(self, key, combine_output_onsets=COMBINE_ONSETS_DETECTION,
                    lambda_=.35, height=0.3, window=0.025, delay=0, **kwargs):
        ground_truth = self.boeck_set.get_piece(key).get_onsets_seconds()
        detections = self.predict_onsets_offline(key=key, lambda_=lambda_, **kwargs)
        if COMBINE_ONSETS:
            if combine_output_onsets:
                detections = COMBINE_ONSETS(detections, ONSET_DELTA)
            ground_truth = COMBINE_ONSETS(ground_truth, ONSET_DELTA)
        count = boeck.onset_evaluation.count_errors(detections, ground_truth, window, delay=delay)
        return count

    def speed_test_on_keys(self, keys,
                           combine_output_onsets=COMBINE_ONSETS_DETECTION, lambda_=.35, smooth=0.05):
        r"""

        :return: processing speed (times of processing audio)
        """
        # calculate total length
        total_len = 0.
        for key in keys:
            samples = len(self.boeck_set.get_piece(key).get_wave())
            length = samples / SAMPLING_RATE
            total_len += length
        t1 = time.perf_counter()
        # speed test
        for key in keys:
            detections = self.predict_onsets_offline(key=key, lambda_=lambda_, smooth=smooth)
            if combine_output_onsets:
                detections = COMBINE_ONSETS(detections, ONSET_DELTA)
        t2 = time.perf_counter()
        return total_len / (t2 - t1)


class TrainingTask(object):
    def __init__(self,
                 trainer: ModelManager,
                 lambdas=None):
        r"""
        :parameter lambdas: list of lambdas for evaluation of onsets. See onset_utils.peak_pick().
        Set to None for dynamic lambdas
        """
        self.dynamic_lambdas = False
        if lambdas is None:
            if PEAK_PICK_MODE == 'static':
                lambdas = np.arange(0.05, 0.6, 0.05)
                self.dynamic_lambdas = False
                self.lambda_step = 0.05
            else:
                lambdas = np.arange(0.1, 5, 0.5)
                self.dynamic_lambdas = True
                self.lambda_step = 0.5
                self.min_lambda_step = 0.05
        self.lambdas = lambdas
        now = datetime.now()
        dstr = now.strftime("%Y%m%d %H%M%S")
        self.report_dir = RUN_DIR + f'report {dstr}/'
        # os.makedirs(self.report_dir)
        self.boeck_set = datasets.BockSet()
        self.trainer = trainer

    def test_model_on_test_set(self,
                               test_set_index=0,
                               train_info=None,
                               report_filename="Report.txt", **kwargs):
        if train_info is None:
            train_info = {}
        # test
        print(f"Evaluating model...")
        t0 = time.perf_counter()
        counts = {}
        print(f"Lambdas: {self.lambdas}")
        # concurrency is implemented by test tasks (trainer.test_on_key)
        for lambda_ in self.lambdas:
            count = self.trainer.test_only(test_set_index, lambda_=lambda_, **kwargs)
            counts[lambda_] = count

        # binary search for best lambda
        if self.dynamic_lambdas:
            while self.lambda_step > self.min_lambda_step:
                self.lambda_step = self.lambda_step * 0.5
                print(f"Lambda step: {self.lambda_step}")
                counts_tuples = sorted(counts.items())
                max_f = 0
                max_index = 0
                for i in range(len(counts_tuples)):
                    if counts_tuples[i][1].fmeasure > max_f:
                        max_f = counts_tuples[i][1].fmeasure
                        max_index = i
                start_index = max_index - 1
                stop_index = max_index + 1
                if start_index < 0:
                    start_index = 0
                if stop_index > len(counts_tuples) - 1:
                    stop_index = len(counts_tuples) - 1
                start_lambda = counts_tuples[start_index][0]
                stop_lambda = counts_tuples[stop_index][0]
                new_lambdas = np.arange(start_lambda, stop_lambda, self.lambda_step)
                # avoid duplicates
                new_lambdas = np.setdiff1d(new_lambdas, self.lambdas)
                # save all lambdas for future use in 8-fold cross validation
                # self.lambdas += new_lambdas
                self.lambdas = np.concatenate((self.lambdas, new_lambdas))
                self.lambdas = np.sort(self.lambdas)
                print(f"New Lambdas: {new_lambdas}")
                for lambda_ in new_lambdas:
                    count = self.trainer.test_only(test_set_index, lambda_=lambda_, **kwargs)
                    counts[lambda_] = count

        t1 = time.perf_counter()
        print(f"Evaluation done. Time elapsed {t1 - t0:.2f}s")
        # report file
        pathlib.Path(self.report_dir).mkdir(parents=True, exist_ok=True)
        with open(self.report_dir + report_filename, 'w') as file:
            file.write("Training and Test Report\n")
            self._write_report_parameters(file, self.trainer.model_config, self.trainer.training_config,
                                          epoch_now=train_info.get('epoch_now'), lr_now=train_info.get('lr'),
                                          valid_loss=train_info.get('valid_loss_record'))
            self.write_report_counts(file, counts)
        return counts

    def train_and_test_model(self,
                             test_set_index=0,
                             show_example_plot=False,
                             show_plot=False,
                             save_model=True,
                             report_filename='Report.txt',
                             model_filename=None,
                             initialize=True,
                             revert_to_best_checkpoint=True,
                             skip_training=False,
                             skip_testing=False,
                             test_every_epoch_no=0,
                             **kwargs):
        r"""
        :param revert_to_best_checkpoint: whether to revert the model to the check point yielding the best loss
        (before training and saving)
        :param report_filename: filename when saving the report. This is convenient for showing fold # in filename
        :param save_model: whether to save the model
        :param show_plot: whether to plot loss function record
        :param show_example_plot: whether to plot the example input and output
        :param test_set_index: the index of the test set
        :param initialize: Whether to initialize the network with random weights
        :param model_filename: filename to save the model. None for auto filename.
        :param skip_training: whether to skip training
        :param skip_testing: whether to skip testing
        :param test_every_epoch_no: test every n epochs. 0 for no test
        -----

        :return: a dict, item defined as (lambda, Counter)
        """
        print(f"device: {networks.device}")
        print(f"cpu {CPU_CORES} cores")
        print("Initializing Trainer and Model...")
        # initialize
        splits = self.trainer.boeck_set.splits
        # training
        print("Training model...")
        if initialize:
            self.trainer.initialize_model()
        if not skip_training:
            self.trainer.model.train()
            train_info = self.trainer.train_only(test_set_index,
                                                 callback=self.middle_test, test_every_epoch_no=test_every_epoch_no,
                                                 report_filename=report_filename,
                                                 **kwargs)
            self.trainer.model.eval()
            if revert_to_best_checkpoint:
                self.trainer.load_model_state_dict(train_info['best_state_dict'])

        else:
            train_info = {}
        # testing
        print("Saving and plotting...")
        if show_example_plot:
            test_output(self.trainer, splits[test_set_index][0])
        if not skip_training and show_plot:
            utils.plot_loss(train_info.get('loss_record'))
            utils.plot_loss(train_info.get('valid_loss_record'), title="Validation Loss")
        if save_model:
            self.trainer.save_cp(filename=model_filename)
        if not skip_testing:
            counts = self.test_model_on_test_set(test_set_index=test_set_index, train_info=train_info,
                                                 report_filename=report_filename, **kwargs)
            return counts
        else:
            return Counter()

    def middle_test(self, test_every_epoch_no=0, report_filename="MidReport.txt", **kwargs):
        r"""
        recommended parameters: filename, test_set_index
        test_set_index
        """
        # skip test if test_every_epoch_no is 0
        if test_every_epoch_no == 0:
            return
        # skip test if epoch_now is not a multiple of test_every_epoch_no
        epoch = kwargs.get('epoch')
        if not epoch or epoch % test_every_epoch_no != 0:
            return
        # expect kwargs epoch=, valid_loss_record=, best_state_dict=, best_valid_loss=,
        # need train_info epoch_now, valid_loss_record
        train_info = {
            'epoch_now': epoch,
            'valid_loss_record': kwargs.get('valid_loss_record'),
        }
        self.test_model_on_test_set(train_info=train_info, report_filename=report_filename + f".@{epoch}.txt", **kwargs)

    def train_and_test_8_fold(self,
                              show_example_plot=False,
                              show_plot=False,
                              save_model=True,
                              summary_report_filename='Report-Summary.txt',
                              skip_test=False,
                              **kwargs):
        # dict of (height, Counter)
        counts = {}
        counts_record = []
        print("[8-fold cross validation]")
        for i in range(0, len(self.boeck_set.splits)):
            print(f'[Fold] {i}')
            results = self.train_and_test_model(
                test_set_index=i,
                show_example_plot=show_example_plot,
                show_plot=show_plot,
                save_model=save_model,
                report_filename=f'Report-fold{i}.txt',
                skip_test=skip_test,
                **kwargs
            )
            if not skip_test:
                counts_record.append(results)
                for result in results.items():
                    if counts.get(result[0]) is None:
                        counts[result[0]] = Counter()
                    counts[result[0]] += result[1]
        if not skip_test:
            pathlib.Path(self.report_dir).mkdir(parents=True, exist_ok=True)
            with open(self.report_dir + summary_report_filename, 'w') as file:
                file.write("Training and 8-fold Test Report\n")
                self._write_report_parameters(file, self.trainer.model_config, self.trainer.training_config)
                file.write("\n**Summary**\n\n")
                self.write_report_counts(file, counts)
                file.write("\n**Reports for Each Fold**\n\n")
                for counts_each in counts_record:
                    self.write_report_counts(file, counts_each)

    def train_and_save_8_fold_concurrent(self,
                                         summary_report_filename='Report-Summary.txt',
                                         model_filename_prefix='model',
                                         skip_test=False,
                                         **kwargs):
        # dict of (height, Counter)
        counts = {}
        counts_record = []
        futures = []
        print("[8-fold cross validation]")
        for i in range(0, len(self.boeck_set.splits)):
            with ThreadPoolExecutor(max_workers=CPU_CORES) as executor:
                futures.append(executor.submit(self.train_and_test_model,
                                               test_set_index=i,
                                               show_example_plot=False,
                                               show_plot=False,
                                               save_model=True,
                                               model_filename=f'{model_filename_prefix}-fold{i}.cp',
                                               report_filename=f'Report-fold{i}.txt',
                                               skip_test=skip_test,
                                               **kwargs))
        print("Task submitted")
        for i in range(0, len(self.boeck_set.splits)):
            results = futures[i].result()
            print(f'[Fold] #{i} completed')
            if not skip_test:
                counts_record.append(results)
                for result in results.items():
                    if counts.get(result[0]) is None:
                        counts[result[0]] = Counter()
                    counts[result[0]] += result[1]
        if not skip_test:
            pathlib.Path(self.report_dir).mkdir(parents=True, exist_ok=True)
            with open(self.report_dir + summary_report_filename, 'w') as file:
                file.write("Training and 8-fold Test Report\n")
                self._write_report_parameters(file, self.trainer.model_config, self.trainer.training_config)
                file.write("\n**Summary**\n\n")
                self.write_report_counts(file, counts)
                file.write("\n**Reports for Each Fold**\n\n")
                for counts_each in counts_record:
                    self.write_report_counts(file, counts_each)

    @staticmethod
    def write_report_counts(file, counts):
        file.write(f"\n[Scores]\n")
        file.writelines([f"Height(Lambda)={ct_tp[0]}\n"
                         f"Precision:{ct_tp[1].precision:.5f} Recall:{ct_tp[1].recall:.5f} "
                         f"F-score:{ct_tp[1].fmeasure:.5f}\n"
                         f"TP:{ct_tp[1].tp} FP:{ct_tp[1].fp} FN:{ct_tp[1].fn}\n\n" for ct_tp in sorted(counts.items())])

    @staticmethod
    def _write_report_parameters(file, model_config, training_config,
                                 epoch_now=0, lr_now=0., valid_loss=None):
        file.write("[Parameters]\n")
        file.write(f"{vars(model_config)}\n")
        file.write(f"{vars(training_config)}\n")
        file.write(f"last learning rate: {lr_now}\n")
        file.write(f"target mode: {TARGET_MODE}\n")
        file.write(f"target delta: {TARGET_DELTA}\n")
        # file.write(f"scheduler gamma: {gamma}\n")
        file.write(f"epochs: {epoch_now}\n")
        if valid_loss:
            file.write(f"Loss(valid): {valid_loss}\n")


# ---------------ENTRIES---------------


def test_output(mgr, test_key, figsize=(14.4, 4.8)):
    boeck_set = datasets.BockSet()

    piece = boeck_set.get_piece(test_key)
    onset_seconds = boeck.onset_evaluation.combine_events(piece.get_onsets_seconds(), ONSET_DELTA)
    onsets_list = np.asarray(onset_seconds) * SAMPLING_RATE
    onsets_list = np.floor_divide(onsets_list, HOP_SIZE)

    output = mgr.predict(key=test_key)
    output = output.squeeze()
    output = output.T
    output = output.cpu()
    utils.plot_odf(output, title="Network(Trained)", onsets=onsets_list, figsize=figsize)

    # length, odfs, target, onsets_samples, audio_len = prepare_data(boeck_set, mgr.features, test_key)
    # utils.plot_odf(odfs, title="SuperFlux")
    # utils.plot_odf(target, title="Target")


def test_saved_model(filename, features=None, lambda_=3.):
    boeck_set = datasets.BockSet()
    model = torch.load(filename, map_location=networks.device)
    mgr = ModelManager(boeck_set, ModelConfig(features=features), TrainingConfig())
    mgr.load_model_state_dict(model.state_dict())
    test_output(mgr, boeck_set.splits[0][0])
    counter = mgr.test_on_keys(boeck_set.get_split(0), lambda_=lambda_)
    print(f"Precision {counter.precision}, Recall {counter.recall}, F-score {counter.fmeasure}")


def test_prepare_data():
    boeck_set = datasets.BockSet()
    splits = boeck_set.splits
    for i in range(len(splits[0])):
        t0 = time.perf_counter()
        length, odfs, target, onsets_samples, audio_len = prepare_data(boeck_set, None, splits[0][i])
        t1 = time.perf_counter()
        print(f"{audio_len:.2f}s, elapsed {t1 - t0:.2f}, {audio_len / (t1 - t0):.1f}x speed")


def test_cp(filename, lambda_):
    boeck_set = datasets.BockSet()
    # model = torch.load(filename, map_location=networks.device)
    mgr = ModelManager(boeck_set, ModelConfig(), TrainingConfig())
    mgr.load(filename)
    # test_output(mgr, boeck_set.splits[0][0])
    counter = mgr.test_on_keys(boeck_set.get_split(0), lambda_=lambda_)
    print(f"Precision {counter.precision}, Recall {counter.recall}, F-score {counter.fmeasure}")


def test_models(model_files, features):
    for file in model_files:
        boeck_set = datasets.BockSet()
        key = boeck_set.get_split(0)[0]
        onsets = boeck_set.get_piece(key).get_onsets_seconds()
        onsets = np.asarray(onsets) * (HOP_SIZE / SAMPLING_RATE)
        mgr = ModelManager(boeck_set, ModelConfig(features=features), TrainingConfig())
        mgr.load(file)
        output = mgr.predict(key=boeck_set.get_split(0)[0])
        output = output.squeeze()
        output = output.T
        output = output.cpu()
        utils.plot_odf(output[100:300], title="Network(Trained)", figsize=(14.4, 4.8))


def find_lr():
    global TARGET_MODE
    global TARGET_DELTA
    TARGET_MODE = 'linear'
    TARGET_DELTA = 2
    boeck_set = datasets.BockSet()
    mgr = ModelManager(boeck_set, ModelConfig(), TrainingConfig())
    lrs, losses = mgr.test_learning_rates()
    print(f"lrs: {lrs}")
    print(f"losses: {losses}")

    # plot lrs and losses
    from matplotlib import pyplot as plt
    fig = plt.figure(1)
    plt.plot(lrs, losses)
    ax = fig.axes[0]
    ax.set_xscale('log')
    plt.show()


def evaluate_saved(model_file, features, lambdas, filename="Report.txt", load_as_cp=True, set_cached=True,
                   test_set_index=0, model_config=None, **kwargs):
    global CACHED_PREPROCESSING
    CACHED_PREPROCESSING = set_cached
    boeck_set = datasets.BockSet()
    # key = boeck_set.get_split(0)[0]
    # onsets = boeck_set.get_piece(key).get_onsets_seconds()
    # onsets = np.asarray(onsets) * (HOP_SIZE / SAMPLING_RATE)
    if not model_config:
        model_config = ModelConfig(features=features)
    mgr = ModelManager(boeck_set, model_config, TrainingConfig())
    if load_as_cp:
        mgr.load_cp(model_file)
    else:
        mgr.load(model_file)
    task = TrainingTask(mgr, lambdas=lambdas)
    task.train_and_test_model(initialize=False, skip_training=True,
                              show_plot=False, show_example_plot=False,
                              report_filename=filename, test_set_index=test_set_index, **kwargs)
    # for lambda_ in lambdas:
    #     count = mgr.test_only(0, lambda_=lambda_)
    #     print(f"lambda: {lambda_}; precision: {count.precision}, recall: {count.recall}, f-score: {count.fmeasure}")


def evalutate_8f_saved(model_file_template, features, lambdas, filename="Report.txt",
                       load_as_cp=True, set_cached=True):
    pass


def train():
    global TARGET_MODE
    global TARGET_DELTA
    TARGET_MODE = 'linear'

    task_no = int(sys.argv[1])
    fold_no = int(sys.argv[2])

    if task_no == 1:
        TARGET_DELTA = 0
        features = ['superflux']
        model_config = ModelConfig(features=features)
        tconfig = TrainingConfig(optimizer_constructor=torch.optim.Adam, optimizer_kwargs={"lr": 1e-3},
                                 batch_size=256, epoch=5000)
        report_filename = f"Report_superfluxnn-fold-%i-delta-{TARGET_DELTA}.txt"
        model_filename = f"Model_superfluxnn-fold-%i-delta-{TARGET_DELTA}.pt"
    elif task_no == 2:
        TARGET_DELTA = 1
        features = ['rcd', 'superflux']
        model_config = ModelConfig(features=features)
        tconfig = TrainingConfig(optimizer_constructor=torch.optim.Adam, optimizer_kwargs={"lr": 1e-3},
                                 batch_size=256, epoch=5000)
        report_filename = f"Report_rcd-spf-fold-%i-delta-{TARGET_DELTA}.txt"
        model_filename = f"Model_rcd-spf-fold-%i-delta-{TARGET_DELTA}.pt"
    elif task_no == 3:
        TARGET_DELTA = 1
        features = ['rcd', 'superflux']
        model_config = ModelConfig(features=features, num_layer_unit=8)
        tconfig = TrainingConfig(optimizer_constructor=torch.optim.Adam, optimizer_kwargs={"lr": 1e-3},
                                 batch_size=256, epoch=5000)
        report_filename = f"Report_rcd-spf-fold-%i-delta-{TARGET_DELTA}-2x8.txt"
        model_filename = f"Model_rcd-spf-fold-%i-delta-{TARGET_DELTA}-2x8.pt"
    elif task_no == 4:
        TARGET_DELTA = 1
        features = ['rcd', 'superflux']
        model_config = ModelConfig(features=features, num_layers=3)
        tconfig = TrainingConfig(optimizer_constructor=torch.optim.Adam, optimizer_kwargs={"lr": 1e-3},
                                 batch_size=256, epoch=5000)
        report_filename = f"Report_rcd-spf-fold-%i-delta-{TARGET_DELTA}-3x4.txt"
        model_filename = f"Model_rcd-spf-fold-%i-delta-{TARGET_DELTA}-3x4.pt"

    # scheduler_constructor = torch.optim.lr_scheduler.CyclicLR,
    # scheduler_kwargs = {"base_lr": 1e-2, "max_lr": 6e-2, "step_size_up": 10,
    #                     "cycle_momentum": False})
    trainer = ModelManager(datasets.BockSet(), model_config, tconfig)
    task = TrainingTask(trainer)
    # task.train_and_test_8_fold()

    trainer.training_config.early_stop_patience = 50
    task.train_and_test_model(initialize=False, save_model=True,
                              skip_testing=True,
                              report_filename=report_filename % fold_no,
                              model_filename=model_filename % fold_no,
                              test_set_index=fold_no,
                              show_plot=False,
                              show_example_plot=False)


def verify_loss():
    tconfig = TrainingConfig()
    features = ['superflux']
    boeck_set = datasets.BockSet()
    trainer = ModelManager(boeck_set, ModelConfig(features=features), tconfig,
                           load_file=r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220402 085854 100fps 50es spf\spf-d0.pt")
    v_keys = boeck_set.generate_splits(0)[1]
    valid_loader = BoeckDataLoader(datasets.BockSet(), v_keys, len(v_keys), features=features, shuffle=False)
    trainer._validate_loss(1, valid_loader)


def verify_save_model_function():
    features = ['superflux']
    filename = "test.pt"
    boeck_set = datasets.BockSet()
    mgr = ModelManager(boeck_set, ModelConfig(features=features),
                       TrainingConfig(optimizer_constructor=torch.optim.SGD,
                                      optimizer_kwargs={"lr": 1e-2},
                                      epoch=10, batch_size=256))
    task = TrainingTask(mgr)
    task.train_and_test_model(show_plot=False, show_example_plot=False, model_filename=filename,
                              skip_testing=True)
    model1 = mgr.model
    mgr_saved = ModelManager(boeck_set, ModelConfig(features=features),
                             TrainingConfig(optimizer_constructor=torch.optim.SGD,
                                            optimizer_kwargs={"lr": 1e-2},
                                            epoch=10, batch_size=256),
                             load_file=RUN_DIR + filename)
    model_state_dict_1 = model1.state_dict()
    model_state_dict_2 = mgr_saved.model.state_dict()
    for ((k_1, v_1), (k_2, v_2)) in zip(
            model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {v_1} vs {v_2}")
    v_keys = boeck_set.generate_splits(0)[1]
    valid_loader = BoeckDataLoader(datasets.BockSet(), v_keys, len(v_keys), features=features)
    mgr_saved._validate_loss(1, valid_loader)


def speed_test(filename, model_config, lambda_=0.3, smooth=0.05):
    global CACHED_PREPROCESSING
    CACHED_PREPROCESSING = False
    networks.device = torch.device('cpu')
    trainer = ModelManager(datasets.BockSet(), model_config, TrainingConfig())
    trainer.load_cp(filename)
    speed = trainer.speed_test_on_keys(datasets.BockSet().get_split(0), lambda_=lambda_, smooth=smooth)
    print(f"{speed}x")


if __name__ == '__main__':
    # evaluate_saved(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220402 085854 100fps\spf-d0.pt"
    #                , ['superflux'], None,
    #                filename="Report-spf-static-d0.txt")
    # train()
    # evaluate_saved(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task1-d0.pt",
    #                ['superflux'], None, filename="Report-task1-d0-static.txt")
    # evaluate_saved(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task1-d1.pt",
    #                ['superflux'], None, filename="Report-task1-d1-static.txt")
    # evaluate_saved(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task1-d2.pt",
    #                ['superflux'], None, filename="Report-task1-d2-static.txt")
    # evaluate_saved(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task2-d0.pt",
    #                ['rcd','superflux'], None, filename="Report-task2-d0-static.txt")
    # evaluate_saved(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task2-d1.pt",
    #                ['rcd','superflux'], None, filename="Report-task2-d1-static.txt")
    # evaluate_saved(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task2-d2.pt",
    #                ['rcd','superflux'], None, filename="Report-task2-d2-static.txt")
    # model_config = ModelConfig(features=['rcd', 'superflux'], num_layers=2, num_layer_unit=8)
    # evaluate_saved(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220522 8-f\Model_rcd-spf-fold-0-delta-1-2x8.pt",
    #                ['rcd','superflux'], None, filename="Report-2x8-d1-static.txt", model_config=model_config,
    #                smooth=0.05)
    # model_config = ModelConfig(features=['rcd', 'superflux'], num_layers=3, num_layer_unit=4)
    # evaluate_saved(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220522 8-f\Model_rcd-spf-fold-0-delta-1-3x4.pt",
    #                ['rcd','superflux'], None, filename="Report-3x4-d1-static.txt", model_config=model_config,
    #                smooth=0.05)

    mc_spnn = ModelConfig(features=['superflux'])
    mc_rsnn = ModelConfig(features=['rcd', 'superflux'])
    mc_rsnn_2_8 = ModelConfig(features=['rcd', 'superflux'], num_layer_unit=8)
    mc_rsnn_3_4 = ModelConfig(features=['rcd', 'superflux'], num_layers=3)

    speed_test(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task1-d0.pt",
               mc_spnn,
               lambda_=0.15, smooth=0)
    speed_test(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task1-d0.pt",
               mc_spnn,
               lambda_=0.25, smooth=0.05)
    speed_test(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task2-d2.pt",
               mc_rsnn,
               lambda_=0.3, smooth=0)
    speed_test(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task2-d1.pt",
               mc_rsnn,
               lambda_=0.5, smooth=0.05)
    speed_test(
        r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220522 8-f\Model_rcd-spf-fold-0-delta-1-2x8.pt",
        mc_rsnn_2_8,
        lambda_=0.45, smooth=0.05)
    speed_test(
        r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220522 8-f\Model_rcd-spf-fold-0-delta-1-2x8.pt",
        mc_rsnn_2_8,
        lambda_=0.3, smooth=0)
    speed_test(
        r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220522 8-f\Model_rcd-spf-fold-0-delta-1-3x4.pt",
        mc_rsnn_3_4,
        lambda_=0.5, smooth=0.05)
    speed_test(
        r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220522 8-f\Model_rcd-spf-fold-0-delta-1-3x4.pt",
        mc_rsnn_3_4,
        lambda_=0.3, smooth=0)
    # HOP_SIZE = 220.5
    # SPF_LAG = 2
    # for i in range(0, 3):
    #     evaluate_saved(
    #         f"E:\\Documents\\Personal\\WorkSpace\\Research\\ComplexDODF\\run 20220329 062429 200fps targetfix 5000epochs spf\\spf-d{i}.pt",
    #         ['superflux'], np.arange(.1, .9, .05), filename=f"spf-200fps-d{i}-smoothed50.txt")
    # HOP_SIZE = 441
    # SPF_LAG = 1
    # for i in range(0, 3):
    #     evaluate_saved(
    #         f"E:\\Documents\\Personal\\WorkSpace\\Research\\ComplexDODF\\run 20220402 085854 100fps 50es spf\\spf-d{i}.pt",
    #         ['superflux'], np.arange(.1, .7, .05), filename=f"spf-100fps-d{i}-smoothed05.txt")
    # for i in range(0, 3):
    #     evaluate_saved(
    #         f"E:\\Documents\\Personal\\WorkSpace\\Research\\ComplexDODF\\run 20220405 134316 100 50es cd+spf\\spf+cd-d{i}.pt",
    #         ['rcd', 'superflux'], np.arange(.7, .9, .05), filename=f"spf+cd-d{i}-smoothed50-more.txt")
    # verify_loss()
    # verify_save_model_function()
    # find_lr()
    #     r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220217 161019\mdl_20220217 190649_tanh_2x4(bi).pt"
    #     , ['superflux'], None)

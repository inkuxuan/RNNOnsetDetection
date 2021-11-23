import random

import boeck.onset_program
import datasets
import networks
import onsets
import onset_functions
import utils

from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import multiprocessing
import gc
import os

import librosa
import scipy.signal
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

SAMPLING_RATE = 44100
N_FFT = 2048
HOP_SIZE = 441
ONSET_DELTA = 0.030
# in ['single', 'linear']
TARGET_MODE = 'linear'
DEFAULT_FEATURES = ['rcd', 'superflux']

# function reference for onset combiner
# in [onsets.combine_onsets_avg, onsets.merge_onsets, None]
# stands for combining onsets by average, by the first onset, and no combination
COMBINE_ONSETS = onsets.combine_onsets_avg
COMBINE_ONSETS_DETECTION = True

now = datetime.now()
dstr = now.strftime("%Y%m%d %H%M%S")
RUN_DIR = f'run {dstr}/'
os.makedirs(RUN_DIR)


def get_features(wave, features=None, n_fft=N_FFT, hop_size=HOP_SIZE, sr=SAMPLING_RATE, center=False) -> np.ndarray:
    r"""
    :param center: Leave this to False, this will disable librosa's centering feature
    :param sr: sampling rate
    :param hop_size: FFT hop length
    :param n_fft: FFT window size
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
            onset_strength = onset_functions.complex_domain_odf(stft)
            f.append(onset_strength)
        elif feature in ['rectified_complex_domain', 'rcd']:
            onset_strength = onset_functions.complex_domain_odf(stft, rectify=True)
            f.append(onset_strength)
        elif feature in ['super_flux', 'superflux']:
            onset_strength, _, _ = onset_functions.super_flux_odf(stft, sr, n_fft, hop_size, center)
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
    if COMBINE_ONSETS:
        onsets_list = COMBINE_ONSETS(piece.get_onsets_seconds(), ONSET_DELTA)
    # convert from second to sample
    onsets_list = np.asarray(onsets_list) * sr
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
    return length, odfs, target, onsets_list, len(wave) / sr


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
        self.optimizer = self.training_config.optimizer(self.model.parameters(),
                                                        *self.training_config.optimizer_args,
                                                        **self.training_config.optimizer_kwargs)
        if self.training_config.scheduler_constructor:
            self.scheduler = self.training_config.scheduler_constructor(self.optimizer,
                                                                        *self.training_config.scheduler_args,
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
        torch.save(self.model, RUN_DIR + filename)

    def save_cp(self, filename):
        checkpoint = {
            'model': self.model.state_dict() if self.model else None,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None
        }
        torch.save(checkpoint, RUN_DIR + filename)

    def load(self, path):
        r"""
        This WILL NOT re-initialize the optimizer and the scheduler.
        Use the constructor instead if wish to train the network.
        """
        self.model = torch.load(path, map_location=networks.device)

    def load_state_dict(self, state_dict):
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

    def predict_onsets_offline(self, key=None, wave=None, height=0.5, **kwargs):
        r"""
        either `key` or `wave` is needed

        :param wave: a wave audio in 1-d array
        :param key: key of the track
        :param height: minimum height of a peak
        :return: Seconds of onsets, not combined
        """
        out = self.predict(key=key, wave=wave, **kwargs)
        out = out.squeeze()
        out = out.cpu()
        if TARGET_MODE != 'precise':
            peaks = scipy.signal.find_peaks(out, height=height)
            frame_rate = SAMPLING_RATE / HOP_SIZE
            onsets_sec = np.array(peaks[0]) / frame_rate
            return onsets_sec
        # TODO precise mode decoding
        return None

    def predict_onsets_online(self, key=None, wave=None, height=0.5, **kwargs):
        r"""
        either `key` or `wave` is needed

        :param wave: a wave audio in 1-d array
        :param key: key of the track
        :param height: trigger value for rising edges
        :return: Seconds of onsets, not combined
        """
        out = self.predict(key=key, wave=wave, **kwargs)
        out = out.squeeze()
        out = out.cpu()
        rising_edges = np.flatnonzero(
            ((out[:-1] <= height) & (out[1:] > height) | (out[:-1] < height) & (out[1:] >= height))) + 1
        frame_rate = SAMPLING_RATE / HOP_SIZE
        onsets_sec = rising_edges / frame_rate
        return onsets_sec

    def generate_splits(self, test_split_index) -> tuple[list, list, list]:
        r"""
        :return: (training_set_keys, validation_set_keys, test_set_keys)
        """
        splits = self.boeck_set.splits.copy()
        # setup training set, validation set, test set
        test_set_keys = splits.pop(test_split_index)
        # set the first one as validation set
        validation_set_keys = splits.pop(test_split_index % len(splits))
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
                       verbose=True,
                       save_checkpoint=True,
                       **kwargs):
        r"""
        Train the network on a specific training set and validation set

        -----
        :param save_checkpoint: if being True or a str, constantly saves the model to a file
        :param validation_keys: list of keys in validation set
        :param training_keys: list of keys in training set
        :param batch_size: How many pieces at a time to train the network.
        Training process uses concurrency for both pre-processing and training. This is default the cores count of CPU.
        :param verbose: bool, Print debug outputs
        :return: dict containing {loss_record, valid_loss_record, epoch_now, lr, best_state_dict}
        """
        loader = BoeckDataLoader(self.boeck_set, training_keys, batch_size, features=self.features)
        valid_loader = BoeckDataLoader(self.boeck_set, validation_keys, len(validation_keys), features=self.features)
        self.model.train()
        continue_epoch = True
        epoch_now = 0
        loss_record = []
        valid_loss_record = []
        best_valid_loss = float('inf')
        checkpoint = self.model.state_dict()
        early_stopping = False
        if self.training_config.early_stop_patience > 0:
            early_stopping = utils.EarlyStopping(patience=self.training_config.early_stop_patience)
        while continue_epoch:
            epoch_now += 1
            print(f"===EPOCH {epoch_now}===")
            print(f"lr={self.optimizer.param_groups[0]['lr']}")
            avg_loss = self._fit(epoch_now, loader, verbose)
            loss_record.append(avg_loss)
            if self.training_config.epoch and epoch_now >= self.training_config.epoch:
                continue_epoch = False
            # VALIDATION
            valid_loss = self._validate_loss(epoch_now, valid_loader)
            valid_loss_record.append(valid_loss)
            # LEARNING RATE UPDATE
            if self.scheduler:
                self.scheduler.step()
            # Checkpoint saving
            if valid_loss < best_valid_loss:
                checkpoint = self.model.state_dict()
                if save_checkpoint:
                    if isinstance(save_checkpoint, str):
                        self.save_cp(save_checkpoint)
                    else:
                        self.save_cp("checkpoint.pt")
            # EARLY STOPPING
            if early_stopping:
                early_stopping(valid_loss)
            if early_stopping.early_stop:
                continue_epoch = False
        return {'loss_record': loss_record,
                'valid_loss_record': valid_loss_record,
                'epoch_now': epoch_now,
                'lr': self.optimizer.param_groups[0]['lr'],
                'best_state_dict': checkpoint}

    # noinspection DuplicatedCode
    def _fit(self, epoch_now, loader, verbose):
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

    def train_only(self, test_split_index, verbose=False, **kwargs):
        r"""
        :return: dict containing {loss_record, valid_loss_record, epoch_now, lr, best_state_dict}
        """
        training_keys, validation_keys, test_keys = self.generate_splits(test_split_index)
        info = self.train_on_split(training_keys, validation_keys, verbose=verbose, **kwargs)
        return info

    def train_and_test(self, test_split_index, verbose=False, online=False, height=0.5, window=0.025, delay=0,
                       **kwargs):
        r"""

        :param verbose: whether to print a message every minibatch
        :param test_split_index: the split index of test set
        :param online: bool
        :param height: minimum height for an onset
        :param window: evaluation window radius, in seconds
        :param delay: time delay for detections, in seconds
        :return: counter, training_info (dict)
        """
        info = self.train_only(test_split_index, verbose=verbose, **kwargs)
        counter = self.test_only(test_split_index, online=online, height=height, window=window, delay=delay, **kwargs)
        return counter, info

    def test_only(self, test_split_index, online=False,
                  combine_output_onsets=COMBINE_ONSETS_DETECTION, height=0.5, window=0.025, delay=0, **kwargs):
        r"""

        :param online: bool
        :param height: minimum height for an onset
        :param window: evaluation window radius, in seconds
        :param delay: time delay for detections, in seconds
        :return: counter
        """
        test_keys = self.boeck_set.get_split(test_split_index)
        return self.test_on_keys(test_keys, online=online, combine_output_onsets=combine_output_onsets,
                                 height=height, window=window, delay=delay, **kwargs)

    def test_on_keys(self, keys, online=False, combine_output_onsets=COMBINE_ONSETS_DETECTION,
                     height=0.5, window=0.025, delay=0,
                     concurrent=True, **kwargs):
        r"""

        :param keys: keys of the dataset to test
        :param online: bool
        :param height: minimum height for an onset
        :param window: evaluation window radius, in seconds
        :param delay: time delay for detections, in seconds
        :return: counter
        """
        count = Counter()
        if concurrent:
            with ThreadPoolExecutor(max_workers=max([CPU_CORES, len(keys)])) as executor:
                futures = []
                for key in keys:
                    future = executor.submit(self.test_on_key, key, online=online, height=height, window=window,
                                             combine_output_onsets=combine_output_onsets,
                                             delay=delay, **kwargs)
                    futures.append(future)
                for future in futures:
                    count += future.result()
        else:
            for key in keys:
                count1 = self.test_on_key(key, online=online,
                                          combine_output_onsets=combine_output_onsets, height=height, window=window,
                                          delay=delay, **kwargs)
                count += count1
        return count

    def test_on_key(self, key, online=False, combine_output_onsets=COMBINE_ONSETS_DETECTION,
                    height=0.5, window=0.025, delay=0, **kwargs):
        ground_truth = self.boeck_set.get_piece(key).get_onsets_seconds()
        if online:
            detections = self.predict_onsets_online(key=key, height=height, **kwargs)
        else:
            detections = self.predict_onsets_offline(key=key, height=height, **kwargs)
        if COMBINE_ONSETS:
            if combine_output_onsets and not online:
                detections = COMBINE_ONSETS(detections, ONSET_DELTA, key=f"detection:{key}, height={height}")
            ground_truth = COMBINE_ONSETS(ground_truth, ONSET_DELTA)
        count = boeck.onset_evaluation.count_errors(detections, ground_truth, window, delay=delay)
        return count


class TrainingTask(object):
    def __init__(self,
                 trainer: ModelManager,
                 heights=None):
        r"""
        :parameter heights: list of thresholds for evaluation of onsets
        """
        if heights is None:
            heights = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
        self.heights = heights
        now = datetime.now()
        dstr = now.strftime("%Y%m%d %H%M%S")
        self.report_dir = RUN_DIR + f'report {dstr}/'
        os.makedirs(self.report_dir)
        self.boeck_set = datasets.BockSet()
        self.trainer = trainer

    def train_and_test_model(self,
                             test_set_index=0,
                             show_example_plot=True,
                             show_plot=True,
                             save_model=True,
                             filename='Report.txt',
                             initialize=True,
                             revert_to_best_checkpoint=True,
                             **kwargs):
        r"""
        :param revert_to_best_checkpoint: whether to revert the model to the check point yielding the best loss
        (before training and saving)
        :param filename: filename when saving the report. This is convenient for showing fold # in filename
        :param save_model: whether to save the model
        :param show_plot: whether to plot loss function record
        :param show_example_plot: whether to plot the example input and output
        :param test_set_index: the index of the test set
        :param initialize: Whether to initialize the network with random weights
        -----

        :return: a dict, item defined as (height, Counter)
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
        train_info = self.trainer.train_only(test_set_index, **kwargs)
        if revert_to_best_checkpoint:
            self.trainer.load_state_dict(train_info['best_state_dict'])
        # testing
        print("Saving and plotting...")
        if show_example_plot:
            test_output(self.trainer, splits[test_set_index][0])
        if show_plot:
            utils.plot_loss(train_info['loss_record'])
            utils.plot_loss(train_info['valid_loss_record'], title="Validation Loss")
        if save_model:
            self.trainer.save_model()
        # test
        print(f"Evaluating model... ({len(self.heights)} tasks)")
        t0 = time.perf_counter()
        counts = {}
        # concurrency is implemented by test tasks (trainer.test_on_key)
        for height in self.heights:
            count = self.trainer.test_only(test_set_index, height=height, **kwargs)
            counts[height] = count

        t1 = time.perf_counter()
        print(f"Evaluation done. Time elapsed {t1 - t0:.2f}s")
        # report
        with open(self.report_dir + filename, 'w') as file:
            file.write("Training and Test Report\n")
            self._write_report_parameters(file, self.trainer.model_config, self.trainer.training_config,
                                          epoch_now=train_info['epoch_now'], lr_now=train_info['lr'],
                                          valid_loss=train_info['valid_loss_record'])
            self.write_report_counts(file, counts)
        return counts

    def train_and_test_8_fold(self,
                              show_example_plot=False,
                              show_plot=False,
                              save_model=True,
                              filename='Report-Summary.txt',
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
                filename=f'Report-fold{i}.txt',
                **kwargs
            )
            counts_record.append(results)
            for result in results.items():
                if counts.get(result[0]) is None:
                    counts[result[0]] = Counter()
                counts[result[0]] += result[1]
        with open(self.report_dir + filename, 'w') as file:
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
        file.writelines([f"Height={ct_tp[0]}\n"
                         f"Precision:{ct_tp[1].precision:.5f} Recall:{ct_tp[1].recall:.5f} "
                         f"F-score:{ct_tp[1].fmeasure:.5f}\n"
                         f"TP:{ct_tp[1].tp} FP:{ct_tp[1].fp} FN:{ct_tp[1].fn}\n\n" for ct_tp in sorted(counts.items())])

    @staticmethod
    def _write_report_parameters(file, model_config, training_config,
                                 epoch_now=0, lr_now=0., valid_loss=None):
        file.write("[Parameters]\n")
        file.write(f"{model_config}\n")
        file.write(f"{training_config}\n")
        file.write(f"last learning rate: {lr_now}\n")
        # file.write(f"scheduler gamma: {gamma}\n")
        file.write(f"epochs: {epoch_now}")
        if valid_loss:
            file.write(f"Loss(valid): {valid_loss}\n")


def test_output(mgr, test_key):
    boeck_set = datasets.BockSet()

    piece = boeck_set.get_piece(test_key)
    onset_seconds = boeck.onset_evaluation.combine_events(piece.get_onsets_seconds(), ONSET_DELTA)
    onsets_list = np.asarray(onset_seconds) * SAMPLING_RATE
    onsets_list = np.floor_divide(onsets_list, HOP_SIZE)

    output = mgr.predict(key=test_key)
    output = output.squeeze()
    output = output.T
    output = output.cpu()
    utils.plot_odf(output, title="Network(Trained)", onsets=onsets_list)

    length, odfs, target, onsets_samples, audio_len = prepare_data(boeck_set, mgr.features, test_key)
    utils.plot_odf(odfs, title="SuperFlux")
    utils.plot_odf(target, title="Target")


def test_saved_model(filename, features=None, height=0.35):
    boeck_set = datasets.BockSet()
    model = torch.load(filename, map_location=networks.device)
    mgr = ModelManager(boeck_set, ModelConfig(features=features), TrainingConfig())
    mgr.model = model
    test_output(mgr, boeck_set.splits[0][0])
    counter = mgr.test_on_keys(boeck_set.get_split(0), height=height)
    print(f"Precision {counter.precision}, Recall {counter.recall}, F-score {counter.fmeasure}")


def test_prepare_data():
    boeck_set = datasets.BockSet()
    splits = boeck_set.splits
    for i in range(len(splits[0])):
        t0 = time.perf_counter()
        length, odfs, target, onsets_samples, audio_len = prepare_data(boeck_set, None, splits[0][i])
        t1 = time.perf_counter()
        print(f"{audio_len:.2f}s, elapsed {t1 - t0:.2f}, {audio_len / (t1 - t0):.1f}x speed")


def test_cp(filename, height):
    boeck_set = datasets.BockSet()
    # model = torch.load(filename, map_location=networks.device)
    mgr = ModelManager(boeck_set, ModelConfig(), TrainingConfig())
    mgr.load(filename)
    # test_output(mgr, boeck_set.splits[0][0])
    counter = mgr.test_on_keys(boeck_set.get_split(0), height=height)
    print(f"Precision {counter.precision}, Recall {counter.recall}, F-score {counter.fmeasure}")


def train_adam():
    features = ['rcd', 'superflux']
    global TARGET_MODE
    TARGET_MODE = 'linear'
    trainer = ModelManager(datasets.BockSet(), ModelConfig(features=features, num_layer_unit=6), TrainingConfig())
    task = TrainingTask(trainer)
    task.train_and_test_model(initialize=False, save_model=True, save_checkpoint=True)

    trainer = ModelManager(datasets.BockSet(), ModelConfig(features=features, num_layer_unit=8), TrainingConfig())
    task = TrainingTask(trainer)
    task.train_and_test_model(initialize=False, save_model=True, save_checkpoint=True)


if __name__ == '__main__':
    # print("Task 1: RCD+SuperFlux")
    # task_sf = TrainingTask(features=['rcd', 'superflux'])
    # task_sf.train_and_test_model(save_model=True)
    # print("Task 2: SuperFlux (baseline)")
    # task_sf = TrainingTask(features=['superflux'])
    # task_sf.train_and_test_model(save_model=True, save_checkpoint=True)
    # print("Task 3: RCD+SuperFlux (8x2)")
    # task_sf = TrainingTask(features=['rcd', 'superflux'], n_layers=2, n_units=8)
    # task_sf.train_and_test_model(save_model=True)
    train_adam()

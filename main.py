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
TARGET_MODE = 'linear'
DEFAULT_FEATURES = ['rcd', 'superflux']

INITIAL_LR = 0.125
MIN_LR = 0.001
GAMMA = 0.8
SCHEDULER_PATIENCE = 5
EARLY_STOP_PATIENCE = 20
MAX_EPOCH = 3000
# in [boeck.onset_evaluation.combine_events, onsets.merge_onsets, None]
# stands for combining onsets by average, by the first onset, and no combination
COMBINE_ONSETS = boeck.onset_evaluation.combine_events


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


# noinspection PyUnusedLocal
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
                 scheduler=None,
                 init_lr=INITIAL_LR,
                 scheduler_patience=SCHEDULER_PATIENCE,
                 early_stop_patience=EARLY_STOP_PATIENCE,
                 gamma=GAMMA,
                 min_lr=MIN_LR):
        r"""

        :param features: list, element in ['rcd', 'cd', 'superflux']; default ['rcd', 'superflux']
        :param num_layer_unit: Number of units in one hidden layer
        :param num_layers: Number of hidden layers
        :param loss_fn: default BCEWithLogitsLoss
        :param optimizer: default SGD
        :param scheduler: default milestone scheduler. Set to False to disable scheduler
        """
        self.early_stop_patience = early_stop_patience
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
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=init_lr)
        self.scheduler = scheduler
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_patience,
                factor=gamma,
                min_lr=min_lr,
                verbose=True
            )

    def initialize_model(self):
        self.model.init_normal()

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
                       verbose=False,
                       debug_max_epoch_count=None,
                       early_stopping=True,
                       **kwargs):
        r"""
        Train the network on a specific training set and validation set

        -----
        :param early_stopping: whether or not to perform early stopping according to patience set when initialized
        :param debug_max_epoch_count: default None, if set to a int, end training in the specified epochs
        :param validation_keys: list of keys in validation set
        :param training_keys: list of keys in training set
        :param batch_size: How many pieces at a time to train the network.
        Training process uses concurrency for both pre-processing and training. This is default the cores count of CPU.
        :param verbose: bool, Print debug outputs
        :return: dict containing {loss_record, valid_loss_record, epoch_now, lr}
        """
        loader = BoeckDataLoader(self.boeck_set, training_keys, batch_size, features=self.features)
        valid_loader = BoeckDataLoader(self.boeck_set, validation_keys, len(validation_keys), features=self.features)
        continue_epoch = True
        epoch_now = 0
        loss_record = []
        valid_loss_record = []
        if early_stopping:
            early_stopping = utils.EarlyStopping(patience=EARLY_STOP_PATIENCE)
        while continue_epoch:
            epoch_now += 1
            print(f"===EPOCH {epoch_now}===")
            print(f"lr={self.optimizer.param_groups[0]['lr']}")
            avg_loss = self._fit(epoch_now, loader, verbose)
            loss_record.append(avg_loss)
            if debug_max_epoch_count and epoch_now >= debug_max_epoch_count:
                continue_epoch = False
            # VALIDATION
            valid_loss = self._validate_loss(epoch_now, valid_loader)
            valid_loss_record.append(valid_loss)
            # LEARNING RATE UPDATE
            if self.scheduler:
                self.scheduler.step(valid_loss)
            # EARLY STOPPING
            if early_stopping:
                early_stopping(valid_loss)
            if early_stopping.early_stop:
                continue_epoch = False
        return {'loss_record': loss_record,
                'valid_loss_record': valid_loss_record,
                'epoch_now': epoch_now,
                'lr': self.optimizer.param_groups[0]['lr']}

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
        :return: dict containing {loss_record, valid_loss_record, epoch_now, lr}
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

    def test_only(self, test_split_index, online=False, height=0.5, window=0.025, delay=0, **kwargs):
        r"""

        :param online: bool
        :param height: minimum height for an onset
        :param window: evaluation window radius, in seconds
        :param delay: time delay for detections, in seconds
        :return: counter
        """
        test_keys = self.boeck_set.get_split(test_split_index)
        return self.test_on_keys(test_keys, online=online, height=height, window=window, delay=delay, **kwargs)

    def test_on_keys(self, keys, online=False, height=0.5, window=0.025, delay=0, concurrent=True, **kwargs):
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
                                             delay=delay, **kwargs)
                    futures.append(future)
                for future in futures:
                    count += future.result()
        else:
            for key in keys:
                count1 = self.test_on_key(key, online=online, height=height, window=window, delay=delay, **kwargs)
                count += count1
        return count

    def test_on_key(self, key, online=False, height=0.5, window=0.025, delay=0, **kwargs):
        ground_truth = self.boeck_set.get_piece(key).get_onsets_seconds()
        if online:
            detections = self.predict_onsets_online(key=key, height=height, **kwargs)
        else:
            detections = self.predict_onsets_offline(key=key, height=height, **kwargs)
        if COMBINE_ONSETS:
            detections = COMBINE_ONSETS(detections, ONSET_DELTA)
            ground_truth = COMBINE_ONSETS(ground_truth, ONSET_DELTA)
        count = boeck.onset_evaluation.count_errors(detections, ground_truth, window, delay=delay)
        return count


class TrainingTask(object):
    def __init__(self,
                 weight=1,
                 init_lr=INITIAL_LR,
                 gamma=GAMMA,
                 epoch=MAX_EPOCH,
                 batch_size=64,
                 nonlinearty='tanh',
                 heights=None,
                 features=None,
                 trainer=None,
                 n_layers=2,
                 n_units=4):
        r"""
        Specify a trainer if detailed settings are needed.
        If a trainer is specified, the following parameters are ignored.
        Otherwise a trainer will be instantiated using the following parameters:
        weight, init_lr, step_size, gamma, epoch, batch_size, nonlinearty, features

        -----
        :parameter heights: list of thresholds for evaluation of onsets
        """
        if heights is None:
            heights = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
        if features is None:
            features = DEFAULT_FEATURES
        self.n_units = n_units
        self.n_layers = n_layers
        self.weight = weight
        self.init_lr = init_lr
        self.gamma = gamma
        self.epoch = epoch
        self.batch_size = batch_size
        self.nonlinearty = nonlinearty
        self.heights = heights
        self.features = features
        now = datetime.now()
        dstr = now.strftime("%Y%m%d %H%M%S")
        self.report_dir = f'report {dstr}/'
        os.makedirs(self.report_dir)
        self.boeck_set = datasets.BockSet()
        self.trainer = trainer

    def train_and_test_model(self,
                             test_set_index=0,
                             show_example_plot=True,
                             show_plot=True,
                             save_model=True,
                             filename='Report.txt',
                             **kwargs):
        r"""
        :return: a dict, item defined as (height, Counter)
        """
        print(f"device: {networks.device}")
        print(f"cpu {CPU_CORES} cores")
        print("Initializing Trainer and Model...")
        # configure
        if not self.trainer:
            self.trainer = ModelManager(self.boeck_set, bidirectional=True,
                                        features=self.features, nonlinearity=self.nonlinearty,
                                        init_lr=self.init_lr, gamma=self.gamma,
                                        num_layers=self.n_layers, num_layer_unit=self.n_units)
            self.trainer.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.weight))
        # initialize
        splits = self.trainer.boeck_set.splits
        # training
        print("Training model...")
        self.trainer.initialize_model()
        train_info = self.trainer.train_only(test_set_index,
                                             debug_max_epoch_count=self.epoch,
                                             verbose=True, batch_size=self.batch_size,
                                             **kwargs)
        # testing
        print("Saving and plotting...")
        if show_example_plot:
            test_output(self.trainer, splits[test_set_index][0])
        if show_plot:
            utils.plot_loss(train_info['loss_record'])
            utils.plot_loss(train_info['valid_loss_record'], title="Validation Loss")
        if save_model:
            self.trainer.save()
        # test
        print(f"Evaluating model... ({len(self.heights)} tasks)")
        t0 = time.perf_counter()
        counts = {}
        # concurrency is implemented by test tasks (trainer.test_on_key)
        for height in self.heights:
            count = self.trainer.test_only(test_set_index, height=height)
            counts[height] = count

        t1 = time.perf_counter()
        print(f"Evaluation done. Time elapsed {t1 - t0:.2f}s")
        # report
        with open(self.report_dir + filename, 'w') as file:
            file.write("Training and Test Report\n")
            self._write_report_parameters(file, self.n_layers, self.n_units,
                                          self.weight, self.init_lr, self.gamma,
                                          self.epoch, self.batch_size, self.features,
                                          epoch_now=train_info['epoch_now'], lr_now=train_info['lr'],
                                          valid_loss=train_info['valid_loss_record'])
            self._write_report_counts(file, counts)
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
            self._write_report_parameters(file, self.n_layers, self.n_units,
                                          self.weight, self.init_lr, self.gamma,
                                          self.epoch, self.batch_size, self.features)
            file.write("\n**Summary**\n\n")
            self._write_report_counts(file, counts)
            file.write("\n**Reports for Each Fold**\n\n")
            for counts_each in counts_record:
                self._write_report_counts(file, counts_each)

    @staticmethod
    def _write_report_counts(file, counts):
        file.write(f"\n[Scores]\n")
        file.writelines([f"Height={ct_tp[0]}\n"
                         f"Precision:{ct_tp[1].precision:.5f} Recall:{ct_tp[1].recall:.5f} "
                         f"F-score:{ct_tp[1].fmeasure:.5f}\n"
                         f"TP:{ct_tp[1].tp} FP:{ct_tp[1].fp} FN:{ct_tp[1].fn}\n\n" for ct_tp in sorted(counts.items())])

    @staticmethod
    def _write_report_parameters(file, layer, unit, weight, init_lr, gamma, epoch, batch_size, features,
                                 epoch_now=0, lr_now=0., valid_loss=None):
        file.write("[Parameters]\n")
        file.write(f"structure: {unit} units x {layer} layers\n")
        file.write(f"weight for positive: {weight}\n")
        file.write(f"learning rate: {lr_now}/{init_lr}\n")
        file.write(f"scheduler gamma: {gamma}\n")
        file.write(f"no. of epochs: {epoch_now}/{epoch}\n")
        file.write(f"batch size: {batch_size}\n")
        file.write(f"Features: {features}\n")
        if valid_loss:
            file.write(f"Loss(valid): {valid_loss}\n")


def test_network_training():
    print(f"device: {networks.device}")
    print(f"cpu {CPU_CORES} cores")

    # configure
    boeck_set = datasets.BockSet()
    trainer = ModelManager(boeck_set, bidirectional=True)
    splits = trainer.boeck_set.splits
    # initialize
    trainer.features = DEFAULT_FEATURES
    trainer.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3))
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.5)
    trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        trainer.optimizer,
        milestones=[30, 230, 430], gamma=0.2)

    # training
    count, info = trainer.train_and_test(0, debug_max_epoch_count=1, verbose=True, batch_size=6)

    # testing
    test_output(trainer, splits[0][0])
    utils.plot_loss(info['valid_loss_record'])

    trainer.save()

    # print(hidden)


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


def test_saved_model(filename, features=None):
    boeck_set = datasets.BockSet()
    model = torch.load(filename, map_location=networks.device)
    mgr = ModelManager(boeck_set, features=features)
    mgr.model = model
    test_output(mgr, boeck_set.splits[0][0])
    counter = mgr.test_on_keys(boeck_set.get_split(0))
    print(f"Precision {counter.precision}, Recall {counter.recall}, F-score {counter.fmeasure}")


def test_prepare_data():
    boeck_set = datasets.BockSet()
    splits = boeck_set.splits
    for i in range(len(splits[0])):
        t0 = time.perf_counter()
        length, odfs, target, onsets_samples, audio_len = prepare_data(boeck_set, None, splits[0][i])
        t1 = time.perf_counter()
        print(f"{audio_len:.2f}s, elapsed {t1 - t0:.2f}, {audio_len / (t1 - t0):.1f}x speed")


def test_data_loader():
    batch_size = 8
    boeck_set = datasets.BockSet()
    mgr = ModelManager(boeck_set)
    training_set, _, _ = mgr.generate_splits(2)
    loader = BoeckDataLoader(boeck_set, training_set, batch_size)
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
    print("Task 1: RCD+SuperFlux")
    task_sf = TrainingTask(features=['rcd', 'superflux'])
    task_sf.train_and_test_8_fold(save_model=True)
    print("Task 2: SuperFlux (baseline)")
    task_sf = TrainingTask(features=['superflux'])
    task_sf.train_and_test_8_fold(save_model=True)
    print("Task 3: RCD+SuperFlux (8x2)")
    task_sf = TrainingTask(features=['rcd', 'superflux'], n_layers=2, n_units=8)
    task_sf.train_and_test_8_fold(save_model=True)

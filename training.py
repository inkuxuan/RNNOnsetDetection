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
import multiprocessing
import gc
import pathlib
import argparse

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

# ---------------META PARAMETERS----------------
# function reference for onset combiner
# in [onsets.combine_onsets_avg, onsets.merge_onsets, None]
# stands for combining onsets by average, by the first onset, and no combination
COMBINE_ONSETS = onset_utils.combine_onsets_avg

now = datetime.now()
dstr = now.strftime("%Y%m%d %H%M%S")
RUN_DIR = f'run {dstr}/'

PEAK_PICK_FUNCTION = onset_utils.peak_pick_static


# ---------------HELPER FUNCTIONS----------------

def init_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument_group('Task tags')
    parser.add_argument("--load-model-file", type=str,
                        help="Load a model from a specified saved checkpoint filepath. "
                             "If not specified, a new one is trained")
    parser.add_argument("--save-model-file", type=str,
                        help="Save the model checkpoint into a filename."
                             "Enter ONLY the filename. The file will be created in 'run xxx\\'."
                             "If not specified, the model WON'T be saved")
    parser.add_argument("--evaluate", action="store_true",
                        help="evaluate the trained or loaded model on the test set (see --test-set-index)")
    parser.add_argument("--input-audio", type=str, nargs="*",
                        help="If specified, one or more input audio file is used for onset detection")
    parser.add_argument("--speed-test", action="store_true",
                        help="Test the processing speed. Make sure if you want to use --cpu-only.")

    parser.add_argument_group('Global settings')
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--threads", type=int)

    parser.add_argument_group('Preprocessing settings')
    parser.add_argument("--no-cached-preprocessing", dest="cached_preprocessing", action="store_false",
                        help="If cache is enabled, all the odf preprocessing result will be cached in MEMORY,"
                             "so use this if you have less than 8GB of RAM and suffer from crashing")
    parser.add_argument("--feature", dest="features", type=str, nargs="+", required=True,
                        choices=['superflux', 'rcd'])
    parser.add_argument("--rcd-log", type=bool, default=True)
    parser.add_argument("--superflux-log", type=bool, default=True)

    parser.add_argument("--sampling-rate", type=int, default=44100)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=float, default=441)
    parser.add_argument("--superflux-lag", type=int, default=1)
    parser.add_argument("--no-normalize-wave", dest="normalize_wave", action="store_false")
    parser.add_argument("--no-normalize-odf", dest="normalize_odf", action="store_false")

    # args for training
    parser.add_argument_group('Training')
    parser.add_argument("--targeting-delta-frames", type=int, default=1)
    parser.add_argument("--onset-merge-interval", type=float, default=0.03,
                        help="This is used for both training and evaluation")
    parser.add_argument("--num-layer-units", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nonlinearity", type=str, default='tanh')
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--weight", type=float, default=1)
    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--early-stop-min-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model-filename", type=str)
    parser.add_argument("--no-revert-to-best", dest="revert_to_best", action="store_false")
    parser.add_argument("--test-set-index", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=5000)
    parser.add_argument("--save-checkpoint-to-file", dest="checkpoint_file", type=str,
                        help="Specify the filename for saving checkpoints at each epoch")

    # for peak-picking
    parser.add_argument_group('Peak-picking')
    parser.add_argument("--peak-picking-threshold", type=float, default=0.35)
    parser.add_argument("--peak-picking-smooth-window-second", type=float, default=0.05)

    # args for evaluation
    parser.add_argument_group('Evaluation')
    parser.add_argument("--peak-picking-threshold-lower", type=float, default=0.05)
    parser.add_argument("--peak-picking-threshold-upper", type=float, default=0.6)
    parser.add_argument("--peak-picking-threshold-step", type=float, default=0.05)
    parser.add_argument("--evaluation-window", type=float, default=0.025)
    parser.add_argument("--evaluation-delay", type=float, default=0., help="add delay to all detections (seconds")
    parser.add_argument("--no-concurrent-testing", dest="concurrent_test", action="store_false")

    # args for detection
    parser.add_argument_group('Detection')
    parser.add_argument("--save-audio-with-clicks", "-sc", action="store_true")
    parser.add_argument("--save-detections", "-sd", action="store_true")
    return parser.parse_args(argv)


def get_fps(args):
    return args.sampling_rate / float(args.hop_length)


def get_datetime_str():
    now = datetime.now()
    return now.strftime("%Y%m%d %H%M%S")


def save_args(filename, args):
    pathlib.Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR + filename, 'w') as file:
        file.write(str(args))

def save_training_report(train_info, args):
    pathlib.Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
    report_filename = args.save_model_file
    if report_filename is None:
        report_filename = get_datetime_str() + "_training_report"
    report_filename += ".txt"
    with open(RUN_DIR + report_filename, 'w') as file:
        file.write(str(train_info))

class Preprocessor:
    odfs_cache = {}
    features_cache = {}

    def __init__(self, boeck_set: datasets.BockSet, args):
        self.boeck_set = boeck_set
        self.args = args
        self.n_fft = args.n_fft
        self.hop_length = args.hop_length
        self.features = args.features
        self.sr = args.sampling_rate
        self.rcd_log = args.rcd_log
        self.superflux_log = args.superflux_log
        self.superflux_lag = args.superflux_lag
        self.normalize_wave = args.normalize_wave
        self.arg_normalize_odf = args.normalize_odf
        self.onset_delta = args.onset_merge_interval
        self.cached = args.cached_preprocessing
        self.target_delta = args.targeting_delta_frames

    def get_features(self, wave) -> np.ndarray:
        stft = onset_functions.stft(wave, n_fft=self.n_fft, hop_length=self.hop_length, online=False)
        f = []
        for feature in self.features:
            if feature in ['complex_domain', 'cd']:
                onset_strength = onset_functions.complex_domain_odf(stft, rectify=False,
                                                                    log=self.rcd_log)
                f.append(onset_strength)
            elif feature in ['rectified_complex_domain', 'rcd']:
                onset_strength = onset_functions.complex_domain_odf(stft, rectify=True,
                                                                    log=self.rcd_log)
                f.append(onset_strength)
            elif feature in ['super_flux', 'superflux']:
                onset_strength, _, _ = onset_functions.super_flux_odf(stft, self.sr,
                                                                      lag=self.superflux_lag,
                                                                      log=self.superflux_log)
                f.append(onset_strength)
        return np.asarray(f)

    def get_embedded_features(self, key):
        r"""
            Notice that odfs is in shape (length, n_features), target is in shape (length)

            :return: length(frames), onset_detection_function_values, target_values, onset_list_in_seconds, audio_length_sec
        """
        piece = self.boeck_set.get_piece(key)
        wave, onsets_list, sr = piece.get_data()
        if self.normalize_wave:
            wave = utils.normalize_wav(wave, type='float')
        if COMBINE_ONSETS:
            onsets_list = COMBINE_ONSETS(piece.get_onsets_seconds(), self.onset_delta)
        # convert from second to sample
        onsets_list = np.asarray(onsets_list) * sr
        # load from cache if available (check if feature type matches)
        if self.cached and (self.odfs_cache.get(key) is not None):
            # retrieve from cached (already transposed and normalised)
            odfs = self.odfs_cache[key]
        else:
            odfs = self.get_features(wave)
            # arrange dimensions so that the model can accept (shape==[seq_len, n_feature])
            odfs = odfs.T
            if self.arg_normalize_odf:
                # Normalize the odfs along each feature so that they range from 0 to 1
                odfs = self.normalize_odf(odfs)
            if self.cached:
                # save transposed odfs
                self.odfs_cache[key] = odfs
                # Prevent from memory overflowing
                gc.collect()
        length = odfs.shape[0]
        target = onset_utils.onset_list_to_target(onsets_list, self.hop_length, length, self.target_delta,
                                                  mode='linear')
        return length, odfs, target, onsets_list, len(wave) / sr

    @staticmethod
    def normalize_odf(odfs):
        r"""
        This normalize the second dimension in an nd-array to range [0,1]
        """
        for i in range(odfs.shape[1]):
            max_value = np.max(odfs[:, i])
            odfs[:, i] = odfs[:, i] / max_value
        return odfs


# ---------------CLASSES----------------


class BoeckDataLoader(object):
    r"""
    DataLoader (Not PyTorch DataLoader) for Boeck Dataset.
    Shuffle and minibatch implemented, concurrent ODF calculation within a batch.
    No. of threads correspond to CPU core count.
    """

    def __init__(self, boeck_set, keys, preprocessor: Preprocessor, args, batch_size=None):
        self.boeck_set = boeck_set
        self.keys = keys
        self.args = args
        self.batch_size = args.batch_size
        if batch_size:
            self.batch_size = batch_size
        self.shuffle = True
        if self.batch_size >= len(keys):
            self.shuffle = False
        self.features = args.features
        self.preprocessor = preprocessor

    def _shuffle(self):
        random.shuffle(self.keys)

    def generate_data(self):
        r"""
        Generator that yields (input_ndarray, target_ndarray, total_audio_length)
        Note that total length is in seconds.
        Input array shape: (batch_size, max_length, n_features)
        Target shape: (batch_size, max_length)
        max_length is the maximum length (frames) of all sequences in a batch
        """
        if self.shuffle and self.batch_size < len(self.keys):
            self._shuffle()
        index = 0
        data_size = len(self.keys)
        while index < data_size:
            # current batch size
            b_size = self.batch_size
            if index + b_size > data_size:
                b_size = data_size - index
            end_index = index + b_size
            keys = self.keys[index:end_index]

            # concurrent, prepare ODFs for every piece in the batch
            with ThreadPoolExecutor(max_workers=max([CPU_CORES, b_size])) as executor:
                results = executor.map(self.preprocessor.get_embedded_features, keys)
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

    def __init__(self, boeck_set, preprocessor: Preprocessor, args):
        self.boeck_set = boeck_set
        self.args = args
        self.features = args.features
        self.sr = args.sampling_rate
        self.hop_length = args.hop_length
        self.preprocessor = preprocessor
        self.test_split_index = args.test_set_index
        if args.load_model_file:
            self.load_model_file = args.load_model_file
            self.load_cp(args.load_model_file)
        else:
            self.model = networks.SingleOutRNN(
                len(args.features),
                args.num_layer_units,
                args.num_layers,
                nonlinearity=args.nonlinearity,
                bidirectional=args.bidirectional,
                sigmoid=False
            ).to(networks.device)
        self.loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor(args.weight, device=networks.device))
        optimizer_name = args.optimizer
        lr = args.lr
        self.early_stop_patience = args.early_stop_patience
        self.early_stop_min_epochs = args.early_stop_min_epochs
        self.max_epochs = args.max_epochs
        self.checkpoint_file = args.checkpoint_file
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # compatibility
        self.scheduler = None
        self.revert_to_best = args.revert_to_best

    def initialize_model(self):
        self.model.init_normal()

    def save_model_legacy(self, filename=None):
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
        if filename is None:
            dstr = get_datetime_str()
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

    def load_model_state_dict(self, state_dict):
        # used in best reverting
        self.model.load_state_dict(state_dict)

    def load_legacy(self, path):
        r"""
        This WILL NOT re-initialize the optimizer and the scheduler.
        Use the constructor instead if wish to train the network.
        """
        self.model = torch.load(path, map_location=networks.device)

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
                _, odfs, _, _, audio_len = self.preprocessor.get_embedded_features(key)
            elif wave is not None:
                wave = utils.normalize_wav(wave)
                odfs = self.preprocessor.get_features(wave)
                audio_len = len(wave) / float(self.args.sampling_rate)
                odfs = odfs.T
                odfs = Preprocessor.normalize_odf(odfs)
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
            out = out.squeeze()
            out = out.cpu()
        return out

    def train_on_split(self,
                       training_keys,
                       validation_keys,
                       verbose=True,
                       callback=None,
                       start_epoch=0,
                       **kwargs):
        r"""
        Train the network on a specific training set and validation set

        -----
        :param validation_keys: list of keys in validation set
        :param training_keys: list of keys in training set
        Training process uses concurrency for both pre-processing and training. This is default the cores count of CPU.
        :param verbose: bool, Print debug outputs
        :param callback: function, callback function to be called after each epoch
            callback(model_manager=, epoch=, valid_loss_record=, best_state_dict=, best_valid_loss=, **kwargs)
        :return: dict containing {loss_record, valid_loss_record, epoch_now, lr, best_state_dict}
        """
        loader = BoeckDataLoader(self.boeck_set, training_keys, self.preprocessor, self.args)
        valid_loader = BoeckDataLoader(self.boeck_set, validation_keys, self.preprocessor, self.args,
                                       batch_size=len(validation_keys))
        continue_epoch = True
        epoch_now = start_epoch
        loss_record = []
        valid_loss_record = []
        best_valid_loss = float('inf')
        best_state_dict = self.model.state_dict()
        early_stopping = False
        if self.early_stop_patience > 0:
            early_stopping = utils.EarlyStopping(patience=self.early_stop_patience,
                                                 min_epoch=self.early_stop_min_epochs)
        while continue_epoch:
            epoch_now += 1
            if self.max_epochs and epoch_now >= self.max_epochs:
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
                if self.checkpoint_file:
                    if isinstance(self.checkpoint_file, str):
                        self.save_cp(self.checkpoint_file)
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
        if self.revert_to_best:
            self.load_model_state_dict(best_state_dict)
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

    def train_only(self, verbose=True, **kwargs):
        r"""
        :return: dict containing {loss_record, valid_loss_record, epoch_now, lr, best_state_dict}
        """
        training_keys, validation_keys, test_keys = self.boeck_set.generate_splits(self.test_split_index)
        info = self.train_on_split(training_keys, validation_keys, verbose=verbose, **kwargs)
        return info

class PeakPicker:
    def __init__(self, args):
        self.args = args
        self.threshold = args.peak_picking_threshold
        self.smooth_window_sec = args.peak_picking_smooth_window_second
        self.fps = get_fps(args)
        self.smooth_window_frames = int(self.fps * self.smooth_window_sec)

    def peak_pick(self, signal, threshold_override=None):
        """returns peaks in seconds"""
        threshold = threshold_override if threshold_override else self.threshold
        (peaks, heights) = onset_utils.peak_pick_static(signal, threshold, smooth_window=self.smooth_window_frames)
        peaks = np.array(peaks) / self.fps
        return peaks


# noinspection PyUnresolvedReferences
class TrainingTask(object):
    def __init__(self,
                 trainer: ModelManager,
                 args):
        self.args = args
        self.dynamic_thresholds = False
        t_lower = args.peak_picking_threshold_lower
        t_upper = args.peak_picking_threshold_upper
        t_step = args.peak_picking_threshold_step
        self.threshold_step = t_step
        self.min_threshold_step = 0.025
        self.thresholds = np.arange(t_lower, t_upper, t_step)
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
        print(f"Thresholds: {self.thresholds}")
        # concurrency is implemented by test tasks (trainer.test_on_key)
        for threshold in self.thresholds:
            count = self.trainer.test_only(test_set_index, threshold, **kwargs)
            counts[threshold] = count

        # binary search for best lambda
        if self.thresholds:
            while self.threshold_step > self.min_threshold_step:
                self.threshold_step = self.threshold_step * 0.5
                print(f"Threshold step: {self.threshold_step}")
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
                start_threshold = counts_tuples[start_index][0]
                stop_threshold = counts_tuples[stop_index][0]
                new_thresholds = np.arange(start_threshold, stop_threshold, self.threshold_step)
                # avoid duplicates
                new_thresholds = np.setdiff1d(new_thresholds, self.thresholds)
                # save all lambdas for future use in 8-fold cross validation
                # self.lambdas += new_lambdas
                self.thresholds = np.concatenate((self.thresholds, new_thresholds))
                self.thresholds = np.sort(self.thresholds)
                print(f"New thresholds: {new_thresholds}")
                for threshold in new_thresholds:
                    count = self.trainer.test_only(test_set_index, threshold, **kwargs)
                    counts[threshold] = count

        t1 = time.perf_counter()
        print(f"Evaluation done. Time elapsed {t1 - t0:.2f}s")
        # report file
        pathlib.Path(self.report_dir).mkdir(parents=True, exist_ok=True)
        with open(self.report_dir + report_filename, 'w') as file:
            file.write("Training and Test Report\n")
            self._write_report_parameters(file, self.args,
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
            train_info = self.trainer.train_only(callback=self.middle_test, test_every_epoch_no=test_every_epoch_no,
                                                 **kwargs)
            self.trainer.model.eval()

        else:
            train_info = {}
        # testing
        print("Saving and plotting...")
        if show_example_plot:
            pass
            # test_output(self.trainer, splits[test_set_index][0])
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
                self._write_report_parameters(file, self.args)
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
    def _write_report_parameters(file, args,
                                 epoch_now=0, lr_now=0., valid_loss=None):
        file.write("[Parameters]\n")
        file.write(f"{vars(args)}\n")
        file.write(f"last learning rate: {lr_now}\n")
        file.write(f"epochs: {epoch_now}\n")
        if valid_loss:
            file.write(f"Loss(valid): {valid_loss}\n")


class ModelEvaluator:
    def __init__(self, boeck_set: datasets.BockSet, peak_picker: PeakPicker, args):
        self.boeck_set = boeck_set
        self.peak_picker = peak_picker
        self.args = args

    def evaluate_model(self, model: ModelManager):
        thresholds = np.arange(self.args.peak_picking_threshold_lower,
                               self.args.peak_picking_threshold_upper,
                               self.args.peak_picking_threshold_step)
        test_set_index = self.args.test_set_index
        test_set_keys = self.boeck_set.get_split(test_set_index)
        concurrent_test = self.args.concurrent_test
        # evaluate model using args
        dict_threshold_count = {}
        for threshold in thresholds:
            dict_threshold_count[threshold] = self._test_on_keys(model, test_set_keys, threshold, concurrent_test)
        # save report file
        report_filename = self.args.load_model_file
        if report_filename is None:
            report_filename = self.args.save_model_file
        if report_filename is None:
            report_filename = get_datetime_str() + "_eval_report"
        report_filename += ".txt"
        self._write_evaluation_results(dict_threshold_count, report_filename)
        pass

    def _write_evaluation_results(self, dict_threshold_count, filename):
        pathlib.Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
        with open(RUN_DIR + filename, 'w') as file:
            file.write("\n[Evaluation Report]\n")
            file.write("\n[Command Line Arguments]\n")
            file.write(str(self.args))
            file.write("\n[Scores]\n")
            file.writelines([f"Height(Lambda)={ct_tp[0]}\n"
                             f"Precision:{ct_tp[1].precision:.5f} Recall:{ct_tp[1].recall:.5f} "
                             f"F-score:{ct_tp[1].fmeasure:.5f}\n"
                             f"TP:{ct_tp[1].tp} FP:{ct_tp[1].fp} FN:{ct_tp[1].fn}\n\n" for ct_tp in
                             sorted(dict_threshold_count.items())])

    def _test_on_keys(self, model, keys, threshold, concurrent=True):
        count = Counter()
        if concurrent:
            with ThreadPoolExecutor(max_workers=max([CPU_CORES, len(keys)])) as executor:
                futures = []
                for key in keys:
                    future = executor.submit(self._test_on_key, model, key, threshold)
                    futures.append(future)
                for future in futures:
                    count += future.result()
        else:
            for key in keys:
                count1 = self._test_on_key(model, key, threshold)
                count += count1
        return count

    def _test_on_key(self, model, key, threshold):
        ground_truth = self.boeck_set.get_piece(key).get_onset_seconds()
        signal = model.predict(key=key)
        detections = self.peak_picker.peak_pick(signal, threshold_override=threshold)
        if COMBINE_ONSETS:
            ground_truth = COMBINE_ONSETS(ground_truth, self.args.onset_merge_interval)
            detections = COMBINE_ONSETS(detections, self.args.onset_merge_interval)
        count = boeck.onset_evaluation.count_errors(detections, ground_truth,
                                                    self.args.evaluation_window,
                                                    delay=self.args.evaluation_delay)
        return count

    def speed_test(self, model: ModelManager):
        # speed test on the test set
        test_set_index = self.args.test_set_index
        test_set_keys = self.boeck_set.get_split(test_set_index)
        total_len = 0.
        for key in test_set_keys:
            samples = len(self.boeck_set.get_piece(key).get_wave())
            length = samples / self.args.sampling_rate
            total_len += length
        t1 = time.perf_counter()
        # giving None as threshold makes peak-picker uses the threshold specified in args
        self._test_on_keys(model, test_set_keys, None, concurrent=False)
        t2 = time.perf_counter()
        speed = total_len / (t2 - t1)
        print(f'Speed test result: {speed}x')
        return speed


# ---------------ENTRIES---------------

def main(argv):
    args = init_args(argv)
    print(f"Detected CPU Cores: {CPU_CORES}")
    if args.threads:
        print(f"Overriding thread numbers: {args.threads}")
        global CPU_CORES
        CPU_CORES = args.threads
    print("Loading Dataset")
    boeck_set = datasets.BockSet(args.sampling_rate)
    if args.cpu_only:
        print("[WARNING] using CPU only")
        networks.device = torch.device('cpu')
    preprocessor = Preprocessor(boeck_set, args)
    # load or train a model
    print(f"Creating model on device: {networks.device}")
    model = ModelManager(boeck_set, preprocessor, args)
    if args.load_model_file:
        print("Loading weights from file")
        model.load_cp(args.load_model_file)
    else:
        print("Training model")
        info = model.train_only()
        save_training_report(info, args)
    if args.save_model_file:
        print(f"Saving weights and args to {args.save_model_file}")
        model.save_cp(args.save_model_file)
        save_args(args.save_model_file + ".args", args)

    peak_picker = PeakPicker(args)
    if args.input_audio:
        raise NotImplementedError()

    evaluator = ModelEvaluator(boeck_set, peak_picker, args)
    if args.speed_test:
        print("Performing Speed Test")
        evaluator.speed_test(model)
    if args.evaluate:
        print("Evaluating model")
        evaluator.evaluate_model(model)


def evalutate_8f_saved(model_file_template, features, lambdas, filename="Report.txt",
                       load_as_cp=True, set_cached=True):
    pass


if __name__ == '__main__':
    main(None)
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

    # mc_spnn = ModelConfig(features=['superflux'])
    # mc_rsnn = ModelConfig(features=['rcd', 'superflux'])
    # mc_rsnn_2_8 = ModelConfig(features=['rcd', 'superflux'], num_layer_unit=8)
    # mc_rsnn_3_4 = ModelConfig(features=['rcd', 'superflux'], num_layers=3)
    #
    # speed_test(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task1-d0.pt",
    #            mc_spnn,
    #            lambda_=0.15, smooth=0)
    # speed_test(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task1-d0.pt",
    #            mc_spnn,
    #            lambda_=0.25, smooth=0.05)
    # speed_test(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task2-d2.pt",
    #            mc_rsnn,
    #            lambda_=0.3, smooth=0)
    # speed_test(r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220415 - Final tune\task2-d1.pt",
    #            mc_rsnn,
    #            lambda_=0.5, smooth=0.05)
    # speed_test(
    #     r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220522 8-f\Model_rcd-spf-fold-0-delta-1-2x8.pt",
    #     mc_rsnn_2_8,
    #     lambda_=0.45, smooth=0.05)
    # speed_test(
    #     r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220522 8-f\Model_rcd-spf-fold-0-delta-1-2x8.pt",
    #     mc_rsnn_2_8,
    #     lambda_=0.3, smooth=0)
    # speed_test(
    #     r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220522 8-f\Model_rcd-spf-fold-0-delta-1-3x4.pt",
    #     mc_rsnn_3_4,
    #     lambda_=0.5, smooth=0.05)
    # speed_test(
    #     r"E:\Documents\Personal\WorkSpace\Research\ComplexDODF\run 20220522 8-f\Model_rcd-spf-fold-0-delta-1-3x4.pt",
    #     mc_rsnn_3_4,
    #     lambda_=0.3, smooth=0)
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

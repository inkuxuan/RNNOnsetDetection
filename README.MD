# RNN Onset Detector
Python version `3.9` is used for this project

## Out-of-the-box Commands

### Analyze Audio

To analyze audio, first load a model and specify the features (ODFs) and network configurations.
Then specify input audios (can be multiple files separated by a space). 
The output will be two files -- a .txt file with all onset timestamps one per line like
```
0.15
0.36
0.45
0.61
```
and a .wav file with onset clicks on the original audio (for sonic evaluation).

`python training.py --load-model-file best_model.pt --feature superflux wpd --num-layer-units 4 --num-layers 3 --input-audio "./demo_audio_content/ENERGY SYNERGY MATRIX.wav"`

### Training a Model

To train a model, specify the training parameters by providing features (ODFs), network layout,
 targeting delta (in frames), and specify a filename to save the model. If you also need to evaluate 
the model, use `--evaluate`

`python training.py --evaluate --feature superflux wpd --num-layer-units 4 --num-layers 3 --targeting-delta-frame 2 --model-filename saved_model.pt`

The model weights and the command line args will be saved as separate files for later reference.
The training record will also be saved as a text file.

### Load and Evaluate a Saved Model

You still need to specify features and network layout when loading a model.
`python training.py --evaluate --load-model-file saved_model.pt --feature superflux wpd --num-layer-units 4 --num-layers 3`

## Deployment
* Prepare the Boeck dataset in `../Datasets/Boeck/`
* Install [PyTorch](https://pytorch.org/get-started/locally/) (1.9+Optional CUDA) for your platform 
* Install requirements by `pip3 install -r requirements.txt`
* run `training.py`

## Arguments

See `python training.py --help`

```
  -h, --help            show this help message and exit
  --load-model-file LOAD_MODEL_FILE
                        Load a model from a specified saved checkpoint
                        filepath. If not specified, a new one is trained
  --save-model-file SAVE_MODEL_FILE
                        Save the model checkpoint into a filename.Enter ONLY
                        the filename. The file will be created in 'run
                        xxx\'.If not specified, the model WON'T be saved
  --evaluate            evaluate the trained or loaded model on the test set
                        (see --test-set-index)
  --input-audio [INPUT_AUDIO ...]
                        If specified, one or more input audio file is used for
                        onset detection
  --speed-test          Test the processing speed of the current setup.
                        Results will only be shown in console. Make sure if
                        you want to use --cpu-only. Note: Speed test evaluates
                        the test set one by one (no concurrency).
  --cpu-only            Use CPU instead of GPU
  --threads THREADS     Manually set the number of threads for concurrency
  --no-cached-preprocessing
                        If cache is enabled, all the odf preprocessing result
                        will be cached in MEMORY,so use this if you have less
                        than 8GB of RAM and suffer from crashing
  --feature {superflux,rcd,wpd} [{superflux,rcd,wpd} ...]
                        List of features (ODFs)
  --rcd-log RCD_LOG
  --superflux-log SUPERFLUX_LOG
  --sampling-rate SAMPLING_RATE
  --n-fft N_FFT         The size of the FFT window
  --hop-length HOP_LENGTH
  --superflux-lag SUPERFLUX_LAG
                        The order to which the difference is calculated for
                        SuperFlux
  --no-normalize-wave
  --no-normalize-odf
  --targeting-delta-frames TARGETING_DELTA_FRAMES
                        The delta value for triangular targeting, in
                        frames.Default 1. Set 0 for single targeting
  --onset-merge-interval ONSET_MERGE_INTERVAL
                        In interval within which onsets are merged into one,
                        in seconds. This is used for both training and
                        evaluation. Default 0.03.
  --num-layer-units NUM_LAYER_UNITS
                        The number of units in each hidden layer. Default 4
  --num-layers NUM_LAYERS
                        The number of hidden layers. Default 2.
  --nonlinearity NONLINEARITY
                        The activation function of hidden layers. Default tanh
  --bidirectional BIDIRECTIONAL
  --weight WEIGHT
  --optimizer {adam,sgd}
  --lr LR
  --early-stop-patience EARLY_STOP_PATIENCE
  --early-stop-min-epochs EARLY_STOP_MIN_EPOCHS
  --batch-size BATCH_SIZE
  --model-filename MODEL_FILENAME
                        The filename to save the model used in the current
                        run.
  --no-revert-to-best   Use this arg if you do not want the training process
                        to revert to the best weights after stopping the
                        training.
  --test-set-index TEST_SET_INDEX
                        The index of the test set of Boeck Dataset. Default 0.
  --max-epochs MAX_EPOCHS
  --save-checkpoint-to-file CHECKPOINT_FILE
                        Specify the filename for saving checkpoints at each
                        epoch. This lowers the performance of training.
  --peak-picking-threshold PEAK_PICKING_THRESHOLD
                        The threshold used for final peak-picking.This is used
                        for analyzing audio, instead of evaluating the model.
                        Default 0.35.
  --peak-picking-smooth-window-second PEAK_PICKING_SMOOTH_WINDOW_SECOND
                        The length of the smoothing window for peak-picking in
                        seconds. Default 0.05.
  --peak-picking-threshold-lower PEAK_PICKING_THRESHOLD_LOWER
                        For evaluating the model, the lower boundary of a best
                        threshold search. Default 0.05.
  --peak-picking-threshold-upper PEAK_PICKING_THRESHOLD_UPPER
                        For evaluating the model, the upper boundary of a best
                        threshold search. Default 0.8.
  --peak-picking-threshold-step PEAK_PICKING_THRESHOLD_STEP
                        For evaluating the model, search step of a best
                        threshold search. Default 0.05.
  --evaluation-window EVALUATION_WINDOW
                        The window within which an onset candidate is counted
                        as a true positive, in seconds. Default 0.025.
  --evaluation-delay EVALUATION_DELAY
                        Add an delay to all detections when evaluating
                        (seconds). Default 0.
  --no-concurrent-testing
                        For test purpose, disable multi-threading.
Process finished with exit code 0

```

## Core Components

### `training.py` 
Entrance of the program. Contains manipulations of the model, i/o processing, args parsing.

### `networks.py` 
Definitions of the RNN model and some pytorch settings.

### `onset_functions.py`
Calculation of onset detection functions. Includes Wave to STFT frames, and tonal filters.

### `onset_utils.py`
Helper functions for onset manipulation and peak-picking.

### `datasets.py`
Definitions of datasets.

### `utils.py`
Utilities.
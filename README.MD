# RNN Onset Detector
Python version `3.10` is used for this project

## Deployment
* Prepare the Boeck dataset in `../Datasets/Boeck/`
* Install [PyTorch](https://pytorch.org/get-started/locally/) (1.9+Optional CUDA) for your platform 
* Install requirements by `pip3 install -r requirements.txt`
* run `training.py`

## Parameters

### Examples

`python training.py --save-model-file <filename> --evaluate --feature superflux rcd --cpu-only`

`python training.py --load-model-file <filename> --evaluate --feature superflus rcd --cpu-only`

## Components

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
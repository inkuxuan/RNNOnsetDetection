# RNN Onset Detector
Python version `3.10` is used for this project
## Deployment
* Prepare the Boeck dataset in `../Datasets/Boeck/`
* Install [PyTorch](https://pytorch.org/get-started/locally/) (1.9+CUDA) for your platform 
* Install requirements by `pip3 install -r requirements.txt`
* run `training.py`

## Parameters

### Frequently used parameters

`--save-model-file <filename> --evaluate --feature superflux rcd --cpu-only`

`--load-model-file <filename> --evaluate --feature superflus rcd --cpu-only`
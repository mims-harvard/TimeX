# TimeSeriesCBM
Practical Concept Bottleneck Model for Time Series

## Installation
You'll need to locally install a reference to the `txai` package, which contains commonly-used utilities, model architectures, and training procedures. To install, navigate to the base directory `*/TimeSeriesCBM/` and run:
```
python -m pip install -e .
```
This should install the references that you need to run scripts in the `experiments` directory.

## Datasets

Instructions for preparing datsets

### CogPilot
 
 1. Login to [PhysioNet](https://physionet.org/)
 2. Approve data use agreement for [Multimodal Physiological Monitoring During
    Virtual Reality Piloting Tasks](https://physionet.org/content/virtual-reality-piloting/1.0.0/)
 3. Run `mkdir -p datasets/downloads`
 4. Run `wget --user <PhysioNet Username> --askpassword https://physionet.org/content/virtual-reality-piloting/get-zip/1.0.0/ -O datasets/downloads/cogpilot.zip` and enter your PhysioNet password when prompted.
 5. Run `unzip datasets/downloads/cogpilot.zip -d datasets/downloads/cogpilot`
 6. Run `python preprocessing/cogpilot.py`

 


### Baseline notes: WinIT

Train generator models with:

```bash
python experiments/evaluation/winit_wrapper.py \
    --dataset scs_better \
    --data_path <path to dataset directory> \
    --models_path <path to models directory>
```

Evaluate:

```bash
python experiments/evaluation/saliency_exp_synth.py \
    --dataset scs_better \
    --data_path <path to to dataset directory>  \
    --model_path <path to models>/Scomb_transformer_split\=1_cpu.pt \
    --exp_method winit
```

# TimeX: Encoding Time-Series Explanations through Self-Supervised Model Behavior Consistency



## Overview
TimeX is a novel time series explainer that explains time series classification models through learning an interpretable surrogate model. 
This interpretable surrogate model learns an explanation embedding space that is optimized to have similar structure to that of the original reference model.
To learn this explanation embedding space, we design a novel training objective, *model behavior consistency*, which trains the model to capture similar relationships between samples represented by the reference model. 
This explanation embedding space allows us to learn *landmark explanations*, which are explanations that are representative of broader predictive patterns in the entire dataset.
Landmark explanations allow users to compare explanations across multiple samples, an important component for time series as predictive patterns are harder to intuitively understand through visual inspection of the time series sample.
We demonstrate TimeX's capabilities on four novel synthetic datasets and four real-world time series datasets. 

![Architecture](https://github.com/mims-harvard/TimeX/blob/main/fig2.jpg)

## Installation
You'll need to locally install a reference to the `txai` package, which contains commonly-used utilities, model architectures, and training procedures. To install, navigate to the base directory `*/TimeX/` and run:
```
python -m pip install -e .
```
This should install the references that you need to run scripts in the `experiments` directory.

**Requirements**: Requirements are found in `requirements.txt`. Please install the necessary requirements via `pip` (recommended) or `conda`.

## Datasets and Model Weights

Processed datasets and model weights will be released upon acceptance.

## Usage

**Example**: Detailed examples of the model are given in the experiment scripts found in `experiments` directory. A good reference for a real-world dataset is given for the Epilepsy dataset in `experiments/epilepsy/bc_model_ptype.py`.

**Locations of important implementations**: The TimeX model can be found in `txai/models/bc_model.py` under the name `BCExplainModel`. The novel model behavior consistency (MBC) loss is found in `txai/utils/predictors/loss_cl.py`.

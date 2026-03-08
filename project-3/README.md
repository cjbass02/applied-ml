# Project 3 KMNIST Experiments

This folder contains a reusable Keras experiment runner for the KMNIST image classification project.

## What the script does

`kmnist_experiments.py` will:

- load the KMNIST train/test `.npz` files from `project-3/data/`
- save a 3-by-4 sample visualization using 4 classes and 3 examples per class
- reshape images to `(N, 28, 28, 1)` and scale pixel values to `[0, 1]`
- run a grid of MLP experiments
- pick the best MLP configuration by validation accuracy
- run a grid of CNN experiments using the best MLP dense-layer widths
- save model summaries, histories, confusion matrices, and evaluation metrics

## Key modeling choices

- This is a `multiclass` classification problem because there are `10` possible character classes.
- The output layer uses `10` nodes with `softmax` so the model returns one probability per class.
- The loss function is `sparse_categorical_crossentropy`, which is appropriate for integer-coded multiclass labels.
- Accuracy is used as the main metric, matching the project instructions.

## Run it

From the repository root:

```powershell
python project-3/kmnist_experiments.py
```

## Saved outputs

After a successful run, outputs will be written to `project-3/outputs/`:

- `figures/`: sample grid, confusion matrices, training-history plots
- `metrics/`: experiment result tables, dataset summary, per-model evaluation JSON, history CSV files
- `models/`: saved `.keras` models
- `summaries/`: model architecture summaries
- `run_summary.txt`: short text summary of the experiment run

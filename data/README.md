# Training Dataset

The full `emi_prediction_dataset.csv` is intentionally not included in this lightweight deployment repository. It is approximately 75 MB and is only needed to rerun the training notebook.

To reproduce training, place the original dataset in this directory with the exact filename:

```text
emi_prediction_dataset.csv
```

The Streamlit dashboard does not require this file. It uses `dashboard_sample.csv` and the exported model bundle.

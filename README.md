# Stimalyzer

Streamlit application for reviewing and annotating motion captured by two x-IMU3 sensors. It synchronizes recordings from the wrist and lower back, highlights high-activity intervals and helps an annotator export consistent labels for a later machine-learning workflow.

> [!IMPORTANT]
> Stimalyzer is a research prototype. Its suggestions indicate dynamic movement only; they do not identify stimming and must not be used to diagnose autism spectrum disorder.

## Features

- synchronized analysis of wrist and lower-back `Inertial.csv` recordings;
- interactive accelerometer, gyroscope and combined-motion plots;
- adjustable suggestions for intervals worth reviewing;
- manual annotation with validation and neutral-gap completion;
- export of an `annotations.csv` file;
- scripts for converting labelled sessions into window-level training data;
- an example notebook for training and evaluating a Random Forest model.

## Tech stack

Python, Streamlit, pandas, NumPy, Plotly and scikit-learn.

## Getting started

Python 3.11 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload two `Inertial.csv` files from the same recording session: one from the wrist and one from the lower back. If the sensors were not started simultaneously, enter the timestamp of the same visible movement in both recordings to align them.

## Preparing a training dataset

Copy the example files before entering local paths or research annotations:

```powershell
Copy-Item data/manifest_template.csv data/manifest.csv
Copy-Item data/annotations_template.csv data/annotations.csv
```

Each manifest row describes one paired-sensor session. Each annotation row describes one labelled time interval. When the whole session has been reviewed, set `fully_annotated` to `TRUE`; only then can unlabelled windows safely be treated as background.

Build the window-level feature table with:

```powershell
python scripts/build_training_dataset.py `
  --manifest data/manifest.csv `
  --annotations data/annotations.csv `
  --output data/training_windows.csv
```

Split training and test sets by `participant_id`, never by individual windows, to prevent data from the same person leaking into both sets.

## Project structure

```text
app.py                     Streamlit interface
scripts/imu_pipeline.py    signal loading, synchronization and detection
scripts/build_training_dataset.py
                           feature extraction for model training
notebooks/                 model-training walkthrough
data/*_template.csv        safe input-file templates
style.css                  application styling
```

Raw sensor recordings, participant annotations, trained models, reports and local Python environments are intentionally excluded from version control.

## Privacy and responsible use

Motion traces and annotations may be sensitive research data. Keep participant identifiers pseudonymous, obtain appropriate consent, and do not commit local datasets or exported annotations to a public repository.

## License

No open-source license has been selected yet. All rights are reserved by the author.

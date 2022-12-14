# `e_muscae_summit_classifier`

> Random Forest Classifier for calling summiting in flies infected with *E. muscae*
> 
> Danylo Lavrentovich, Carolyn Elya
>
> de Bivort lab 2022


## Table of Contents

- [Installation](#installation)
- [Files](#files)
- [Running the classifier](#running-the-classifier)


## Installation

### Clone

Clone this repository to your machine:

```
$ git clone https://github.com/dlavrent/e_muscae_summit_classifier`
```

### Setup

Install Anaconda and run the following line to install a Python 3.6 environment `zombie_classifier`, that houses all necessary packages for running the classifier:

```
$ conda env create -f classifier_environment.yml
```

## Files

### Directory Structure
```bash
.
├── input_files
│   └── input_file_example.py
├── models
│   ├── clf_used_in_experiments.p
│   └── params_used_in_experiments.p
├── training
│   ├── data_processing_overview.ipynb
│   ├── summary_table_experiments.csv
│   ├── test_classification.ipynb
│   └── train_random_forest.ipynb
├── utils
│   ├── __init__.py
│   ├── classifier_plot_utils.py
│   ├── email_utils.py
│   ├── gmail_account_info.py
│   ├── load_data_utils.py
│   ├── plot_trajs_with_class_probs.py
│   ├── prediction_utils.py
│   ├── process_data_utils.py
│   └── time_utils.py
├── .gitignore
├── README.md
├── classifier_environment.yml
└── run_classifier.py
```

### Contents

#### `input_files/`

- Contains Python files that are read in by `run_classifier.py` each time a classifier experiment is run. The input file defines a Python dictionary containing information such as the name of the output directory for the experiment, the location of the tracking data to classify, whether to classify in real-time or post-factum, and whether/to whom to send email alerts
- This directory includes an example file, `input_file_example.py`, that classifies an existing tracking dataset under the same parameters as those done in all summiting experiments (classifying every 4 minutes), and it does not send an email to anyone

#### `models/`
- Houses files built from `training/train_random_forest.ipynb` necessary for classification:
	- `clf` is a `scikit-learn` `RandomForestClassifier` object that is trained on existing data and is used to predict new data
	- `params` is a dictionary with relevant information for training and generating `clf`
- This directory includes a pair of files, `clf_used_in_experiments.p` and `params_used_in_experiments.p`, that were generated from `training/train_random_forest.ipynb`, used in classifier runs for all experiments

#### `training/`
- `data_processing_overview.ipynb` outlines feature selection and what choices were made for building the classifier
- `train_random_forest.ipynb` reads in collected experiments, splits data into training/validation sets, trains a Random Forest classifier and evaluates on the validation set, outputting trained model files into `models/`
- `test_classification.ipynb` takes a trained `RandomForestClassifier` object `clf` and a held-out tracking experiment and simulates classification, to compare different summiting call strategies
- `summary_table_experiments.csv` outlines information on collected experiments (time, number of summiting/non-summiting flies, etc.) used for training/validation/testing

#### `utils/`
- Necessary utility/helper functions used throughout the repository

#### `classifier_environment.yml` 
- Houses all necessary packages/dependencies for all the code in the repository in an Anaconda environment

#### `run_classifier.py`
- The real-time classifier code! 
- Reads in centroid and time binary files outputted from margo, uses a trained `clf` and a specified summit calling rule to classify the streaming data


## Running the classifier
The master file for running the classifier is `run_classifier.py`, which has two critical inputs:
- files stored in `models/` that correspond to a trained model (generated by running `training/train_random_forest.ipynb`)
- an input file stored in `input_files/` that is designed to be specific to a particular run of the classifier, which specifies the filepath of the tracking data to classify, the filepath of the classifier output directory, the filepath of the trained classifier model, classifier settings such as frequency, and email addresses for sending summiting notifications

With a trained model in place, the intended workflow for a classification experiment is to create an input file with details specific to the experiment (notably, pointing to which tracking data to classify, which can be collected in real-time), and then to specify that input file within `run_classifier.py`:

```
# IMPORT INPUT FILE HERE
from input_files.input_file_example import input_d
```

Then, the classifier code can be run with the following code in Anaconda Prompt/terminal:
```
python run_classifier.py
```

An output directory will be created according to what is specified in the input file.


### Example run

Running the classifier using the provided `input_files/input_file_example.py` classifies the tracking data in  `11-05-2019-18-38-20__Circadian_CsWF-BoardC10_MF_Emuscae_1-128_Day3`, performing classification every 4 minutes, without sending any notification emails. A summiter fly is found at 33:35:59 (9:35:59 local time):
```
    Processing ROIs at (33:55:59):
    Elapsed 2.617 sec: data loaded, last binary file time 37.638
    Elapsed 3.290 sec: ROIs converted to features and plotted
    Elapsed 3.304 sec: classification done
    Elapsed 3.374 sec: 1 ROIs called summiting
            !~!~!~!~! PREDICTED SUMMITING: ROI 6
                    !predicted non-summiter: ROI 24, score: 0.946
    Elapsed 6.960 sec: summiting ROIs plotted
    Elapsed 6.961 sec: not emailed
    Elapsed 6.962 sec: done with frame
    ROIs processed.
```
---

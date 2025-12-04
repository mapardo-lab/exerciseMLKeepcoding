# Summary 

This work leverages available Airbnb data to develop a machine learning model for predicting prices of new accommodations. The model is usefull when a client has a property with specific features and a particular location and wants to determine an appropiate price at which to offer the accommodation.

After evaluating several classical machine learning models using a protocol based on Optuna library, a model was selected for production use in predicting accommodation prices.

A Support Vector Machine model was trained, achieving a median absolute error of 13.78 and showing low overfitting. This model was also used to predict prices for accommodations with missing price data.

Detailed results are available in the `notebooks` directory (see **exerciseML.pdf** or **exerciseML.html**)

# Methodology

Hyperparameter optimization was performed using the **run_optuna_ML.py** script, which executes the optimization procedures defined in the configuration files stored in the `optuna/run_config` directory.

The methodology was designed to streamline the optimization process by enabling easy configuration of models, search spaces, and scoring metrics. All optimization results were saved in the `optuna/file_config` directory for use in subsequent steps. These optimized parameters were then utilized in the model training phase (**training_model_ML.py**) to build production-ready models. The trained models can subsequently be deployed for inference using the **model_prediction_ML.py** script. This comprehensive protocol ensures full traceability across all studies and experimental runs.

# Code Structure


```
.
├── data
├── environment.yaml
├── models
├── notebooks
├── optuna
│   ├── file_config
│   ├── run_config
|   └── optuna_ML_exercise.db
├── README.md
└── src
    ├── model_prediction_ML.py
    ├── run_optuna_ML.py
    ├── training_model_ML.py
    ├── utilsDataset.py
    ├── utilsFT.py
    ├── utilsModel.py
    ├── utilsOptuna.py
    ├── utilsPlots.py
    ├── utilsPreproc.py
    └── utils.py

```

# Reproducibility

Create an environment with conda (version 25.1.1)

```
conda env create -f environtment.yaml
conda activate exerciseML
```

# Create HTML and PDF files

Quarto version 1.8.24

```
quarto render exerciseDL.ipynb --to html
quarto render exerciseDL.ipynb --to pdf
```

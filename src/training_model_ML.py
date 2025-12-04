#!/usr/bin/env python3

import optuna
import sys
import os
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from utils import set_random_seed, calculate_scores
from utils import load_objects, load_class_from_config
from utilsModel import TrainModels

def main():
    optuna_dir = 'optuna'
    pkl_dir = os.path.join(optuna_dir,'file_config')
    # TODO Select best trial or number of trial
    # Check if study_name argument is provided
    if len(sys.argv) < 2:
        print("Error: No study name provided")
        print("Usage: script.py <study_name>")
        sys.exit(1)
    study_name = sys.argv[1]

    storage = 'sqlite:///optuna/optuna_ML_exercise.db'
    study = optuna.load_study(
        study_name = study_name,
        storage = storage
    )
    trial = study.best_trial
    trial_number = trial.number
    config = load_objects(os.path.join(pkl_dir, study_name + '.pkl'))

    # Random reproducibility
    set_random_seed()
    random_state = config['random_state']

    # Load dataset
    print('Loading data...')
    data_file = config['datafile']
    load = config['load']
    df = load(data_file) 

    # Proprocess data to train/validate model
    # simple process features + create new features
    print('Preprocessing data...')
    preproc_features = config['preproc_features']
    preproc_target = config['preproc_target']
    df_preproc = preproc_features(preproc_target(df))
    
    # split data into train and test datasets
    print('Splitting dataset in train and test...')
    test_size_test = config['split_test']
    if 'stratify' in config:
        df_train, df_test = train_test_split(df_preproc, test_size = test_size_test, random_state = random_state, stratify=df_preproc[config['stratify']]) 
    else:
        df_train, df_test = train_test_split(df_preproc, test_size = test_size_test, random_state = random_state) 
    print(f'Train: {df_train.shape[0]}/Test: {df_test.shape[0]}')

    # preprocess data Imputation/Encoding/Transformation
    print('Processing data...')
    proc = load_class_from_config(config['proc'])
    proc.fit(df_train, df_train['target'])

    # Loading parameters
    print('Loading parameters...')
    model_fixed_params = trial.user_attrs['fixed_params']['model']
    model_opt_params = trial.user_attrs['params']['model']
    print(model_fixed_params)
    print(model_opt_params)

    # Build Model
    model = config['model']
    model_params = model_fixed_params | model_opt_params
    model = model(**model_params)

    # Train model
    print('Training model...')
    X_train = proc.transform(df_train)
    y_train = df_train['target']
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    # Validate model/Plot
    print('Testing model...')
    X_test = proc.transform(df_test)
    y_pred = model.predict(X_test)
    y_true = df_test['target']
    # Calculate scoring
    scores = calculate_scores(y_true, y_pred, config['scoring'])
    print('Saving model...')
    model_name = study_name + '_T' + str(trial_number)
    file_model = os.path.join('models', model_name + '.pkl')
    with open(file_model, 'wb') as f:
        pickle.dump(model, f)
    print('Writing model info...')
    models_trained = TrainModels()
    # TODO Save scores for test
    info_model = {
        'study': study_name,
        'number_trial': trial_number,
        'data_file': data_file,
        'preproc_features': preproc_features,
        'proc': proc,
        'file_model': file_model,
        'model_params': model_params,
        'date': datetime.now(),
        'scores': scores
    }
    models_trained.append_model(model_name, info_model)
    models_trained.save()

if __name__ == "__main__":
    main()
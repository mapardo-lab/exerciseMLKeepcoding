#!/usr/bin/env python3

import os
import sys
import optuna
from sklearn.model_selection import train_test_split
from utils import set_random_seed, save_objects, load_yaml_config, load_from_config, load_make_scorer_from_config, load_class_from_config, check_params
from utilsOptuna import ObjectiveFunctionML, create_study

def main():
    optuna_dir = 'optuna'
    yaml_dir = os.path.join(optuna_dir,'run_config')
    pkl_dir = os.path.join(optuna_dir,'file_config')
    # Check if run_name argument is provided
    if len(sys.argv) < 2:
        print("Error: No run name provided")
        print("Usage: script.py <.yaml file>")
        sys.exit(1)
    print('Reading configuration .yaml file...')
    config = load_yaml_config(os.path.join(yaml_dir,sys.argv[1]))

    study_name = sys.argv[1].replace('.yaml','').replace('./','')
    run_name = config['run_name']
    print(f'Study: {study_name} \nRun: {run_name}')    
    
    print('Setting random state for reproducibility...')
    set_random_seed()
    random_state = config['random_state']
    
    print('Loading dataset...')
    data_file = os.path.join('data', config['data_file'])
    load = load_from_config(config['load_data'])
    df = load(data_file) 

    print('Preprocessing data... simple processing + new features')
    preproc_features = load_from_config(config['preproc_features'])
    preproc_target = load_from_config(config['preproc_target'])
    df_processed = preproc_target(preproc_features(df))
    
    # split data into train and test datasets
    print('Splitting dataset in train and test...')
    test_size_test = config['test_size_test']
    if 'stratify' in config:
        df_train, df_test = train_test_split(df_processed, test_size = test_size_test, random_state = random_state, stratify=df_processed[config['stratify']]) 
    else:
        df_train, df_test = train_test_split(df_processed, test_size = test_size_test, random_state = random_state) 
    print(f'Train: {df_train.shape[0]}/Test: {df_test.shape[0]}')

    # preprocess data Imputation/Encoding/Transformation
    print('Loading transformation data process...')
    proc = load_class_from_config(config['proc'])
    proc.fit(df_train, df_train['target'])
    
    # Modify to new framework
    print('Processing data...')
    X_train = proc.transform(df_train)
    y_train = df_train['target']

    # hyperparameter to optimizate
    print('Loadind search space...')
    search_space = config['search_space']

    # fixed hyperparameters
    print('Loading fixed parameters...')
    fixed_params = config['fixed_params']

    # scores output
    print('Loading scores to monitorize...')
    scoring = load_make_scorer_from_config(config['scoring'])

    # model configuration 
    print('Loadind model...')
    model = load_class_from_config(config['model'])

    # objetive function ML
    objective = ObjectiveFunctionML(
        X=X_train, y=y_train,
        model=model,
        fixed_params=fixed_params,
        search_space=search_space,
        scoring=scoring,
        score = config['score'],
        cv = config['cv'],
        run_name=run_name
    )

    # parameters for study
    # TODO Write complete sampler info into .yaml file
    config_study = config['study']
    study_params = {
        'direction': config_study['direction'],
        'storage': config_study['storage'],
        'study_name': study_name,
        'sampler': optuna.samplers.TPESampler(**config_study['sampler']),
        'load_if_exists': True
    }

    # save parameters
    file_config = os.path.join(pkl_dir, study_name + '.pkl')
    save_config = {
        'random_state': random_state,
        'datafile': data_file,
        'load': load,
        'preproc_features': preproc_features,
        'preproc_target': preproc_target,
        'split_test': test_size_test,
        'proc': config['proc'],
        'model': model,
        'scoring': config['scoring']
    }

    # check if .pkl file exists. If it exists check if the configuration is the same
    if os.path.exists(file_config):
        print('Configuration file found. Checking for consistency...')
        if check_params(file_config, save_config):
            print('Consistency OK')
        else:
            raise Exception('Parameters configuration FAIL. Check parameter configuration.') 
    else:
        print('Saving configuration...')
        save_objects(save_config, file_config)

    # user attributes for study
    user_attr = [
        ('script', f'{sys.argv[0]}'),
        ('config_file', file_config),
        ('comments', config['comments'])
    ]

    # create study
    study = create_study(study_params, user_attr)

    ## run trials
    print('Running hyperparameter optimization...')
    study.optimize(objective, n_trials=config['num_trials'])
    print(f"Completed {len(study.trials)} trials")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

if __name__ == "__main__":
    main()
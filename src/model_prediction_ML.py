#!/usr/bin/env python3

import pandas as pd
import pickle
from utilsModel import TrainModels

def main():
    # TODO Input for this exe
    models_trained = TrainModels()
    model_name = 'MML_xgboost_metadata1_T3'
    config = models_trained.list_of_models[model_name]

    print('Reading data...')
    df = pd.read_csv('data/example_dataset.csv')

    print('Preprocessing data...')
    preproc_features = config['preproc_features']
    df_preproc = preproc_features(df)

    print('Processing data...')
    proc = config['proc']
    X = proc.transform(df_preproc)

    print('Loading model...')
    with open(config['file_model'], 'rb') as f:
        model = pickle.load(f)

    # TODO Use this model as a API (model deployment)
    print('Making predictions...')
    y_pred = model.predict(X)
    print(y_pred)

if __name__ == "__main__":
    main()
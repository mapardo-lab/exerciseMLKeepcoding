#!/usr/bin/env python3

import sys
import os
import pandas as pd
import pickle
from utilsModel import TrainModels

def main():
    if len(sys.argv) < 2:
        print("Error: No data file provided")
        print("Usage: script.py <data_file.csv>")
        sys.exit(1)
    data_file = sys.argv[1]

    # TODO Input for this exe
    models_trained = TrainModels()
    model_name = 'MML_SVR_RBF_data1_T1'
    config = models_trained.list_of_models[model_name]

    print('Reading data...')
    df = pd.read_csv(os.path.join('data', data_file))

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
    df['target'] = y_pred
    df.to_csv('data/dataset_with_prediction.csv', index=False)

if __name__ == "__main__":
    main()
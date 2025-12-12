import pandas as pd
import numpy as np
from utils import load_objects, save_objects

class TrainModels:
    """
    Manages a collection of trained models, handling their storage, retrieval, and summary.

    This class abstracts the operations of loading, saving, adding, and inspecting
    machine learning models that have been trained and saved as pickle files.
    """
    def __init__(self):
        """Initializes the TrainModels manager by loading the collection from a file."""
        self.filename = 'models/list_trained_models.pkl'
        self.list_of_models = self.load()

    def load(self):
        """
        Loads the dictionary of trained models from a pickle file.

        If the file does not exist, it prints a warning and returns an empty dictionary.

        Returns:
            dict: The dictionary of trained models.
        """
        result_dict = {}
        try:
            result_dict =  load_objects(self.filename)
        except FileNotFoundError:
            print(self.filename + ' not found.\nCreating new empty list of trained models')
        return result_dict

    def append_model(self, name, info):
        """
        Adds information about a new trained model to the collection.

        Args:
            name (str): A unique name or identifier for the model.
            info (dict): A dictionary containing metadata and results for the model.
        """
        self.list_of_models[name] = info
        print(f'New trained model added: {name}')

    def save(self):
        """Saves the current collection of trained models to the pickle file."""
        save_objects(self.list_of_models, self.filename)

    def summary(self, score):
        """
        Generates a summary DataFrame of the trained models.

        Args:
            score (str): The specific score metric to include in the summary.

        Returns:
            pd.DataFrame: A DataFrame containing the model name, training date,
                          data file used, and the specified score.
        """
        results ={
            'model_name': [],
            'date_train': [],
            'data_file': [],
            score: []
        }
        for model_name, info in self.list_of_models.items():
            results['model_name'].append(model_name)
            results['date_train'].append(info['date'].strftime("%Y-%m-%d %H:%M:%S"))
            results['data_file'].append(info['data_file'])
            results[score].append(info['scores'][score][0])
        return pd.DataFrame(results)

    def info_model_results(self, model_name):
        """
        Displays detailed information for a specific trained model.

        This includes metadata like the training date and performance scores.
        If a confusion matrix is available, it is printed and plotted.

        Args:
            model_name (str): The name of the model to inspect.
        """
        info = self.list_of_models[model_name]

        print(f'Model name: {model_name}')
        print(f"Date training: {info['date'].strftime('%Y-%m-%d %H:%M:%S')}")

        for score, value in info['scores'].items():
            if (score != 'loss') & (score != 'confusion_matrix'):
                print(f'{score}: {round(value[0], 2)}')

        if 'confusion_matrix' in info['scores']:
            print(f"Confusion matrix: {info['scores']['confusion_matrix'][0]}")

            cm = np.array(info['scores']['confusion_matrix'][0] , dtype=int)
            confusion_matrix_plot(cm)


    def info_model_summary(self, model_name):
        """
        Displays detailed information for a specific trained model.

        This includes metadata like the training date and performance scores.
        If a confusion matrix is available, it is printed and plotted.

        Args:
            model_name (str): The name of the model to inspect.
        """
        info = self.list_of_models[model_name]

        print(f'Model name: {model_name}')
        print(f"Study: {info['study']}")
        print(f"Trial: {info['number_trial']}")
        print(f"Date training: {info['date'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data file: {info['data_file']}")
        print(f"Feature preprocessing: {info['preproc_features'].__name__}")
        print(f"Feature processing: {info['proc']}") # TODO Output simpler module/[class]/function/parameters --> Function in training_model_ML.py for config['proc']
        print(f"Model: {info['model']}") 
        print(f"Hyperparameters: {info['model_params']}")
        print(f"Model file: {info['file_model']}") 

    def remove_model(self, model_name):
        """
        Removes a model from the collection and saves the changes.

        Args:
            model_name (str): The name of the model to remove.
        """
        self.list_of_models.pop(model_name)
        self.save()
        print(f'Model {model_name} removed')

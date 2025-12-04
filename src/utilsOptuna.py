import pandas as pd
import matplotlib.pyplot as plt
import optuna
import os
import pickle
import statistics
from pprint import pprint
from sklearn.model_selection import cross_validate
from collections import Counter
from optuna.trial import TrialState

class ObjectiveFunction():
    """
    Base class for defining objective functions with configurable hyperparameter search spaces.
    """
    def __init__(self, fixed_params, search_space, score, run_name):
        self.fixed_params = fixed_params
        self.search_space = search_space
        self.score = score
        self.run_name = run_name

    def get_optimizable_params(self, trial): 
        # Define optimizable parameters based on search_space config
        optimizable_params = {}
        for module_name, module_config in self.search_space.items():
            optimizable_params[module_name] = {}
            for param_name, param_config in module_config.items():
                if param_config['type'] == 'float':
                    if param_config.get('log', False):
                        optimizable_params[module_name][param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high'], 
                            log=True
                        )
                    else:
                        optimizable_params[module_name][param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high']
                        )
                elif param_config['type'] == 'int':
                    if param_config.get('exp2', False):
                        optimizable_params[module_name][param_name] = 2**trial.suggest_int(
                            param_name, 
                            param_config['low'], 
                            param_config['high']
                        )
                    else:
                        optimizable_params[module_name][param_name] = trial.suggest_int(
                            param_name, 
                            param_config['low'], 
                            param_config['high']
                        )
                elif param_config['type'] == 'categorical':
                    optimizable_params[module_name][param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config['choices']
                    )
        return optimizable_params

class ObjectiveFunctionML(ObjectiveFunction):
    """
    Objective function for machine learning models using cross-validation for hyperparameter optimization.
    """
    def __init__(self, X, y, model, fixed_params, search_space, scoring, score, cv, run_name):
        super().__init__(fixed_params, search_space, score, run_name)
        self.model = model
        self.scoring = scoring
        self.X = X
        self.y = y 
        self.cv = cv
    
    def __call__(self, trial):
        params =  self.get_optimizable_params(trial)
        model_params = {**self.fixed_params['model'], **params['model']}

        trial.set_user_attr('run', self.run_name)
        
        model = self.model(**model_params)

        cv_results = cross_validate(model, self.X, self.y, cv=self.cv, scoring=self.scoring, return_train_score=True, n_jobs=1)
        score = cv_results['test_' + self.score].mean()

        results_as_lists = {key: value.tolist() for key, value in cv_results.items()}
        trial.set_user_attr('model', f'{type(model)}') 
        trial.set_user_attr('k-fold', self.cv)
        trial.set_user_attr('score', self.score)
        trial.set_user_attr('train_results', results_as_lists) 
        trial.set_user_attr('fixed_params', self.fixed_params)
        trial.set_user_attr('params', params)

        return score

def create_study(study_params, user_attr):
    """
    Creates and configures an Optuna study with specified parameters and user attributes.
    """
    study = optuna.create_study(**study_params)
    for attr in user_attr:
        study.set_user_attr(attr[0],attr[1])
    return study

def info_studies_from_storage(db_url):
    """
    Get study names and best scores from Optuna database storage.
    """
    
    storage = optuna.storages.RDBStorage(url=db_url)
    list_studies = storage.get_all_studies()
    
    studies_data = []
    
    for study_summary in list_studies:
        study = optuna.load_study(study_name=study_summary.study_name, storage=storage)
        study_info = {
            'study_name': study.study_name,
            'best_score': study.best_value,
            'description': study.user_attrs['comments']
        }
        
        studies_data.append(study_info)
    
    return pd.DataFrame(studies_data)

def best_trial_scores_ML(db_url, list_scores, list_studies): 
    """
    Retrieves and computes the performance metrics from the best trial in an Optuna study (Machine Learning), 
    returning the model name along with the mean training and validation scores rounded 
    to three decimal places for concise evaluation. 
    """
    storage = optuna.storages.RDBStorage(url=db_url)

    results = []
    for study_name in list_studies:
        study = optuna.load_study(study_name = study_name, storage = storage)
        results_study = [study.study_name]
        for score in list_scores:
            results_study.append(round(statistics.mean(study.best_trial.user_attrs['train_results'][score]), 3))
        results.append(results_study)
    return results

class Study():
    def __init__(self, storage, study_name):
        self.storage = storage
        self.study_name = study_name
        self.study = self._load_study()
        self.config_study = self._get_config_study()
        self.list_runs = self._build_list_runs()

    def show_study_config(self):
        print(f'Showing {self.study_name} study...')
        print(self.study.user_attrs['comments'])
        pprint(self.config_study, sort_dicts = False)

    def _get_config_study(self):
        config_file = self.study.user_attrs['config_file']
        if not os.path.exists(config_file):
            raise FileNotFoundError(config_file + ' does not exist.')
        with open(config_file, 'rb') as f:
            return pickle.load(f)

    def _load_study(self):
        print(f'Loading {self.study_name} study...\n')
        study = optuna.load_study(
            study_name = self.study_name,
            storage = self.storage
        )
        return study

    def _build_list_runs(self):
        list_runs_name = set([trial.user_attrs.get('run', None) for trial in self.study.trials])
        direction = self.study.direction.name
        list_runs = {}
        for run_name in list_runs_name:
            trials = []
            for trial in self.study.trials:
                name = trial.user_attrs.get('run', None)
                if name == run_name:
                    trials.append(trial)
            list_runs[run_name] = Run(trials, direction)
        return list_runs

    def show_runs(self):
        print(f'Showing {self.study_name} study...')
        result = {
            'name_run': [],
            'start': [],
            'complete': [],
            'pruned': [],
            'fail': [],
            'total': [],
            'best_score': []
        }
        for name, run in self.list_runs.items():
            result['name_run'].append(name)
            result['start'].append(run.start)
            state = run.get_state()
            result['complete'].append(state[0])
            result['pruned'].append(state[1])
            result['fail'].append(state[2])
            result['total'].append(state[3])
            result['best_score'].append(round(run.best_score, 3))
        df = pd.DataFrame(result)
        
        return df.sort_values('start')

    def show_runs_extended(self):
        print(f'Showing {self.study_name} study...')
        for name, run in self.list_runs.items():
            print(f'\nrun\t\tstart\t\t\tC/P/F\tbest')
            state = run.get_state()
            print(f'{name}\t{run.start}\t{state[0]}/{state[1]}/{state[2]}\t{round(run.best_score, 3)}')
            print('Optimized')
            pprint(run.trials[0].distributions)
            print('Fixed')
            pprint(run.trials[0].user_attrs['fixed_params'])

    def show_best_result_study(self):
        """
        Prints the best trial results from an Optuna study, 
        including hyperparameter values
        """
        print(f'Showing {self.study_name} study...')
        print("Best trial:")
        trial = self.study.best_trial

        print("  Number: ", trial.number)
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}:\t{value:.5f}")

    def plot_slice(self):
        """
        Generate slice plots of objective values versus specified hyperparameters for a study's completed trials.
        """
        print(f'Showing {self.study_name} study...')
        # TODO Carefull not more than six hyperparameters
        trials = self.study.trials
        params = self.get_params_log()
        values = self.get_params_score()

        # Columns to plot
        n_plots = len(params)
        
        # Create subplots
        fig, axes = plt.subplots(1, n_plots, figsize=(2*n_plots, 4), sharey=True)
        axes = axes.flatten()

        best_score = self.study.best_value
        
        i = 0
        for param, log in params.items():
            axes[i].axhline(y=best_score, color='red', linestyle='--', linewidth=2, alpha=0.3)
            axes[i].scatter(values[param], values['score'], alpha=0.9, color='lightblue', s=30, edgecolor='grey')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Objetive value')
            axes[i].grid(True, alpha=0.3)     
            if log:
                axes[i].set_xscale('log')
            # Hide y-axis label for all except first plot
            if i > 0:
                axes[i].set_ylabel('')
            i = i + 1

        plt.tight_layout()
        plt.show()

    def plot_training_curves(self, number, score):
        """
        From the model training output, the training progress is plotted 
        for loss function and accuracy values. Optionally, accuracy for
        the test is also plotted.
        """
        trial = self.study.trials[number]
        train_loss = trial.user_attrs['train_results']['loss']
        train_score = trial.user_attrs['train_results'][score]
        val_loss = trial.user_attrs['val_results']['loss']
        val_score = trial.user_attrs['val_results'][score]
        num_epochs = len(train_loss)
        plt.style.use("ggplot")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(num_epochs), train_loss, label="Train loss")
        plt.plot(range(num_epochs), val_loss, label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(num_epochs), train_score, label="Train")
        plt.plot(range(num_epochs), val_score, label="Validation")
        plt.xlabel("Epoch #")
        plt.ylabel(score)
        plt.legend()
        plt.tight_layout()
        plt.show()
    

    def get_params_log(self):
        total_param = {}
        for run in self.list_runs.values():
            params = {}
            for param, distribution in run.trials[0].distributions.items():
                params[param] = distribution.log
            total_param = total_param | params
        return total_param

    def get_params_score(self):
        list_values = []
        for trial in self.study.trials:
            list_values.append(trial.params | {'score': trial.value})

        df = pd.DataFrame(list_values)
        return df

class Run():
    def __init__(self, list_trials, direction):
        self.trials = list_trials
        self.direction = direction
        self.state = self._build_total_state()
        self.start = self._build_start()
        self.end = self._build_end()
        self.search_space = self._build_search_space()
        self.best_score = self._build_best_score()

    def _build_total_state(self):
        state = []
        for trial in self.trials:
            state.append(trial.state)
        fixed_keys = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL]
        state_count = Counter(state)
        for key in fixed_keys:
            state_count[key] = state_count.get(key, 0)
        return state_count
    
    def _build_start(self):
        datetime = []
        for trial in self.trials:
            datetime.append(trial.datetime_start)
        return min(datetime).replace(microsecond=0)

    def _build_end(self):
        datetime = []
        for trial in self.trials:
            datetime.append(trial.datetime_complete)
        return max(datetime).replace(microsecond=0)

    def _build_search_space(self):
        return self.trials[0].distributions
    
    def _build_best_score(self):
        scores = []
        for trial in self.trials:
            scores = scores + trial.values
        if len(scores) == 0:
            return 0
        elif self.direction == 'MINIMIZE':
            return min(scores)
        elif self.direction == 'MAXIMIZE':
            return max(scores)
    
    def get_state(self):
        counts = []
        for count in self.state.values():
            counts.append(count)
        total = sum(counts)
        counts.append(total)
        return counts

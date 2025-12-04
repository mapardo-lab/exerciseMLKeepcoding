import pandas as pd
import numpy as np
from utils import load_objects, save_objects
#from utilsPlots import confusion_matrix_plot

class ModelBase():
    def __init__(self, model_config, params, fixed_params = None):
        self.architecture = model_config['architecture']
        self.arch_params = params['architecture'] | fixed_params['architecture']
        self.model = self.architecture(**self.arch_params)
        self.criterion = model_config['criterion'](**params['criterion'], **fixed_params['criterion'])
        self.optimizer = model_config['optimizer'](params = self.model.parameters(), **params['optimizer'], **fixed_params['optimizer'])
        self.device = model_config['device']
        self.model.to(self.device)
        self.scoring = model_config['scoring']
        self.results_train = StatsModel(self.scoring)
        self.results_val = StatsModel(self.scoring)

    def train_epoch(self, train_loader): 
        """
        Train neural network for one epoch and return the training metrics.
        """
        self.model.train()
        train_loss = 0
        y_pred = np.array([])
        y_true = np.array([])

        for data in train_loader:
            for key, value in data.items():
                data[key] = value.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data['target'])
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            y_true = np.concatenate([y_true, data['target'].cpu().numpy()])
            y_pred = np.concatenate([y_pred, predicted.cpu().numpy()])

        self.results_train.update(train_loss, y_true, y_pred)

    def eval_model(self, val_loader): 
        """
        Evaluates network model on validation data and returns metrics
        """
        self.model.eval()
        val_loss = 0
        y_pred = np.array([])
        y_true = np.array([])

        with torch.no_grad():
            for data in val_loader:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, data['target'])
                val_loss += loss.item()
                _, predicted = output.max(1)
                y_true = np.concatenate([y_true, data['target'].cpu().numpy()])
                y_pred = np.concatenate([y_pred, predicted.cpu().numpy()])
        self.results_val.update(val_loss, y_true, y_pred)

    def train_model(self, num_epochs, train_loader):
        """
        Train model and plot metrics from training
        """
        self.model.to(self.device)

        for epoch in range(num_epochs):
            self.train_epoch(train_loader)

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename + ".pth")

class ModelScheduler(ModelBase):
    def __init__(self, model_config, params, fixed_params = None):
        super().__init__(model_config, params, fixed_params)
        self.scheduler = model_config['scheduler'](self.optimizer, **params['scheduler'], **fixed_params['scheduler'])

    def train_epoch(self, train_loader):
        """
        Train neural network for one epoch and return the training metrics.
        """
        self.model.train()
        train_loss = 0
        y_pred = np.array([])
        y_true = np.array([])

        for data in train_loader:
            for key, value in data.items():
                data[key] = value.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data['target'])
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            y_true = np.concatenate([y_true, data['target'].cpu().numpy()])
            y_pred = np.concatenate([y_pred, predicted.cpu().numpy()])

        self.results_train.update(train_loss, y_true, y_pred)
        self.scheduler.step()

class ModelL1(ModelBase):
    def __init__(self, model_config, params, fixed_params = None):
        super().__init__(model_config, params, fixed_params)
        self.l1_lambda = params['regularization'].get('l1_lambda', fixed_params['regularization'].get('l1_lambda', 0.01))

    def train_epoch(self, train_loader): 
        self.model.train()
        train_loss = 0
        y_pred = np.array([])
        y_true = np.array([])
        for data in train_loader:
            for key, value in data.items():
                data[key] = value.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data['target'])
            l1_penalty = sum(p.abs().sum() for p in self.model.parameters())
            loss += self.l1_lambda * l1_penalty
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            y_true = np.concatenate([y_true, data['target'].cpu().numpy()])
            y_pred = np.concatenate([y_pred, predicted.cpu().numpy()])

        self.results_train.update(train_loss, y_true, y_pred)

class ModelL2(ModelBase):
    def __init__(self, model_config, params, fixed_params = None):
        super().__init__(model_config, params, fixed_params)
        self.l2_lambda = params['regularization'].get('l2_lambda', fixed_params['regularization'].get('l2_lambda', 0.01))

    def train_epoch(self, train_loader): 
        self.model.train()
        train_loss = 0
        y_pred = np.array([])
        y_true = np.array([])
        for data in train_loader:
            for key, value in data.items():
                data[key] = value.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data['target'])
            l2_penalty = sum((p ** 2).sum() for p in self.model.parameters())
            loss += self.l2_lambda * l2_penalty
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            y_true = np.concatenate([y_true, data['target'].cpu().numpy()])
            y_pred = np.concatenate([y_pred, predicted.cpu().numpy()])

        self.results_train.update(train_loss, y_true, y_pred)

class StatsModel:
    """
    Tracks and accumulates training and validation metrics throughout the training process.
    """
    def __init__(self, scoring):
        """
        Initialize Train object with empty lists for losses and accuracies.
        """
        self.scoring = scoring
        self.results = {}
        for score_name, _ in scoring.items():
            self.results['loss'] = []
            self.results[score_name] = []

    def update(self, loss, y_true, y_pred):
        new_results = self.get_scores(loss, y_true, y_pred)
        for score_name, score in new_results.items():
            self.results[score_name].append(score)

    def get_scores(self, loss, y_true, y_pred):
        """
        Computes and returns a dictionary of evaluation metrics including loss and specified scoring functions.
        """
        results = {'loss': loss}
        for score_name, score_func in self.scoring.items():
            results[score_name] = score_func(y_true, y_pred)
        return results
    
class TrainModels:
    def __init__(self):
        self.filename = 'models/list_trained_models.pkl'
        self.list_of_models = self.load()

    def load(self):
        result_dict = {}
        try:
            result_dict =  load_objects(self.filename)
        except FileNotFoundError:
            print(self.filename + ' not found.\nCreating new empty list of trained models')
        return result_dict

    def append_model(self, name, info):
        self.list_of_models[name] = info
        print(f'New trained model added: {name}')

    def save(self):
        save_objects(self.list_of_models, self.filename)

    def summary(self, score):
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

    def info_model(self, model_name):
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

    def remove_model(self, model_name):
        self.list_of_models.pop(model_name)
        self.save()
        print(f'Model {model_name} removed')

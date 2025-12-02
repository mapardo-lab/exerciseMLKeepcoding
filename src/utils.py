import numpy  as np  
import random
import base64
import pickle
import os
import yaml
import importlib
from functools import partial
from sklearn.metrics import confusion_matrix, make_scorer

def set_random_seed(seed=42): 
  """
  Fixed seeds for reproducibility
  """
  np.random.seed(seed)
  random.seed(seed)

def build_scorer(metric_func, **metric_params):
    """
    Create a scorer function with fixed parameters (f1_score, recall_score, ...)
    """
    def scorer(y_true, y_pred):
        return metric_func(y_true, y_pred, **metric_params)
    return scorer

def get_column_transformer_info(column_transformer): 
    """
    Get comprehensive info for a ColumnTransformer
    """
    
    def get_transformer_info(transformer):
        return {
            'class': transformer.__class__.__name__,
            'module': transformer.__module__,
        }
    
    ct_info = {
        'type': 'ColumnTransformer',
        'transformers': [],
    }
    
    for name, transformer, columns in column_transformer.transformers:
        ct_info['transformers'].append({
            'name': name,
            'columns': columns,
            'transformer': get_transformer_info(transformer)
        })
    
    return ct_info

def info_object(obj):
    results = {}
    results['name'] = obj.__name__
    results['module'] = obj.__module__
    return results

def info_train(train_config):
    result = {}
    for key, value in train_config.items():
        if (isinstance(value, type) | callable(value)):
            result[key] = info_object(value)
    return result

def serial_encode(obj):
    serialized = dill.dumps(obj)
    encoded = base64.b64encode(serialized).decode("utf-8")
    return encoded

def serial_decode(encoded):
    decoded = dill.loads(base64.b64decode(encoded))
    return decoded

def confusion_matrix_list(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).tolist()

def save_objects(obj_dict, filename):
    """Save multiple objects to a file"""
    with open(filename, 'wb') as f:
        pickle.dump(obj_dict, f)

def load_objects(filename):
    """Load objects from a file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(filename + ' does not exist.')
    
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def load_yaml_config(file_path: str):
    """
    Load and parse YAML configuration file
    
    Args:
        file_path (str): Path to the YAML file
        
    Returns:
        Dict[str, Any]: Parsed configuration data
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{file_path}' not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}

def load_from_config(config):
    if 'class' in config:
        return load_class_from_config(config)
    elif 'function' in config:
        return load_function_from_config(config)
    
def load_function_from_config(config):
    module_name = config['module']
    function_name = config['function']
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if 'parameters' in config:
        return partial(function, **config['parameters'])
    else:
        return function

def load_class_from_config(config):
    module_name = config['module']
    class_name = config['class']
    module = importlib.import_module(module_name)
    class_load = getattr(module, class_name)
    if 'function' in config:
        function_name = config['function']
        function = getattr(class_load, function_name)
        return function(**config['parameters'])
    elif 'parameters' in config:
        return class_load(**config['parameters'])
    else:
        return class_load

def load_dict_from_config(config):
    result = {}
    for key, value in config.items():
        if isinstance(value, dict) and 'module' in value:
            result[key] = load_from_config(value)
        else:
            result[key] = value
    return result

def load_make_scorer_from_config(config):
    result = {}
    for name_score, score in config.items():
        module_name = score['module']
        function_name = score['function']
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        if 'parameters' in score:
            result[name_score] = make_scorer(function, **score['parameters'])
    return result

def calculate_scores(y_true, y_pred, scoring):
    results = {}
    for score, function in scoring.items():
         function_score = load_function_from_config(function)
         results[score] = [function_score(y_true, y_pred)]
    return results

def check_params(file_config, save_config):
    with open(file_config, 'rb') as f:
        saved_config = pickle.load(f)
    if saved_config != save_config:
        return False
    else:
        return True
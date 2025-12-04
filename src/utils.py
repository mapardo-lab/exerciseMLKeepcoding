import numpy  as np  
import random
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
    """
    Loads a class instance or a function (potentially partially applied)
    based on the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing 'class' or 'function' keys
                       along with 'module' and optional 'parameters'.

    Returns:
        object or callable: An instance of the specified class or the loaded function.
    """
    if 'class' in config:
        return load_class_from_config(config)
    elif 'function' in config:
        return load_function_from_config(config)
    
def load_function_from_config(config):
    """
    Loads a function from a module based on a configuration dictionary.

    If 'parameters' are provided in the config, it returns a partial function
    with those parameters pre-applied.

    Args:
        config (dict): Configuration dictionary with 'module', 'function', and optional 'parameters'.

    Returns:
        callable: The loaded function or a partial function.
    """
    module_name = config['module']
    function_name = config['function']
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if 'parameters' in config:
        return partial(function, **config['parameters'])
    else:
        return function

def load_class_from_config(config):
    """
    Loads and instantiates a class or calls a class method from a module
    based on a configuration dictionary.

    - If 'parameters' are present, it instantiates the class with them.
    - If 'function' is present, it calls that class method with 'parameters'.
    - Otherwise, it returns the class definition itself.

    Args:
        config (dict): Configuration dictionary with 'module', 'class', and optional 'parameters' or 'function'.

    Returns:
        object or callable: An instance of the class, the result of a class method, or the class itself.
    """
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
    """
    Recursively loads objects from a configuration dictionary.

    It iterates through the dictionary and, for any value that is a dictionary
    containing a 'module' key, it loads the corresponding object.

    Args:
        config (dict): A dictionary possibly containing sub-dictionaries for object loading.

    Returns:
        dict: A new dictionary with the specified objects loaded.
    """
    result = {}
    for key, value in config.items():
        if isinstance(value, dict) and 'module' in value:
            result[key] = load_from_config(value)
        else:
            result[key] = value
    return result

def load_make_scorer_from_config(config):
    """
    Creates a dictionary of scikit-learn scorer objects from a configuration.

    Args:
        config (dict): A dictionary where keys are scorer names and values are
                       configurations for the metric functions.

    Returns:
        dict: A dictionary of scikit-learn scorer objects.
    """
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
    """
    Calculates scores based on true and predicted values using a scoring configuration.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        scoring (dict): A dictionary where keys are score names and values are
                        configurations for the scoring functions.

    Returns:
        dict: A dictionary with score names as keys and their calculated values.
    """
    results = {}
    for score, function in scoring.items():
         function_score = load_function_from_config(function)
         results[score] = [function_score(y_true, y_pred)]
    return results

def check_params(file_config, save_config):
    """
    Compares a saved configuration from a pickle file with a given configuration object.

    Args:
        file_config (str): Path to the pickled configuration file.
        save_config (dict): The configuration object to compare against.

    Returns:
        bool: True if the configurations are identical, False otherwise.
    """
    with open(file_config, 'rb') as f:
        saved_config = pickle.load(f)
    result = []
    for key, value in saved_config.items():
        if isinstance(value, partial):
            result.append(equal_partial(saved_config[key], save_config[key]))
        else:
            result.append(saved_config[key] == save_config[key])
    return all(result)

def equal_partial(p1, p2):
    """
    Checks if two functools.partial objects are equivalent.

    Args:
        p1 (functools.partial): The first partial object.
        p2 (functools.partial): The second partial object.

    Returns:
        bool: True if the functions, arguments, and keywords are the same, False otherwise.
    """
    return (
        p1.func == p2.func and
        p1.args == p2.args and
        p1.keywords == p2.keywords
    )

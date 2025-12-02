import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import cross_val_score

def cramers_v(x, y, bias_correction=False):
    """
    Calculate Cramer's V statistic with optional bias correction
    Bias correction for smaller sample size or when number of
    categories in the variables is large.
    
    Args:
        x, y: pandas Series or array-like categorical data
        bias_correction: Whether to apply Bergsma's bias correction
        
    Returns:
        Cramer's V between 0 (no association) and 1 (perfect association)
        0.0 - 0.1 No association
        0.1 - 0.3 Weak association
        0.3 - 0.5 Moderate association
        > 0.5 Strong association
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    if bias_correction:
        # Bergsma's bias correction
        phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        r_corr = r - ((r-1)**2)/(n-1)
        k_corr = k - ((k-1)**2)/(n-1)
        denominator = min((k_corr-1), (r_corr-1))
    else:
        phi2_corr = phi2
        denominator = min(k-1, r-1)
    
    # Handle edge case where denominator is 0
    if denominator <= 0:
        return 0.0
    
    v = np.sqrt(phi2_corr / denominator)
    return min(v, 1.0)  # Ensure result doesn't exceed 1 due to floating point

def univariate_predictive_power(X, y, model, cv=5):
    """
    Calculate predictive power for each feature using 1D models
    """
    scores = {}
    scores['Feature'] = []
    scores['Score'] = []
    
    for feature in X.columns:
        X_single = X[[feature]]
        
        cv_scores = cross_val_score(model, X_single, y, cv=cv, scoring='neg_root_mean_squared_error')
        scores['Feature'].append(feature)
        scores['Score'].append(-1*np.mean(cv_scores))
    
    return pd.DataFrame(scores).sort_values(['Score'],ascending=True)

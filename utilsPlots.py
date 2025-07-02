import numpy  as np  
import pandas as pd
import math

import seaborn as sns
import matplotlib.pyplot as plt

from adjustText import adjust_text

from scipy.stats import ttest_ind


def plot_bars(df: pd.DataFrame, features: list , n_rows: int, n_cols: int, sort = False, log = False):
    """
    From dataframe plot several bar plots for the features contained in the list.
    Plots are distributed in rows and columns
    """
    plt.figure(figsize=(6 * n_cols, 4 * n_rows))  # Dynamic figure size
    for i, feature in enumerate(features, start=1):
        counts = df[feature].value_counts()
        if log:
            counts = np.log10(counts)
            ylabel = 'log(count)'
        else:
            ylabel = 'Count'
        if not sort:
            counts = counts.loc[sorted(counts.index)]
        plt.subplot(n_rows, n_cols, i)
        plt.xlabel(feature)
        plt.ylabel(ylabel)
        ax = counts.plot.bar()
        ax.set_xticklabels(ax.get_xticklabels(), 
                      rotation=45,
                      ha='right',  # Horizontal alignment
                      rotation_mode='anchor')
        plt.tight_layout()  # Prevent label clipping
    plt.show()

def plot_density(df: pd.DataFrame, features: list , n_rows: int, n_cols: int):
    """
    From dataframe plot several density plots for the features contained in the list.
    Plots are distributed in rows and columns
    """
    plt.figure(figsize=(6 * n_cols, 4 * n_rows))  # Dynamic figure size
    for i, feature in enumerate(features,start=1):
        plt.subplot(n_rows, n_cols, i)
        sns.kdeplot(df[feature], fill=True, color='skyblue', alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel('Density')
    plt.show()
    
def plot_estimator_feature_contquant(df: pd.DataFrame, estimator: str, features: list , n_rows: int, n_cols: int):
    """
    Scatter plots features vs estimator showing PCC and MI
    """
    plt.figure(figsize=(6 * n_cols, 5 * n_rows))  # Dynamic figure size
    for i, feature in enumerate(features['feature'],start=1):
        plt.subplot(n_rows, n_cols, i)
        plt.subplots_adjust(hspace=0.4)
        plt.plot(df[feature], df[estimator], 'o')
        plt.xlabel(feature)
        plt.ylabel(estimator)
        plt.title("PCC={:.2f}, MI={:.2f}".format(features['pcc'][i-1], features['mi'][i-1]),
              fontsize=16)
    plt.show()


def plot_correlation_vs_mi(features):
    """
    Scatter plot for PCC vs MI
    """
    plt.figure(figsize=(4, 4))
    plt.plot(features['pcc'], features['mi'],'o')
    plt.xlabel('PCC')
    plt.ylabel('MI')
    texts = [plt.text(
        features['pcc'][i], features['mi'][i], features['feature'][i], ha='center', va='center') for i in range(len(features['feature']))]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.show()
    
def plot_anova_vs_mi(features):
    """
    Scatter plot for ANOVA vs MI
    """
    plt.figure(figsize=(4, 4))
    sns.scatterplot(x=features['ftest'], y=features['mi'], hue=features['feature'], palette='viridis')
    plt.xlabel('F-test')
    plt.ylabel('MI')
    plt.show()

def plot_estimator_feature_qualit_bi(df: pd.DataFrame, estimator: str, features: list , n_rows: int, n_cols: int):
    """
    Violin plots and t-test p-value for correlation between estimator and features with two factors
    """
    plt.figure(figsize=(6 * n_cols, 5 * n_rows))  # Dynamic figure size
    for i, feature in enumerate(features,start=1):
        group1 = df[df[feature] == 0][estimator]
        group2 = df[df[feature] == 1][estimator]
        t_stat, p_value = ttest_ind(group1, group2)

        plt.subplot(n_rows, n_cols, i)
        plt.subplots_adjust(hspace=0.4)
        sns.violinplot(x=feature, y=estimator, data=df, inner='quartile', palette='pastel')
        plt.xlabel(feature)
        plt.ylabel(estimator)
        plt.title("t-Test p-value ={:.2E}".format(p_value),
              fontsize=16)
    plt.show()

def get_indexes_from_list(lst: list, targets: list):
    """
    Return the indexes in input list (lst) for the list of features in targets
    """
    return(list(filter(lambda x: lst[x] in targets, range(len(lst)))))

def get_elements_from_list(lst: list, indexes: list):
    """
    Return the elements of list lst in positions indexes
    """
    return(lst[i] for i in indexes)

def na_plot(df, threshold = 1):
    """
    Visualizes columns with missing values exceeding a specified threshold percentage.
    """
    df_na_features = df.isna().mean()*100
    df_na_features = df_na_features[df_na_features > threshold]
    fig_width = max(df_na_features.shape[0] * 0.3, 3)
    plt.figure(figsize=(fig_width,4))
    ax = df_na_features.sort_values(ascending=False).plot.bar()
    ax.set_xticklabels(ax.get_xticklabels(), 
                      rotation=45,
                      ha='right',  # Horizontal alignment
                      rotation_mode='anchor')
    plt.tight_layout()
    plt.show()
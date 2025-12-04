import numpy  as np  
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text
from scipy.stats import ttest_ind
from sklearn.feature_selection import mutual_info_classif

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

def plot_predictive_power(pp, score, n = None):
    """
    Plot a bar chart showing the predictive power of features.
    """
    if (n is None) or (n >= len(pp)):
        n = len(pp) 
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Score', y='Feature', data=pp[:n])
    plt.ylabel('')
    plt.xlabel(score)
    plt.show()

def heatmap_01_plot(df, triangle = False, threshold = None, label = ''):
    """
    Plot a 0–1 heatmap with optional upper-triangle masking and threshold-based coloring.
    """
    mask = np.zeros_like(df, dtype=bool)
    if triangle:
        # generate a mask for the upper triangle
        mask[np.triu_indices_from(mask)] = True

    # set up the matplotlib figure
    size = df.shape[0] * 0.25
    f, ax = plt.subplots(figsize=(size+1, size))

    if threshold is not None:
        # create custom colormap
        n_threshold_colors = int(threshold*256)
        init_cmap = plt.get_cmap('YlGnBu', n_threshold_colors)
        new_colors = init_cmap(np.linspace(0, 1, n_threshold_colors))
        # set colors above threshold to solid blue
        blue_color = np.array([0.03137255, 0.11372549, 0.34509804, 1])
        new_colors = np.vstack([new_colors, np.tile(blue_color,(256-n_threshold_colors,1))])
        cmap = LinearSegmentedColormap.from_list('trunc_YlGnBu', new_colors)
    else:
        cmap = 'YlGnBu'

    sns.heatmap(df, mask=mask, cmap=cmap, vmin=0, vmax=1,
                center=0.5, linewidths=.1, 
                fmt = ".2f",
                cbar_kws={"shrink": .8, "label": label})
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

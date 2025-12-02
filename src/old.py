def plot_mutual_info_features_target(X, y, n = None):
    mi = mutual_info_classif(X, y).tolist()
    if (n is None) or (n >= len(mi)):
        n = len(mi) - 1
    df_mi = pd.DataFrame({'Feature': X.columns, 'MI': mi}).sort_values('MI', ascending = False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='MI', y='Feature', data=df_mi[:n])
    plt.ylabel('')
    plt.xlabel('')
    plt.show()

    return df_mi

def heatmap_triangle(df, label):
    """
    Plot a lower triangular heatmap with values between 0 and 1.
    """
    # generate a mask for the upper triangle
    mask = np.zeros_like(df, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # set up the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 5))

    # draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(df, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
            linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8, "label": label})

    plt.show()


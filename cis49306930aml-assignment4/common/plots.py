# -*- coding: utf-8 -*-
""" CIS4930/6930 Applied ML --- plots.py
"""

import numpy as np
from matplotlib import pyplot as plt   

def heatmap(map_data, row_labels, col_labels, title=None, rot=45, fsz=None, 
                show_values=False, colorbar=True, cmap='magma'):
    # see: https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

    plt.rcParams.update({'font.size': 18})

    if fsz is not None:
        fig, ax = plt.subplots(figsize=fsz)
    else:
        fig, ax = plt.subplots()
    im = ax.imshow(map_data, cmap=cmap)

    assert len(row_labels) == map_data.shape[0]
    assert len(col_labels) == map_data.shape[1]

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(row_labels)))
    ax.set_yticks(np.arange(len(col_labels)))

    # ... and label them with the respective list entries
    ax.set_xticklabels(row_labels)
    ax.set_yticklabels(col_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=rot, ha="right",
             rotation_mode="anchor")

    if show_values:
        # Loop over data dimensions and create text annotations.
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                if i != j:
                    text = ax.text(j, i, '{:.2f}'.format(map_data[i,j]),
                               ha="center", va="center", color="w")

    if title is not None:
        ax.set_title(title)

    if colorbar:
        fig.colorbar(im)

    fig.tight_layout()
    plt.show()  


## Fill in your own implementation
def lines_plot(x_data, y_series, x_label=None, series_labels=None, y_label = None, x_lim = None, y_lim = None, title=None, fsz=None, 
                markers=None, colors=None, linestyles=None, linewidths=None):
    
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size': 18})

    plt.figure(figsize=fsz)

    num_series = y_series.shape[0]
    if series_labels is None:
        series_labels = ['Series {}'.format(i) for i in range(0, num_series)]    

    for i in range(0, num_series):
        marker = None
        if markers is not None:
            assert len(markers) == num_series
            marker = markers[i]

        color = None
        if colors is not None:
            assert len(colors) == num_series
            color = colors[i]

        linestyle = None
        if linestyles is not None:
            assert len(linestyles) == num_series
            linestyle = linestyles[i]

        linewidth = None
        if linewidths is not None:
            assert len(linewidths) == num_series
            linewidth = linewidths[i]


        plt.plot(x_data, y_series[i,:], label=series_labels[i], marker=marker, color=color,
                        linestyle=linestyle, linewidth=linewidth)

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    plt.legend()
    

## from: https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


## from: https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
def contours(ax, model, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

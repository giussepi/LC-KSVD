# -*- coding: utf-8 -*-
""" utils/plot_tools """

import copy

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from ..constants import PlotFilter
from ..core.exceptions.plot_tools import ColourMapInvalid, ColourListInvalid


class LearnedRepresentationPlotter:
    """
    Hold method to plot the representation matrix in three ways:
    - plot_basic_figure          : blue markers
    - plot_colored_basic_figure  : colored markers
    - plot_filtered_colored_image: markers filtered. See method defintion for options

    Usage:
        plotter = LearnedRepresentationPlotter(predictions=predictions, gamma=gamma,
                                           label_index=Label.INDEX, custom_colours=COLOURS)
        plotter.plot_basic_figure(show_legend=False, show_grid=False, file_saving_name='', marker=',')
        plotter.plot_colored_basic_figure(show_legend=False, show_grid=False, file_saving_name='', marker=',')
        plotter.plot_filtered_colored_image(show_legend=False, show_grid=False, file_saving_name='', filter_by=PlotFilter.UNIQUE, marker='.')

        # Functor style

        # plot_basic_figure
        LearnedRepresentationPlotter(predictions=predictions, gamma=gamma,label_index=Label.INDEX, custom_colours=COLOURS)(simple='')
        # plot_colored_basic_figure
        LearnedRepresentationPlotter(predictions=predictions, gamma=gamma,label_index=Label.INDEX, custom_colours=COLOURS)()
        # plot_filtered_colored_image
        LearnedRepresentationPlotter(predictions=predictions, gamma=gamma,label_index=Label.INDEX, custom_colours=COLOURS)(filter_by=PlotFilter.SHARED)
    """

    clusters = None

    def __init__(self, *args, **kwargs):
        """
        Verifies and initializes the instance

        Args:
            fontsize                 (int): font size
            figsize                (tuple): size of the figure
            dpi                      (int): image resolution (default 200)
            predictions       (np.ndarray): first object returned by lcksvd.classification
            gamma             (np.ndarray): second object returned by lcksvd.classification
            label_index             (dict): dictionary with keys and labels. e.g.:
                {0: 'label1', 1: 'label2', ...}
            custom_colours (list or tuple): list containig the colour names for labels. See https://matplotlib.org/3.2.1/tutorials/colors/colors.html
            colormap                 (str): matplotlib colormap name. e.g. Set1 or Dark2. see: https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
        """

        self.fontsize = kwargs.get('fontsize', 12)
        self.figsize = kwargs.get('figsize', (10.4, 4.8))
        self.dpi = kwargs.get('dpi', 200)
        self.predictions = kwargs.get('predictions', None)
        self.gamma = kwargs.get('gamma', None)
        self.label_index = kwargs.get('label_index', dict)
        custom_colours = kwargs.get('custom_colours', None)
        colormap = kwargs.get('colormap', None)

        if custom_colours:
            assert isinstance(custom_colours, (list, tuple))
            if len(custom_colours) < len(self.label_index):
                raise ColourListInvalid
            self.colours = custom_colours
        elif colormap:
            assert isinstance(colormap, str)
            try:
                cmap = cm.get_cmap(colormap)
            except ValueError as error:
                raise error
            else:
                if len(cmap.colors) < len(self.label_index):
                    raise ColourMapInvalid
                self.colours = cmap.colors[:len(self.label_index)]
        else:
            self.colours = tuple([np.random.rand(3) for _ in range(len(self.label_index))])

        assert isinstance(self.fontsize, int)
        assert isinstance(self.figsize, tuple)
        assert isinstance(self.dpi, int)
        assert self.predictions is not None
        assert isinstance(self.predictions, np.ndarray)
        assert self.gamma is not None
        assert isinstance(self.gamma, np.ndarray)

        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = plt.subplot(111)
        self.sorted_indexes = np.argsort(self.predictions)
        self.gamma = self.gamma[:, self.sorted_indexes]
        self.cluster_lengths = []

        # Getting length of clusters
        for i in range(len(self.label_index)):
            self.cluster_lengths.append(len(np.nonzero(self.predictions == i)[0]))

    def __call__(self, **kwargs):
        """ Functor """
        if 'simple' in kwargs:
            kwargs.pop('simple')
            return self.plot_basic_figure(**kwargs)
        if 'filter_by' in kwargs:
            return self.plot_filtered_colored_image(**kwargs)

        return self.plot_colored_basic_figure(**kwargs)

    def __plot_vertical_lines(self):
        """ Plots vertical lines that separates signals from different classes """
        jump = 0
        num_clusters = len(self.cluster_lengths)
        for index, length in enumerate(self.cluster_lengths):
            print("cluster length {}".format(length))
            if index != num_clusters - 1:
                plt.axvline(length + jump, c='black')
                jump += length

    def __plot_clusters(self, marker='.', markersize=12):
        """
        Plots the cluster using the provided marker

        Args:
            marker       (str): plot marker. See https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html
            markersize (float): marker size (only works when marker != ',')
        """
        assert isinstance(marker, str)
        assert isinstance(markersize, float)

        for index, item in enumerate(zip(self.clusters, self.colours)):
            np_cluster = np.array(tuple(item[0]))

            if np_cluster.any():
                plt.plot(
                    np_cluster[:, 0].tolist(), np_cluster[:, 1].tolist(),
                    marker, markersize=markersize, color=item[1],
                    label=self.label_index[index]
                )

    def __plot_and_save(self, show_legend=False, show_grid=False, file_saving_name=''):
        """
        * Adds axes names, legend (optional), grid (optional) to the plot
        * Displays the image plotted
        * Saves the image as PNG (optional)

        Args:
            show_legend      (bool): display or not the legend
            show_grid        (bool): display or not the grid
            file_saving_name (str): filename without extension
        """
        assert isinstance(show_legend, bool)
        assert isinstance(show_grid, bool)
        assert isinstance(file_saving_name, str)

        plt.xlabel('Sparse representation from test signals', fontsize=self.fontsize)
        plt.ylabel('Atoms', fontsize=self.fontsize)

        if show_legend:
            self.ax.legend(loc='upper center', bbox_to_anchor=(0., 1.02, 1., .102),
                           ncol=4, mode="expand", fancybox=True, shadow=True)
        if show_grid:
            plt.grid(True)

        plt.show()

        if file_saving_name:
            self.fig.savefig('{}.png'.format(file_saving_name), bbox_inches='tight')

    def __create_clusters(self):
        """
        Creates the clusters of learned representations
        It assumes that the same number of singals per class was provided
        """
        self.clusters = []
        jump = 0

        for length in self.cluster_lengths:
            x_s, y_s = np.nonzero(self.gamma[:, jump:jump+length].astype(bool).T)
            x_s = x_s + jump
            self.clusters.append(np.array(tuple(zip(x_s, y_s))))
            jump += length

    def plot_basic_figure(
            self, show_legend=False, show_grid=False, file_saving_name='', marker=',',
            markersize=12
    ):
        """
        Plots an image using only blue pixels

        Args:
            show_legend      (bool): display or not the legend
            show_grid        (bool): display or not the grid
            file_saving_name  (str): filename without extension
            marker            (str): plot marker. See https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html
            markersize      (float): marker size (only works when marker != ',')
        """
        assert isinstance(marker, str)
        assert isinstance(markersize, float)

        self.__plot_vertical_lines()
        nonzeros = np.nonzero(self.gamma.astype(bool).T)
        plt.plot(*nonzeros, 'b{}'.format(marker), markersize=markersize)
        self.__plot_and_save(show_legend, show_grid, file_saving_name)

    def plot_colored_basic_figure(
            self, show_legend=False, show_grid=False, file_saving_name='', marker=',',
            markersize=12
    ):
        """
        Plots and image using class-colored markers

        Args:
            show_legend      (bool): display or not the legend
            show_grid        (bool): display or not the grid
            file_saving_name  (str): filename without extension
            marker            (str): plot marker. See https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html
            markersize      (float): marker size (only works when marker != ',')
        """
        self.__plot_vertical_lines()
        self.__create_clusters()
        self.__plot_clusters(marker, markersize)
        self.__plot_and_save(show_legend, show_grid, file_saving_name)

    def __filter_clusters(self, filter_by=PlotFilter.UNIQUE):
        """
        Filters the clusters and returnss shared attoms or class unique atoms based on the
        filter_by parameter.

        Args:
            filter_by (str): plots only class unique atoms [u] or class shared atoms [s]. See constants.PlotFilter definition
        """
        assert isinstance(filter_by, str)
        assert PlotFilter.is_valid_option(filter_by)

        cleaned_clusters = []
        for index, cluster in enumerate(self.clusters):
            y_s = set(cluster[:, 1])
            for index_2, cluster_ in enumerate(self.clusters):
                if index != index_2:
                    if filter_by == PlotFilter.SHARED:
                        y_s.intersection_update(set(cluster_[:, 1]))
                    else:
                        y_s.difference_update(set(cluster_[:, 1]))

            cleaned_clusters.append(copy.deepcopy(cluster))
            cleaned_index = np.isin(cleaned_clusters[index][:, 1], tuple(y_s))
            cleaned_clusters[index] = cleaned_clusters[index][cleaned_index, :]

        self.clusters = cleaned_clusters

    def plot_filtered_colored_image(
            self, show_legend=False, show_grid=False, file_saving_name='',
            filter_by=PlotFilter.UNIQUE, marker='.', markersize=12
    ):
        """
        Filters the class-colored clusters and plots them

        Args:
            show_legend      (bool): display or not the legend
            show_grid        (bool): display or not the grid
            file_saving_name  (str): filename without extension
            filter_by (str): plots only class unique atoms [u] or class shared atoms [s]. See constants.PlotFilter definition
             marker           (str): plot marker. See https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html
            markersize (float): marker size (only works when marker != ',')
        """
        self.__plot_vertical_lines()
        self.__create_clusters()
        self.__filter_clusters(filter_by)
        self.__plot_clusters(marker, markersize)
        self.__plot_and_save(show_legend, show_grid, file_saving_name)

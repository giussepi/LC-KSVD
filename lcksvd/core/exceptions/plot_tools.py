# -*- coding: utf-8 -*-
""" core/exceptions/plot_tools """


class ColourMapInvalid(Exception):
    """
    Exception to be raised when the matplolib colormap has less colours than the labels to plot.
    """

    def __init__(self, message=''):
        """  """
        message = 'The selected matplolib colormap does not contain enough colours to plot '\
            'all the labels.'
        super().__init__(message)


class ColourListInvalid(Exception):
    """
    Exception to be raised when the provided list of colours does not have enough colours to
    plot all the labels.
    """

    def __init__(self, message=''):
        """  """
        message = 'The provided list of colours does not have enough colours to plot all '\
            'the labels.'
        super().__init__(message)

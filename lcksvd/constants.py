# -*- coding: utf-8 -*-
""" constants """


class PlotFilter:
    """ holds options for filters applied before plotting """
    UNIQUE = 'u'  # unique atoms per label
    SHARED = 's'  # shared atoms between labels

    CHOICES = (UNIQUE, SHARED)

    @classmethod
    def is_valid_option(cls, id_):
        """ Returns true if the id_ is in the choice """
        return id_ in cls.CHOICES

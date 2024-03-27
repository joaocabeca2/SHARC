# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:34:31 2017

@author: edgar
"""


class Plot(object):
    """
    Object with plot information

    Parameters
    ----------
    x : list
        x data
    y : list
        y data
    x_label : str
        x-axis label
    y_label : str
        y-axis label
    title : str
        figure title
    file_name : str
        file name with plot data
    **kwargs: dict (optional)
        x_lim: x-axis limit
        y_lim: y-axis limit
    """
    def __init__(self, x: list, y: list, x_label: str, y_label: str, 
                 title: str, file_name: str, **kwargs):


        self.x = x
        self.y = y
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.file_name = file_name

        if "x_lim" in kwargs:
            self.x_lim = kwargs["x_lim"]
        else:
            self.x_lim = None

        if "y_lim" in kwargs:
            self.y_lim = kwargs["y_lim"]
        else:
            self.y_lim = None

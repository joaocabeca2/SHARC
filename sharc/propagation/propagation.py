# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:03:12 2017

@author: edgar
"""

from abc import ABC, abstractmethod
import numpy as np

class Propagation(ABC):
    """
    Abstract base class for propagation models
    """

    def __init__(self, random_number_gen: np.random.RandomState):
        self.random_number_gen = random_number_gen
        # Inicates whether this propagation model is for links between earth and space
        self.is_earth_space_model = False

    @abstractmethod
    def get_loss(self, *args, **kwargs) -> np.array:
        pass

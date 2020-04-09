# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:33:58 2020

@author: markb
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

dat= load_iris()
X=dat.data
y=dat.target

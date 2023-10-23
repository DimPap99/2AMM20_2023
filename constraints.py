import numpy as np
from scipy.stats import skew

def var(x):
    return np.var(x)

def mean(x):
    return np.mean(x)

def std(x):
    return np.std(x)

def skewness(x):
    return skew(x)
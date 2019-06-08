# Author: Ovidiu Mura
# Date: 5/21/2019

# Reference Formula: https://anomaly.io/understand-auto-cross-correlation-normalized-shift/
# Referrence Video: https://www.youtube.com/watch?v=ngEC3sXeUb4

import math
from scipy.signal import fftconvolve
import numpy as np

# It implements the normalized, and standard correlation and an extra
# method to calculate the Discrete Linear Convolution
class Correlate:

    def __init__(self):
        self.signal_1 = None
        self.signal_2 = None

    # norm_corr(x, y) = sum(x[n]*y[n])/sqrt(sum(x^2)*sum(y^2))
    # 0 <= n <= n-1
    def normalized_correlation(self, x1, x2):
        self.signal_1 = np.array(x1)
        self.signal_2 = np.array(x2)
        x1 = np.array(x1)/10000
        x2 = np.array(x2)/10000
        return sum(x1*x2)/math.sqrt(sum(x1**2)*sum(x2**2))

    # Calculate: corr(x,y) = Sum (x[n]*y[n]), 0 <= n <= n-1
    def standard_correlate(self, x1, x2):
        self.signal_1 = x1
        self.signal_2 = x2
        x1 = np.array(x1)/10000
        x2 = np.array(x2)/10000
        lx1 = len(x1)
        lx2 = len(x2)
        size = min(lx1, lx2)
        return sum(x1[:size]*x2[:size])

    # http://pilot.cnxproject.org/content/collection/col10064/latest/module/m10087/latest
    # Calculate: (f∗g)(n) = sum(f(k)*g(n−k)), where −∞ <= k <= ∞
    def discrete_linear_convolution(self, f, g):
        return fftconvolve(f, g, mode='same')

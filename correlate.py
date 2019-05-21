# Author: Ovidiu Mura
# Date: 5/21/2019

# Reference Formula: https://anomaly.io/understand-auto-cross-correlation-normalized-shift/
# Referrence Video: https://www.youtube.com/watch?v=ngEC3sXeUb4

import math

class Correlate:

    def __init__(self):
        self.signal_1 = None
        self.signal_2 = None


    def normalized_correlation(self, x1, x2):
        self.signal_1 = x1
        self.signal_2 = x2
        return sum(x1*x2)/math.sqrt(sum(x1**2)*sum(x2**2))

    def standard_correlate(self, x1, x2):
        self.signal_1 = x1
        self.signal_2 = x2
        return sum(x1*x2)

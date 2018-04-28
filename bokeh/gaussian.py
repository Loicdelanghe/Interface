import pandas as pd
from collections import OrderedDict
from datetime import date
from sklearn import preprocessing
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


Scores = {'Roman': ["Villa des Roses","Een Ontgoocheling","De Verlossing","Lijmen","Kaas","Tsjip","Pensioen","Lijmen/Het been","De Leeuwentemmer","Het tankschip","Het dwaallicht"],
         'Jaartal': [1913,1921,1921,1924,1933,1934,1937,1938,1940,1942,1946],
         'scores': [90,78,88,86,87,94,94,88,95,89,87],
         'leeftijd': [31, 39, 39, 42, 51, 52, 55, 56, 58, 60, 64]}


leeftijd=[31, 39, 39, 42, 51, 52, 55, 56, 58, 60, 64]
Scores=[90,78,88,86,87,94,94,88,95,89,87]


L=np.array(leeftijd)
X = L.reshape(-1,1)
y = Scores




plt.figure(0)
kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)
X_ = np.linspace(0, 100)
y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9)


plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
plt.xlabel("age")
plt.ylabel("idea density")



plt.show()

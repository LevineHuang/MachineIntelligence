# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
np.random.seed(1)
N = 50
X = np.concatenate((np.random.normal(0, 1, 0.3 * N), np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
bins = np.linspace(-5, 10, 10)

fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.05, wspace=0.05)
# 直方图 1 'Histogram'
ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)
ax[0, 0].text(-3.5, 0.31, '(a) Histogram')
# 直方图 2 'Histogram, bins shifted'
ax[1, 0].hist(X[:, 0], bins=bins + 0.55, fc='#AAAAFF', normed=True)
ax[1, 0].text(-3.5, 0.31, '(b) Histogram, bins shifted 0.55')
# 直方图 2 'Histogram, bins shifted'
ax[2, 0].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', normed=True)
ax[2, 0].text(-3.5, 0.31, '(c) Histogram, bins shifted 0.75')

# 核密度估计 2 'Gaussian KDE'
kde = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(X)
log_dens = kde.score_samples(X_plot)
ax[0, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax[0, 1].text(-3.5, 0.31, '(d) Gaussian Kernel Density, bandwidth=0.25')

# 核密度估计 2 'Gaussian KDE'
kde = KernelDensity(kernel='gaussian', bandwidth=0.55).fit(X)
log_dens = kde.score_samples(X_plot)
ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax[1, 1].text(-3.5, 0.31, '(e) Gaussian Kernel Density, bandwidth=0.55')

# 核密度估计 2 'Gaussian KDE'
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)
ax[2, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax[2, 1].text(-3.5, 0.31, '(f) Gaussian Kernel Density, bandwidth=0.75')
for axi in ax.ravel():
    axi.plot(X[:, 0], np.zeros(X.shape[0])-0.01, '+k')
    axi.set_xlim(-4, 9)
    axi.set_ylim(-0.02, 0.34)
for axi in ax[:, 0]:
    axi.set_ylabel('Normalized Density')
for axi in ax[1, :]:
    axi.set_xlabel('x')
plt.show()

fig2, ax2 = plt.subplots(3, 2, sharex=True, sharey=True)
fig2.subplots_adjust(hspace=0.05, wspace=0.05)
# 核密度估计 1 'tophat KDE'
kde = KernelDensity(kernel='tophat', bandwidth=0.55).fit(X)
log_dens = kde.score_samples(X_plot)
ax2[0, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax2[0, 0].text(-3.5, 0.31, '(a) Tophat Kernel Density')

# 核密度估计 2 'Gaussian KDE'
kde = KernelDensity(kernel='gaussian', bandwidth=0.55).fit(X)
log_dens = kde.score_samples(X_plot)
ax2[0, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax2[0, 1].text(-3.5, 0.31, '(b) Gaussian Kernel Density')

# 核密度估计 2 'Gaussian KDE'
kde = KernelDensity(kernel='epanechnikov', bandwidth=0.55).fit(X)
log_dens = kde.score_samples(X_plot)
ax2[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax2[1, 0].text(-3.5, 0.31, '(c) Epanechnikov Kernel Density')

# 核密度估计 2 'Gaussian KDE'
kde = KernelDensity(kernel='exponential', bandwidth=0.55).fit(X)
log_dens = kde.score_samples(X_plot)
ax2[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax2[1, 1].text(-3.5, 0.31, '(d) Exponential Kernel Density')

# 核密度估计 2 'Gaussian KDE'
kde = KernelDensity(kernel='linear', bandwidth=0.55).fit(X)
log_dens = kde.score_samples(X_plot)
ax2[2, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax2[2, 0].text(-3.5, 0.31, '(e) Linear Kernel Density')

# 核密度估计 2 'Gaussian KDE'
kde = KernelDensity(kernel='cosine', bandwidth=0.55).fit(X)
log_dens = kde.score_samples(X_plot)
ax2[2, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax2[2, 1].text(-3.5, 0.31, '(f) Cosine Kernel Density')


for axi in ax2.ravel():
    axi.plot(X[:, 0], np.zeros(X.shape[0])-0.01, '+k')
    axi.set_xlim(-4, 9)
    axi.set_ylim(-0.02, 0.34)
for axi in ax[:, 0]:
    axi.set_ylabel('Normalized Density')
for axi in ax[1, :]:
    axi.set_xlabel('x')
plt.show()
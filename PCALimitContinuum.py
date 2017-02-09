from __future__ import division
from Main import *
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d


MinMaturity = 3
MaxMaturity = 10
FrequencyRange = np.arange(0, 0.2, 1/252)


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


inds = indices(df, lambda x: x >= MinMaturity and x <= MaxMaturity)
Maturities = np.array([(mat['Maturities'].iloc[i]) for i in inds])

c = np.log(np.array(df.iloc[:, inds]))
b = np.dot(np.ones(len(c))[:, np.newaxis], np.array(Maturities)[np.newaxis, :])
ZeroSpot = -np.divide(c, b)
ChangesZeroSpot = np.diff(ZeroSpot, axis=0)
Covariance = np.dot(ChangesZeroSpot.transpose(), ChangesZeroSpot) / ChangesZeroSpot.shape[0]
A = np.diag(np.diag(Covariance))
B = np.linalg.inv(np.linalg.cholesky(A))
Correlation = np.matmul(np.matmul(B, Covariance), B)

X = np.dot(np.ones((len(Maturities), len(Maturities))), (np.eye(len(Maturities)) * Maturities))
Y = np.dot((np.eye(len(Maturities)) * Maturities), np.ones((len(Maturities), len(Maturities))))


def FitError(gamma):
    e = (np.trace(np.matmul((np.exp(- gamma * np.abs(X - Y)) - Correlation), (np.exp(-gamma * np.abs(X-Y))-Correlation))))
    return e

Maturities = np.array(Maturities)[np.newaxis, :]
Gamma = minimize_scalar(FitError)['x']
Correlation_fit = np.exp(-Gamma * np.abs(X-Y))


fig = plt.figure(1)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(Maturities, Maturities.T, Correlation,  rstride=1, cstride=1, alpha=0.3)
ax1.ticklabel_format(style='sci', scilimits=(-3, 4), axis='both')
ax1.set_title('Empirical Correlation', fontsize=20)
ax1.set_xlabel('yrs to maturity', fontsize=14)
ax1.set_ylabel('yrs to maturity', fontsize=14)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(Maturities, Maturities.T, Correlation_fit,  rstride=1, cstride=1, alpha=0.3)
ax2.ticklabel_format(style='sci', scilimits=(-3, 4), axis='both')
ax2.set_title('Theoretical Correlation', fontsize=20)
ax2.set_xlabel('yrs to maturity', fontsize=14)
ax2.set_ylabel('yrs to maturity', fontsize=14)

fig = plt.figure(2)
EigValues = np.square(Gamma) / (np.square(Gamma) + np.square(FrequencyRange))
ax1 = plt.subplot(311)
plt.plot(FrequencyRange, EigValues, linewidth=2, color='k')
ax1.set_ylim([-0.1, 1.1])
ax1.axhline(y=0, color='k')
ax1.set_title('eigenvalues', fontsize=16)
ax1.set_xlabel('frequency', fontsize=12)
ax1.fill_between(FrequencyRange, 0, EigValues, color='gray')
R_2 = 2 / np.pi * np.arctan(FrequencyRange/Gamma)
ax2 = plt.subplot(312)
plt.plot(FrequencyRange, R_2, linewidth=2, color='k')
ax2.set_ylim([-0.1, 1.1])
ax2.axhline(y=0, color='k')
ax2.set_title('r-square', fontsize=16)
ax2.set_xlabel('frequency cut-off', fontsize=12)
ax2.fill_between(FrequencyRange, 0, R_2, color='gray')
ax3 = plt.subplot(313)
CutOff = .2
Steps = 3
MaturityDiff = np.array(range(31))[np.newaxis, :]
Omegas = np.linspace(0, CutOff, 3, endpoint=True)[np.newaxis, :]
maxY = 0
minY = 0
a = np.cos(Omegas * MaturityDiff.T)
EigFunction = np.array([a[:, i] / np.sqrt(np.diag(np.dot(a.T, a)))[i] for i in range(0, 3)])
plt.plot(MaturityDiff.T, EigFunction[0, :].T, linewidth=2, color='k')
plt.plot(MaturityDiff.T, EigFunction[1, :].T, linewidth=2, color='k')
plt.plot(MaturityDiff.T, EigFunction[2, :].T, linewidth=2, color='k')
ax3.set_title('eigenfunctions', fontsize=16)
ax2.set_xlabel('time to maturity', fontsize=12)
plt.subplots_adjust(left=0.12, bottom=0.05, right=0.90, top=0.95, wspace=0.20, hspace=0.50)
plt.show()

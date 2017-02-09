from __future__ import division
from Main import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import matplotlib.mlab as mlab


MinMaturity = 2
MaxMaturity = 10
Nodes = 1


def indices(a, func):
    return [i+1 for (i, val) in enumerate(a) if func(val)]

inds = indices(df, lambda x: x >= MinMaturity and x <= MaxMaturity)
SelectCurve = list(sorted(set(inds).intersection(set(range(0, 41, 4*Nodes)))))
Maturities = pd.Series([(mat['Maturities'].iloc[j-1]) for j in SelectCurve])
b = np.dot(np.ones(len(mat['DateString']))[:, np.newaxis], np.array(Maturities)[np.newaxis, :])
c = np.log(df.loc[:, Maturities].values)

ZeroSpot = -np.divide(c, b)
CurrentZeroSpot = ZeroSpot[-1, :]
ChangesZeroSpot = -np.diff(ZeroSpot, axis=0)
Covariance = np.dot(ChangesZeroSpot.transpose(), ChangesZeroSpot)/ChangesZeroSpot.shape[0]

[Eigenvalues, Eigenvectors] = np.linalg.eig(Covariance)
idx = Eigenvalues.argsort()[::-1]
Eigenvalues = Eigenvalues[idx]
Eigenvectors = -Eigenvectors[:, idx]

EigV1 = Eigenvectors[:, 0]
EigV2 = Eigenvectors[:, 1]
EigV3 = Eigenvectors[:, 2]
Factor1 = np.dot(ChangesZeroSpot, EigV1)
Factor2 = np.dot(ChangesZeroSpot, EigV2)
Factor3 = np.dot(ChangesZeroSpot, EigV3)
Std1 = np.sqrt(Eigenvalues[0])
Std2 = np.sqrt(Eigenvalues[1])
Std3 = np.sqrt(Eigenvalues[2])

R_Square = np.array([np.sum(Eigenvalues[:i+1])/np.sum(Eigenvalues) for i in range(0, len(Eigenvalues))])

from __future__ import division
from Main import *
import scipy as sp
from scipy import interpolate, stats
import matplotlib.pyplot as plt



MinMaturity = 2
MaxMaturity = 10
FixedRate = 0.049
Horizon = 1/12
Notional = 1000000
NumSimul = 100000

EstimationInterval = 1/52
Time2Mats = np.linspace(MinMaturity, MaxMaturity, 33)
Time2MatsHor = Time2Mats-Horizon


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

inds = indices(df, lambda x: x >= MinMaturity and x <= MaxMaturity)
Maturities = np.array([(mat['Maturities'].iloc[i]) for i in inds])

c = np.log(np.array(df.iloc[:, inds]))
b = np.dot(np.ones(len(c))[:, np.newaxis], np.array(Maturities)[np.newaxis, :])
ZeroSpot = -np.divide(c, b)
InterpZeroSpot = np.empty((678, 33))
for i in range(678):
    tck = np.array(interpolate.splrep(Maturities, ZeroSpot[i, :]))
    InterpZeroSpot[i, :] = interpolate.splev(Time2MatsHor, tck, der=0)

ChangesZeroSpot = -np.diff(InterpZeroSpot, axis=0)
Covariance = np.cov(ChangesZeroSpot, rowvar=False)
Mean = np.mean(ChangesZeroSpot, axis=0)[np.newaxis, :]
A = np.diag(np.diag(Covariance))
B = np.linalg.inv(np.linalg.cholesky(A))
Correlation = np.matmul(np.matmul(B, Covariance), B)
[EigValues, EigVectors] = np.linalg.eig(Covariance)
EigVectors[:, 0] = EigVectors[:, 0] * (-1)
EigV1 = -EigVectors[:, 0]
Ones = np.ones((len(EigV1), 1))
Alpha = np.sum(EigV1) / np.dot(EigV1.T, EigV1)
Var1 = EigValues[0] / np.square(Alpha)
FactorLoadings = np.array([Ones, EigVectors[:,1][:,np.newaxis], EigVectors[:,2][:,np.newaxis]]).reshape(3,33).T
FactorCovariance = np.diag(np.array([Var1, EigValues[1], EigValues[2]]))

Coefficients = FixedRate * 0.25 * Ones
Coefficients[-1] = Coefficients[-1] + 1
Coefficients[0] = -1

tck = np.array(interpolate.splrep(Maturities, ZeroSpot[0, :]))
CurrentZeroSpot = interpolate.splev(Time2Mats, tck, der=0)[np.newaxis, :]
BondPrices = (np.exp(-CurrentZeroSpot * Time2Mats) * Notional)
CurrentZeroSpotHor = InterpZeroSpot[0, :][np.newaxis, :]
BondPricesSlide = (np.exp(-CurrentZeroSpotHor * Time2MatsHor) * Notional)

SwapCurrentValue = np.dot(BondPrices, Coefficients)
Slide = np.dot(BondPricesSlide, Coefficients)
PVBP = np.dot((BondPricesSlide * Time2MatsHor), Coefficients)
Convexity = 0.5 * np.dot((BondPricesSlide * Time2MatsHor[np.newaxis, :] * Time2MatsHor[np.newaxis, :]), Coefficients)


Mu = np.zeros(33)
Sigma = np.dot(np.dot(FactorLoadings, FactorCovariance), FactorLoadings.T) * Horizon / EstimationInterval
X = np.random.multivariate_normal(Mu, Sigma, int(NumSimul/2))
X = np.concatenate((X, -X), 0)
d = (np.dot(np.ones((NumSimul, 1)), Time2MatsHor[np.newaxis, :]))
e = X * d
SwapHorizonValue = np.dot(np.exp(e), (Coefficients.T * BondPrices).T)

BottomRangeP = -Notional / 30 * np.sqrt(Horizon / EstimationInterval)
TopRangeP = Notional / 30 * np.sqrt(Horizon / EstimationInterval)
Step = (TopRangeP - BottomRangeP) / 1000
PriceRange = np.append(np.arange(BottomRangeP, TopRangeP, Step), TopRangeP)
PriceDensOrdOne = sp.stats.norm(Slide, np.sqrt((PVBP**2 * Var1 * Horizon / EstimationInterval))).pdf(PriceRange)


g = (Slide - PVBP * PVBP / (4 * Convexity))
q = Convexity * Var1 * Horizon / EstimationInterval
DegFreedom = 1
NonCentrality = (-PVBP / (2 * Convexity * np.sqrt(Var1 * Horizon / EstimationInterval)))**2
PriceDensOrdTwo = 1 / q * sp.stats.ncx2(DegFreedom, NonCentrality).pdf((PriceRange - g) / q)


fig = plt.figure(1)
ax1 = plt.subplot(311)
plt.plot(PriceRange, np.squeeze(PriceDensOrdOne), linewidth=2, color='k')
plt.gca().xaxis.grid(True)
ax1.fill_between(PriceRange, 0, np.squeeze(PriceDensOrdOne), color='gray')
ax1.scatter(SwapCurrentValue, 0, color='k')
ax1.scatter(Slide, 0, color='k', marker='s')
ax1.axhline(y=0, color='k')
ax1.set_title('parallel shift, duration approx.', fontsize=16)
ax1.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
ax2 = plt.subplot(312, sharex=ax1)
plt.plot(PriceRange, np.squeeze(PriceDensOrdTwo), linewidth=2, color='k')
plt.gca().xaxis.grid(True)
ax2.fill_between(PriceRange, 0, np.squeeze(PriceDensOrdOne), color='gray')
ax2.scatter(SwapCurrentValue, 0, color='k')
ax2.scatter(Slide, 0, color='k', marker='s')
ax2.axhline(y=0, color='k')
ax2.set_title('parallel shift, duration-convexity approx.', fontsize=16)
ax2.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
ax3 = plt.subplot(313, sharex=ax1)
NumBins = int(round(10 * np.log(NumSimul)))
n, bins, patches = plt.hist(SwapHorizonValue, NumBins, normed=1, facecolor='grey', alpha=0.75)
plt.gca().xaxis.grid(True)
ax3.set_title('shift-steepening-bending, no pricing approx.', fontsize=16)
ax3.scatter(SwapCurrentValue, 0, color='k')
ax3.scatter(Slide, 0, color='k', marker='s')
ax3.axhline(y=0, color='k')
ax3.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
plt.xlim(-70000, 70000)
plt.subplots_adjust(left=0.12, bottom=0.05, right=0.90, top=0.95, wspace=0.20, hspace=0.50)
plt.show()

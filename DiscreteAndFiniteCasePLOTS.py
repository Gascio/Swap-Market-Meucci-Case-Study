from __future__ import division
from DiscreteAndFiniteCase import *

L = np.size(ChangesZeroSpot, 1)
Y = 10000 * ChangesZeroSpot[:, [0, int(np.round(L/2)), L-1]]
S = np.cov(Y, rowvar=False)
[EVls, EVrs] = np.linalg.eig(S)
EVls = np.atleast_2d(EVls).reshape(3, 1)
EVrs = -EVrs

u = np.linspace(0.5 * np.pi, 1.5 * np.pi, 100)
v = np.linspace(0, np.pi, 25)
Scale = 3
x = Scale * np.outer(np.cos(u), np.sin(v))
y = Scale * np.outer(np.sin(u), np.sin(v))
z = Scale * np.outer(np.ones(np.size(u)), np.cos(v))

for i in range(len(u)):
    for j in range(len(v)):
        [x[i, j], y[i, j], z[i, j]] = np.dot(np.dot(EVrs, (np.eye(3) * np.sqrt(EVls))), [x[i, j], y[i, j], z[i, j]])

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot_wireframe(x, y, z,  rstride=1, cstride=1, color='grey')
ax.set_aspect('equal')

PrincipalAxes = np.dot(EVrs, (np.eye(3) * np.sqrt(EVls))) * Scale

ax.plot([0, PrincipalAxes[0, 0]], [0, PrincipalAxes[1, 0]], [0, PrincipalAxes[2, 0]], color='r', linewidth=2)
ax.plot([0, PrincipalAxes[0, 1]], [0, PrincipalAxes[1, 1]], [0, PrincipalAxes[2, 1]], color='r', linewidth=2)
ax.plot([0, PrincipalAxes[0, 2]], [0, PrincipalAxes[1, 2]], [0, PrincipalAxes[2, 2]], color='r', linewidth=2)
ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2])
fig.suptitle('Location-Dispersion Ellipsoid', fontsize=20)
ax.set_xlabel('change in 2yr yield', fontsize=16)
ax.set_ylabel('change in 5yr yield', fontsize=16)
ax.set_zlabel('change in 10yr yield', fontsize=16)


fig = plt.figure(2)
Maturities = np.atleast_2d(Maturities).reshape(9, 1)
ax = fig.gca(projection='3d')
ax.plot_surface(Maturities, Maturities.T, Covariance,  rstride=1, cstride=1, alpha=0.3)
ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='both')
fig.suptitle('Covariance Matrix', fontsize=20)
ax.set_xlabel('yrs to maturity', fontsize=16)
ax.set_ylabel('yrs to maturity', fontsize=16)

fig = plt.figure(3)
e = np.arange(9) + 1
ax1 = plt.subplot(311)
plt.bar(e, Eigenvalues)
ax1.set_title('eigenvalues')
ax1.set_xlabel('# factor', fontsize=12)
ax1.ticklabel_format(style='sci', scilimits=(-3, 4), axis='both')
ax2 = plt.subplot(312)
plt.bar(e, R_Square)
ax2.set_title('r-square')
ax2.set_xlabel('cumulative # factors', fontsize=12)
ax2.set_ylim([0.90, 1.00])
ax3 = plt.subplot(313)
plt.plot(Maturities, EigV1, label='#1', linewidth=2, color='k')
plt.plot(Maturities, EigV2, label='#2', linewidth=2, color='k')
plt.plot(Maturities, EigV3, label='#3', linewidth=2, color='k')
ax3.set_title('eigenvectors')
plt.text(3.2, 0, '#3')
plt.text(6, 0, '#2')
plt.text(8, 0.4, '#1')
plt.subplots_adjust(left=0.12, bottom=0.05, right=0.90, top=0.95, wspace=0.20, hspace=0.50)

fig = plt.figure(4)
nbins = int(round(9 * np.log(len(Factor1))))
ax1 = plt.subplot(311)
n, bins, patches = plt.hist(10000*Factor1, nbins, normed=1, facecolor='grey', alpha=0.75)
y = mlab.normpdf(bins, np.mean(Factor1) * 10000, 10000 * np.sqrt(Eigenvalues[0]))
l = plt.plot(bins, y, 'k-', linewidth=2)
ax1.set_title('1st factor: shift')
ax1.ticklabel_format(style='sci', scilimits=(-1, 4), axis='both')
ax2 = plt.subplot(312)
n, bins, patches = plt.hist(10000*Factor2, nbins, normed=1, facecolor='grey', alpha=0.75)
y = mlab.normpdf(bins, np.mean(Factor2) * 10000, 10000 * np.sqrt(Eigenvalues[1]))
l = plt.plot(bins, y, 'k-', linewidth=2)
ax2.set_title('2nd factor: sttepening')
ax3 = plt.subplot(313)
n, bins, patches = plt.hist(10000*Factor3, nbins, normed=1, facecolor='grey', alpha=0.75)
y = mlab.normpdf(bins, np.mean(Factor3) * 10000, 10000 * np.sqrt(Eigenvalues[2]))
l = plt.plot(bins, y, 'k-', linewidth=2)
ax3.set_title('3rd factor: bending')
plt.subplots_adjust(left=0.12, bottom=0.05, right=0.90, top=0.95, wspace=0.20, hspace=0.50)

fig = plt.figure(5)
ax1 = plt.subplot(311)
Shift = 3 * Std1 * EigV1
plt.plot(Maturities, 100 * CurrentZeroSpot, linewidth=2, color='k')
plt.plot(Maturities, 100 * (CurrentZeroSpot - Shift), linewidth=2, color='k')
plt.plot(Maturities, 100 * (CurrentZeroSpot + Shift), linewidth=2, color='k')
ax1.set_ylim([7.5, 9.5])
ax1.set_title('1st factor: shift')
ax2 = plt.subplot(312)
Slope = 3 * Std2 * EigV2
plt.plot(Maturities, 100 * CurrentZeroSpot, linewidth=2, color='k')
plt.plot(Maturities, 100 * (CurrentZeroSpot + Slope), linewidth=2, color='k')
plt.plot(Maturities, 100 * (CurrentZeroSpot - Slope), linewidth=2, color='k')
ax2.set_ylim([8, 9])
ax2.set_title('2nd factor: steepening')
ax3 = plt.subplot(313)
Hump = 3 * Std2 * EigV3
plt.plot(Maturities, 100 * CurrentZeroSpot, linewidth=2, color='k')
plt.plot(Maturities, 100 * (CurrentZeroSpot + Hump), linewidth=2, color='k')
plt.plot(Maturities, 100 * (CurrentZeroSpot - Hump), linewidth=2, color='k')
ax3.set_ylim([8, 9])
ax3.set_title('3rd factor: bending')
plt.subplots_adjust(left=0.12, bottom=0.05, right=0.90, top=0.95, wspace=0.20, hspace=0.50)

plt.show()

from __future__ import division
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime, timedelta

mat = loadmat('DB_SwapCurve.mat')
mdata = mat['DF']
mat['DateString'] = pd.Series([datetime.fromordinal(date) - timedelta(days=366) for date in mat['Dates']])
mat['Maturities'] = pd.Series(np.linspace(0.25, 10, num=40))
df = pd.DataFrame(mdata, index=mat['DateString'], columns=mat['Maturities'])

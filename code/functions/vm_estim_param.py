#region Example
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html#scipy.optimize.lsq_linear
import numpy as np
from scipy.sparse import rand
from scipy.optimize import lsq_linear

np.random.seed(0)
m = 200
n = 100

A = rand(m, n, density=1e-4)
b = np.random.randn(m)
lb = np.random.randn(n)
ub = lb + 1

res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=1)
# # may vary
# The relative change of the cost function is less than `tol`.
# Number of iterations 16, initial cost 1.5039e+04, final cost 1.1112e+04,
# first-order optimality 4.66e-08.
res.x # estimated parameters
#endregion

#region Application to prices
# A := m*n variables (m rows => timestep, n columns => variables with first as intercept)
# b := m observed prices (on a single row)

A = [[1, 0],
    [1, 1],
    [1, 2]]

b = np.array([1.5,3.5,5.5])

lb = np.array([0,1])
ub = np.array(lb + 2)

res = lsq_linear(A, b, lsmr_tol='auto', verbose=1)
res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=1)

res.x

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_numpy.html

#endregion

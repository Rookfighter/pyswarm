import pyswarm

###############################################################################

print('*' * 65)
print('Example minimization of 4th-order banana function (no constraints)')
def myfunc(x):
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2 * x2 * x1**2 + x2**2 + x1**2 - 2 * x1 + 5

bounds = [(-3, 2), (-1, 6)]

xopt1, fopt1 = pyswarm.pso.minimize(myfunc, bounds, disp=True)

print('The optimum is at:')
print('    {}'.format(xopt1))
print('Optimal function value:')
print('    myfunc: {}'.format(fopt1))

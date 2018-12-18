from functools import partial
import numpy as np

def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)


def minimize(func, bounds, args=(),
swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
xeps=1e-8, feps=1e-8, disp=False, processes=1, callback=None):
    """
    Perform a particle swarm optimization (PSO)

    Parameters
    ==========
    func : function
        The function to be minimized
    bounds : array
        list of pairs with lower and upper bounds of the state variable

    Optional
    ========
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    xeps : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    feps : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    disp : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and
        constraints (default: 1)
    callback : function

    Returns
    =======
    xg : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``xg``
    p : array
        The best known position per particle
    pf: arrray
        The objective values at each position in p

    """
    bounds = np.array(bounds)

    assert(hasattr(func, '__call__'))
    assert(callback is None or hasattr(callback, '__call__'))
    assert(bounds.shape[1] == 2)
    assert(np.all(bounds[:, 1] > bounds[:, 0]))

    lb, ub = bounds[:, 0], bounds[:, 1]
    bound_diff = ub - lb
    vhigh = np.abs(bound_diff)
    vlow = -vhigh
    vdiff = vhigh - vlow

    # Initialize objective function
    obj = partial(_obj_wrapper, func, args)

    # Initialize the multiprocessing module if necessary
    mp_pool = None
    chunksize = None
    if processes > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(processes)
        chunksize = np.ceil(swarmsize / (multiprocessing.cpu_count() * 10))
        chunksize = int(chunksize)

    # Initialize the particle swarm
    S = swarmsize
    D = bounds.shape[0]  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros((S, D))  # particle velocities
    p = np.zeros((S, D))  # best particle positions
    fx = np.zeros(S)  # current particle function values
    fp = np.ones(S) * np.inf  # best particle function values
    xg = np.zeros(D)  # best swarm position
    fg = np.inf  # best swarm position starting value
    fdiff = np.inf
    xdiff = np.inf
    xeps *= xeps

    # Initialize the particle's position
    x *= bound_diff
    x += lb

    # Calculate objective and constraints for each particle
    if mp_pool is not None:
        fx = np.array(mp_pool.map(obj, x, chunksize)).flatten()
    else:
        for i in range(x.shape[0]):
            fx[i] = obj(x[i, :])

    # Store particle's best position
    p[:, :] = x[:, :]
    fp[:] = fx[:]

    # Update swarm's best position
    i_min = np.argmin(fp)
    fg = fp[i_min]
    xg[:] = p[i_min, :]

    # Initialize the particle's velocity
    v = np.random.rand(S, D)
    v *= vdiff
    v += vlow

    # Iterate until termination criterion met
    it = 1
    while it <= maxiter and fdiff < feps and xdiff < xeps:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

        # Update the particles velocities
        rp *= p - x
        rp *= phip
        rg *= xg - x
        rg *= phig
        v *= omega
        v += rp
        v += rg

        # Update the particles' positions
        x += v
        # Correct for bound violations
        maskl = x < lb
        masku = x > ub

        x *= ~np.logical_or(maskl, masku)
        x += lb * maskl
        x += ub * masku

        # Update objective
        if mp_pool is not None:
            fx = np.array(mp_pool.map(obj, x, chunksize)).flatten()
        else:
            for i in range(x.shape[0]):
                fx[i] = obj(x[i, :])

        # Store particle's best position
        i_update = fx < fp
        p[i_update, :] = x[i_update, :]
        fp[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            xdiff = np.sum((xg - p[i_min, :])**2)
            fdiff = np.abs(fg - fp[i_min])

            xg[:] = p[i_min, :]
            fg = fp[i_min]

        if disp:
            tmp = [
                'it={}'.format(it),
                'fdiff={:.06f}'.format(fdiff),
                'xdiff={:.06f}'.format(xdiff),
                'f={:.06f}'.format(fg),
                'x={}'.format(' '.join(['{:.06f}'.format(v) for v in xg])),
            ]
            print('\t'.join(tmp))

        if callback is not None:
            callback(xg, fg, p, fp)

        it += 1

    return xg, fg, p, fp

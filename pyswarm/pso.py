from functools import partial
import numpy as np

def _obj_wrapper(func, args, state):
    return func(state, *args)


def minimize(func, bounds, args=(),
swarm=None, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
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
    state_g: array
        The swarm's best known position (optimal design)
    fval: scalar
        The objective value at ``state_g``
    state_p: array
        The best known position per particle
    fval_p: arrray
        The objective values at each position in state_p

    """
    bounds = np.array(bounds)

    if swarm is None:
        swarm = int(20 * bounds.shape[0])
    if type(swarm) is int:
        swarm = np.random.rand(swarm, bounds.shape[0])
    if type(swarm) is list:
        swarm = np.array(swarm)
    assert(type(swarm) is np.ndarray)
    swarmsize = swarm.shape[0]

    assert(hasattr(func, '__call__'))
    assert(callback is None or hasattr(callback, '__call__'))
    assert(bounds.shape[1] == 2)
    assert(np.all(bounds[:, 1] > bounds[:, 0]))

    low_bound, up_bound = bounds[:, 0], bounds[:, 1]
    vhigh = np.abs(up_bound - low_bound)
    vlow = -vhigh

    # Initialize objective function
    obj = partial(_obj_wrapper, func, args)

    # Initialize the multiprocessing module if necessary
    mp_pool = None
    chunksize = None
    if processes > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(processes)
        chunksize = np.ceil(swarmsize / (processes * 10))
        chunksize = int(chunksize)

    # Initialize the particle swarm
    S = swarmsize
    D = bounds.shape[0]  # the number of dimensions each particle has
    state = swarm.copy()  # particle positions
    vel = np.zeros((S, D))  # particle velocities
    state_p = np.zeros((S, D))  # best particle positions
    fval = np.zeros(S)  # current particle function values
    fval_p = np.ones(S) * np.inf  # best particle function values
    state_g = np.zeros(D)  # best swarm position
    fval_g = np.inf  # best swarm position starting value
    fdiff = np.inf
    xdiff = np.inf
    xeps *= xeps

    # Initialize the particle's position
    state *= up_bound - low_bound
    state += low_bound

    # Calculate objective and constraints for each particle
    if mp_pool is not None:
        fval = np.array(mp_pool.map(obj, state, chunksize)).flatten()
    else:
        for i in range(state.shape[0]):
            fval[i] = obj(state[i, :])

    # Store particle's best position
    state_p[:, :] = state[:, :]
    fval_p[:] = fval[:]

    # Update swarm's best position
    i_min = np.argmin(fval_p)
    fval_g = fval_p[i_min]
    state_g[:] = state_p[i_min, :]

    # Initialize the particle's velocity
    vel = np.random.rand(S, D)
    vel *= vhigh - vlow
    vel += vlow

    # Iterate until termination criterion met
    it = 1
    while it <= maxiter and fdiff > feps and xdiff > xeps:
        rp = np.random.rand(S, D)
        rg = np.random.rand(S, D)

        # Update the particles velocities
        rp *= state_p - state
        rp *= phip
        rg *= state_g - state
        rg *= phig
        vel *= omega
        vel += rp
        vel += rg

        # Update the particles' positions
        state += vel
        # Correct for bound violations
        maskl = state < low_bound
        masku = state > up_bound

        state *= ~np.logical_or(maskl, masku)
        state += low_bound * maskl
        state += up_bound * masku

        # Update objective
        if mp_pool is not None:
            fval = np.array(mp_pool.map(obj, state, chunksize)).flatten()
        else:
            for i in range(state.shape[0]):
                fval[i] = obj(state[i, :])

        # Store particle's best position
        i_update = fval < fval_p
        state_p[i_update, :] = state[i_update, :]
        fval_p[i_update] = fval[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(fval_p)
        if fval_p[i_min] < fval_g:
            xdiff = np.sum((state_g - state_p[i_min, :])**2)
            fdiff = np.abs(fval_g - fval_p[i_min])

            state_g[:] = state_p[i_min, :]
            fval_g = fval_p[i_min]

        if disp:
            tmp = [
                'it={}'.format(it),
                'fdiff={:.06f}'.format(fdiff),
                'xdiff={:.06f}'.format(xdiff),
                'f={:.06f}'.format(fval_g),
                'state={}'.format(' '.join(['{:.06f}'.format(v)
                    for v in state_g])),
            ]
            print('\t'.join(tmp))

        if callback is not None:
            callback(state_g, fval_g, state_p, fval_p)

        it += 1

    return state_g, fval_g

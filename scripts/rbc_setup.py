"""Script to set up RBC simulations in Dedalus."""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


def build_solver(aspect, Nx, Nz, Rayleigh, Prandtl, hyper_coef):
    """
    Builds a Dedalus solver for RBC.

    Args:
        aspect: Domain aspect ratio.
        Nx: Number of horizontal modes.
        Nz: Number of vertical modes.
        Rayleigh: Rayleigh number.
        Prandtl: Prandtl number.
        hyper_coef: Dimensionless hyperviscosity coefficient (in units of
            (layer thickness)^(hyper_order - 2) * (normal kinematic viscosity))

    Returns:
        Solver object.
    """

    # Log the parameters
    logger.info('Building solver. Parameters:')
    logger.info(f'\tRa = {Rayleigh:.2e}')
    logger.info(f'\tPr = {Prandtl:.3f}')
    logger.info(f'\thyperviscosity = {hyper_coef:.3f}')
    logger.info(f'\taspect = {aspect:.1f}')
    logger.info(f'\tNx = {Nx:d}')
    logger.info(f'\tNz = {Nz:d}')

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(
        coords['x'], size=Nx, bounds=(0, aspect), dealias=3/2)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, 1), dealias=3/2)

    # Fields
    pi = dist.Field(name='pi', bases=(xbasis, zbasis))
    theta = dist.Field(name='theta', bases=(xbasis, zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
    tau_pi = dist.Field(name='tau_pi')
    tau_theta1 = dist.Field(name='tau_theta1', bases=xbasis)
    tau_theta2 = dist.Field(name='tau_theta2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    x, z = dist.local_grids(xbasis, zbasis)
    _, z_hat = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    taper = dist.Field(bases=zbasis)
    taper['g'] = 1 - (2*z - 1)**10

    def lift(A):
        """Multiplies by the highest Chebyshev polynomial."""
        return d3.Lift(A, lift_basis, -1)

    grad_u = d3.grad(u) - z_hat*lift(tau_u1)
    grad_theta = d3.grad(theta) - z_hat*lift(tau_theta1)

    # Problem
    problem = d3.IVP(
        [pi, theta, u, tau_pi, tau_theta1, tau_theta2, tau_u1, tau_u2],
        namespace=locals(),
    )
    # Momentum equation
    problem.add_equation(
        'Rayleigh/Prandtl*dt(u) '
        '- div(grad_u) '
        '+ hyper_coef*taper*lap(lap(u))'
        '+ grad(pi) '
        '- theta*z_hat '
        '+ lift(tau_u2) '
        '= - Rayleigh/Prandtl*u@grad_u'
    )
    # Energy equation
    problem.add_equation(
        'Rayleigh*dt(theta) '
        '- div(grad_theta) '
        '+ hyper_coef*taper*lap(lap(theta))'
        '+ lift(tau_theta2) '
        '= -Rayleigh*u@grad_theta'
    )
    # Continuity equation
    problem.add_equation('trace(grad_u) + tau_pi = 0')

    # No-slip, isothermal boundary conditions
    problem.add_equation('u(z=0) = 0')
    problem.add_equation('u(z=1) = 0')
    problem.add_equation('theta(z=0) = 1/2')
    problem.add_equation('theta(z=1) = -1/2')
    # Pressure gauge condition
    problem.add_equation('integ(pi) = 0')

    # Solver
    solver = problem.build_solver(d3.RK222)
    return solver


def get_local_grid(field, axis):
    """
    Convenience function to get the coordinate grid for a Field.

    Args:
        field: Dedalus Field object.
        axis: String specifying the desired axis (i.e. 'x' or 'z')

    Returns:
        Coordinate array.
    """

    return field.dist.local_grid(field.get_basis(field.domain.get_coord(axis)))


def get_field(solver, name):
    """
    Convenience function to retrieve a Field from a Solver.

    Args:
        solver: Dedalus Solver object.
        name: Field name.

    Returns:
        Dedalus Field object.
    """

    names = [x.name for x in solver.state]
    return solver.state[names.index(name)]


def set_initial_conditions(solver, type, restart_file=None):
    """
    Sets the initial conditions for the solver.

    Args:
        solver: Dedalus Solver object.
        type: String specifying the type of initial condition. Options:
            'random_theta', 'wavy_theta', 'restart'.
        restart_file: Path to restart file when type == 'restart'.
    """

    if type == 'random_theta':
        # Log the noise amplitude
        logger.info(
            'Initial condition: random_theta with sigma = 1e-2'
        )

        # Fill the initial theta array with random noise
        theta = get_field(solver, 'theta')
        theta.fill_random('g', distribution='normal', scale=1e-2)

        # Reduce the amplitude of the noise near the walls
        z = get_local_grid(theta, 'z')
        theta['g'] *= 4*z*(1 - z)
    elif type == 'wavy_theta':
        logger.info('Initial condition: wavy_theta')
        theta = get_field(solver, 'theta')
        x = get_local_grid(theta, 'x')
        z = get_local_grid(theta, 'z')

        k_base = 4
        k_perturb = 41
        aspect = theta.domain.bases_by_axis[0].bounds[1]

        def exp_term(x):
            base_wave = np.cos(2*np.pi*k_base*x/aspect)
            perturb_wave = np.cos(2*np.pi*k_perturb*(x - 0.25)/aspect)
            mod_wave = np.cos(np.pi*x/2)**4
            return base_wave + 0.1*perturb_wave*mod_wave

        exp = 6 - np.where(z < 0.5, exp_term(x), exp_term(x - 1))
        theta['g'] = -0.5*np.sign(z - 0.5)*np.abs(2*z - 1)**exp
    elif type == 'restart' and restart_file:
        logger.info(f'Initial condition: restart from {restart_file}')
        solver.load_state(restart_file)
    else:
        raise ValueError('Invalid initial condition specification.')


def add_file_handler(solver, data_dir, save_interval, restart=False):
    """
    Tells the solver to save output to a file.

    Args:
        solver: Dedalus Solver object.
        data_dir: Directory for data output.
        save_interval: Number of time steps between saves.
        restart: Whether or not this is a restart run (default False).
    """

    if restart:
        def save_schedule(iteration, **_):
            save = (
                (iteration % save_interval == 0)
                and (iteration != solver.initial_iteration)
            )
            return save
    else:
        def save_schedule(iteration, **_):
            return iteration % save_interval == 0

    snapshots = solver.evaluator.add_file_handler(
        data_dir, custom_schedule=save_schedule, max_writes=1000,
        mode=('append' if restart else 'overwrite'),
    )
    u = get_field(solver, 'u')
    theta = get_field(solver, 'theta')
    snapshots.add_tasks([u, theta], layout='g')
    logger.info(f'Output: {data_dir:s}')
    logger.info(f'Logging interval: {save_interval:d}*dt')

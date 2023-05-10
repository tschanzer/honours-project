"""Script to set up RBC simulations in Dedalus."""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


def build_solver(Lx, Nx, Nz, Rayleigh, Prandtl):
    """
    Builds a Dedalus solver for RBC.

    Args:
        Lx: Dimensionless domain length (relative to thickness).
        Nx: Number of horizontal modes.
        Nz: Number of vertical modes.
        Rayleigh: Rayleigh number.
        Prandtl: Prandtl number.

    Returns:
        Solver object.
    """

    # Log the parameters
    logger.info('Building solver. Parameters:')
    logger.info(f'\tRa = {Rayleigh:.2e}')
    logger.info(f'\tPr = {Prandtl:.3f}')
    logger.info(f'\tLx = {Lx:.1f}')
    logger.info(f'\tNx = {Nx:d}')
    logger.info(f'\tNz = {Nz:d}')

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=3/2)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, 1), dealias=3/2)

    # Fields
    pi = dist.Field(name='pi', bases=(xbasis,zbasis))
    theta = dist.Field(name='theta', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_pi = dist.Field(name='tau_pi')
    tau_theta1 = dist.Field(name='tau_theta1', bases=xbasis)
    tau_theta2 = dist.Field(name='tau_theta2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    x, z = dist.local_grids(xbasis, zbasis)
    _, z_hat = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) - z_hat*lift(tau_u1)
    grad_theta = d3.grad(theta) - z_hat*lift(tau_theta1)

    # Problem
    problem = d3.IVP(
        [pi, theta, u, tau_pi, tau_theta1, tau_theta2, tau_u1, tau_u2],
        namespace=locals(),
    )
    # Momentum equation
    problem.add_equation(
        'Prandtl**(-1)*dt(u) - div(grad_u) + grad(pi) - theta*z_hat'
        ' + lift(tau_u2) = - Prandtl**(-1)*u@grad_u'
    )
    # Energy equation
    problem.add_equation(
        'dt(theta) - div(grad_theta) - Rayleigh*u@z_hat + lift(tau_theta2) '
        '= -u@grad_theta'
    )
    # Continuity equation
    problem.add_equation('trace(grad_u) + tau_pi = 0')

    # No-slip, isothermal boundary conditions
    problem.add_equation('u(z=0) = 0')
    problem.add_equation('u(z=1) = 0')
    problem.add_equation('theta(z=0) = 0')
    problem.add_equation('theta(z=1) = 0')
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


def set_initial_conditions(solver, type, **kwargs):
    """
    Sets the initial conditions for the solver.

    Args:
        solver: Dedalus Solver object.
        type: String specifying the type of initial condition. Options:
            'random_theta'.
        **kwargs: Parameters specific to the type of initial condition.
    """

    if type == 'random_theta' and 'sigma' in kwargs.keys():
        # Log the noise amplitude
        logger.info(
            'Initial condition: random_theta with sigma = '
            f"{kwargs['sigma']:.3e}"
        )

        # Fill the initial theta array with random noise
        theta = get_field(solver, 'theta')
        theta.fill_random(
            'g', distribution='normal', scale=kwargs['sigma'],
        )

        # Reduce the amplitude of the noise near the walls
        z = get_local_grid(theta, 'z')
        theta['g'] *= 4*z*(1 - z)
    else:
        raise ValueError('Invalid initial condition specification.')

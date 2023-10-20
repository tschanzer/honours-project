"""Module for coarse-graining high-resolution data."""

import dedalus.public as d3
import numpy as np


class CoarseGrainer:
    """Diffusion model for coarse-graining high-resolution data."""
    def __init__(self, aspect, Nx, Nz):
        """
        Builds the solver.

        Args:
            aspect: Domain aspect ratio.
            Nx: Number of horizontal modes.
            Nz: Number of vertical modes.
        """

        self.Nx = Nx
        self.Nz = Nz

        # Fundamental objects
        coords = d3.CartesianCoordinates('x', 'z')
        dist = d3.Distributor(coords, dtype=np.float64)
        xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, aspect))
        zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, 1))
        _, z_hat = coords.unit_vector_fields(dist)

        # Fields
        self.fields = {
            'u': dist.VectorField(
                coords, name='u', bases=(xbasis, zbasis)),
            'theta': dist.Field(
                name='theta', bases=(xbasis, zbasis)),
            'pi': dist.Field(
                name='pi', bases=(xbasis, zbasis)),
            'tau_pi': dist.Field(
                name='tau_pi'),
            'tau_theta1': dist.Field(
                name='tau_theta1', bases=xbasis),
            'tau_theta2': dist.Field(
                name='tau_theta2', bases=xbasis),
            'tau_u1': dist.VectorField(
                coords, name='tau_u1', bases=xbasis),
            'tau_u2': dist.VectorField(
                coords, name='tau_u2', bases=xbasis),
        }

        # Substitutions
        def lift(field):
            """Multiplies by the highest Chebyshev polynomial."""
            return d3.Lift(field, zbasis.derivative_basis(1), -1)

        grad_u = d3.grad(self.fields['u']) - z_hat*lift(self.fields['tau_u1'])
        grad_theta = (
            d3.grad(self.fields['theta'])
            - z_hat*lift(self.fields['tau_theta1'])
        )

        # Problem
        problem = d3.IVP(list(self.fields.values()))
        problem.add_equation((
            (
                d3.dt(self.fields['u'])
                - d3.div(grad_u)
                + d3.grad(self.fields['pi'])
                + lift(self.fields['tau_u2'])
            ),
            0,
        ))
        problem.add_equation((
            (
                d3.dt(self.fields['theta'])
                - d3.div(grad_theta)
                + lift(self.fields['tau_theta2'])
            ),
            0,
        ))
        problem.add_equation((d3.trace(grad_u) + self.fields['tau_pi'], 0))
        problem.add_equation((d3.integ(self.fields['pi']), 0))
        problem.add_equation((self.fields['u'](z=0), 0))
        problem.add_equation((self.fields['u'](z=1), 0))
        problem.add_equation((self.fields['theta'](z=0), 1/2))
        problem.add_equation((self.fields['theta'](z=1), -1/2))

        # Solver
        self.solver = problem.build_solver(d3.RK222)

    def run(self, u, w, theta, time, dt, to_shape=None):
        """
        Smooths a state by running the solver.

        Args:
            u, w, theta: np.ndarray.
            time: Run time (i.e. amount of smoothing)
            to_shape: Output resolution.

        Returns:
            Smoothed u, w and theta.
        """

        # Zero all fields just to be safe
        for field in self.fields:
            self.fields[field]['g'] = 0

        # Load initial condition
        self.fields['u'].load_from_global_grid_data(np.stack([u, w]))
        self.fields['theta'].load_from_global_grid_data(theta)

        # Run until stop time is reached
        run_time = 0
        while run_time < time:
            self.solver.step(dt)
            run_time += dt

        # Return final state
        if to_shape is None:
            to_shape = (self.Nx, self.Nz)
        new_scales = (to_shape[0]/self.Nx, to_shape[1]/self.Nz)
        self.fields['u'].change_scales(new_scales)
        self.fields['theta'].change_scales(new_scales)
        u_data = self.fields['u'].allgather_data('g')
        theta_data = self.fields['theta'].allgather_data('g')
        return u_data[0,:,:], u_data[1,:,:], theta_data

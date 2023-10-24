"""Module for simulating Rayleigh-Bénard convection."""

from functools import wraps
import logging
import os

import dask
import dedalus.public as d3
import numpy as np
import pandas as pd
import xarray as xr

from modules import coarse_graining

logger = logging.getLogger(__name__)
# pylint: disable=no-member, too-many-arguments, too-many-instance-attributes


class ModelBase:
    """
    Base class for Rayleigh-Bénard convection models.

    This class only handles the initial set-up of the solver; additional
    functionality including the ability to run the model is provided
    by subclasses.
    """

    def __init__(
            self, equations, aspect, nx, nz, rayleigh, prandtl,
            *eqn_args, **eqn_kwargs):
        """
        Builds the solver.

        Args:
            equations: Equation class. Currently implemented options are
                - models.DNS
                - models.SmagorinskyLES
                - models.ResolvedTendencyParametrisation
            aspect: Domain aspect ratio.
            nx: Number of horizontal modes.
            nz: Number of vertical modes.
            rayleigh: Rayleigh number.
            prandtl: Prandtl number.
            eqn_args, eqn_kwargs: Additional arguments for the equation
                class.
        """

        # Parameters
        self.aspect = aspect
        self.nx = nx
        self.nz = nz
        self.rayleigh = rayleigh
        self.prandtl = prandtl

        # Fundamental objects
        coords = d3.CartesianCoordinates('x', 'z')
        self.dist = d3.Distributor(coords, dtype=np.float64)
        self.xbasis = d3.RealFourier(
            coords['x'], size=nx, bounds=(0, aspect), dealias=3/2)
        self.zbasis = d3.ChebyshevT(
            coords['z'], size=nz, bounds=(0, 1), dealias=3/2)
        self.local_grids = dict(zip(
           ('x', 'z'),  self.dist.local_grids(self.xbasis, self.zbasis)
        ))
        self.unit_vectors = dict(zip(
           ('x_hat', 'z_hat'),  coords.unit_vector_fields(self.dist)
        ))

        # Fields
        self.fields = {
            'u': self.dist.VectorField(
                coords, name='u', bases=(self.xbasis, self.zbasis)),
            'theta': self.dist.Field(
                name='theta', bases=(self.xbasis, self.zbasis)),
            'pi': self.dist.Field(
                name='pi', bases=(self.xbasis, self.zbasis)),
            'tau_pi': self.dist.Field(
                name='tau_pi'),
            'tau_theta1': self.dist.Field(
                name='tau_theta1', bases=self.xbasis),
            'tau_theta2': self.dist.Field(
                name='tau_theta2', bases=self.xbasis),
            'tau_u1': self.dist.VectorField(
                coords, name='tau_u1', bases=self.xbasis),
            'tau_u2': self.dist.VectorField(
                coords, name='tau_u2', bases=self.xbasis),
        }

        # Substitutions
        def lift(field):
            """Multiplies by the highest Chebyshev polynomial."""
            return d3.Lift(field, self.zbasis.derivative_basis(1), -1)

        self.substitutions = {
            'lift': lift,
            'grad_u': (
                d3.grad(self.fields['u'])
                - self.unit_vectors['z_hat']*lift(self.fields['tau_u1'])
            ),
            'grad_theta': (
                d3.grad(self.fields['theta'])
                - self.unit_vectors['z_hat']*lift(self.fields['tau_theta1'])
            ),
        }

        # Problem
        self.problem = d3.IVP(list(self.fields.values()))
        equations = equations(self, *eqn_args, **eqn_kwargs)
        self.problem.add_equation(equations.momentum_equation)
        self.problem.add_equation(equations.energy_equation)
        self.problem.add_equation(equations.continuity_equation)
        self.problem.add_equation(equations.gauge_condition)
        for condition in equations.boundary_conditions:
            self.problem.add_equation(condition)

        # Solver
        self.solver = self.problem.build_solver(d3.RK222)


class Model(ModelBase):
    """
    Model for normal simulations of Rayleigh-Bénard convection.

    Usage:
        1. Instantiate model (see documentation for models.ModelBase).
        2. Set initial conditions using one of the following methods:
            - Model.set_initial_conditions
            - Model.restart
            - Model.load_from_existing
        3. Set up data logging using the Model.log_data method.
        4. Optionally set up logging of snapshot pairs using the
            Model.log_data_plusdt method.
        5. Run the model using the Model.run method.
    """

    @wraps(ModelBase.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.restarted = False
        self.dir = None
        self.save_iter = None
        self.max_writes = None

    def set_initial_conditions(self):
        """Initialises the variables for a new run."""
        k_base = 4
        k_perturb = 41

        def exp_term(x):
            base_wave = np.cos(2*np.pi*k_base*x/self.aspect)
            perturb_wave = np.cos(2*np.pi*k_perturb*(x - 0.25)/self.aspect)
            mod_wave = np.cos(np.pi*x/2)**4
            return base_wave + 0.1*perturb_wave*mod_wave

        x = self.local_grids['x']
        z = self.local_grids['z']
        exp = 6 - np.where(z < 0.5, exp_term(x), exp_term(x - 1))
        self.fields['theta']['g'] = -0.5*np.sign(z - 0.5)*np.abs(2*z - 1)**exp

    def restart(self, file):
        """Loads a model state from a restart file."""
        self.solver.load_state(file)
        self.restarted = True

    def load_from_existing(self, file, time, filter_time, filter_dt):
        """
        Regrids and loads the state from an existing model run.

        Args:
            file: NetCDF data file or glob pattern.
            time: Time of state to be loaded.
            filter_time: Run time of diffusion model for smoothing.
            filter_dt: Time step of diffusion model.
        """

        data = xr.open_mfdataset(file)
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            data = data.drop_duplicates('t')
        data = data.sel(t=time, method='nearest').compute()
        actual_time = data.t.item()
        filter = coarse_graining.CoarseGrainer(
            self.aspect, data.x.size, data.z.size)

        u, w, theta = filter.run(
            data.u, data.w, data.theta,
            time=filter_time, dt=filter_dt,
            to_shape=(self.nx, self.nz),
        )
        self.fields['u'].load_from_global_grid_data(np.stack([u, w]))
        self.fields['theta'].load_from_global_grid_data(theta)

    def log_data(self, dir, save_sim_time, max_writes, timestep):
        """
        Tells the solver to save output to a file.

        Args:
            dir: Directory for data output.
            save_sim_time: Snapshot interval in model time units
                (will be rounded to nearest multiple of time step).
            max_writes: Maximum number of writes per output file
                (default 1000).
        """

        self.dir = dir
        self.save_iter = round(save_sim_time/timestep)
        self.max_writes = max_writes

        snapshots = self.solver.evaluator.add_file_handler(
            dir, iter=self.save_iter, max_writes=max_writes,
            mode=('append' if self.restarted else 'overwrite'),
        )
        snapshots.add_tasks(
            [self.fields['u'], self.fields['theta']], layout='g')

    def log_data_plusdt(self, dir):
        """
        Saves a second dataset with snapshots lagged by one time step.

        Args:
            dir: Directory for data output.
        """

        snapshots = self.solver.evaluator.add_file_handler(
            dir,
            custom_schedule=(
                lambda **kw: kw['iteration'] % self.save_iter == 1
            ),
            max_writes=self.max_writes,
            mode=('append' if self.restarted else 'overwrite'),
        )
        snapshots.add_tasks(
            [self.fields['u'], self.fields['theta']], layout='g')

    def run(self, sim_time, timestep):
        """
        Runs the simulation.

        Args:
            sim_time: Total simulation time.
            timestep: Time step.
        """

        self._log_checkpoints(sim_time, timestep)
        self.solver.stop_sim_time = sim_time

        flow = d3.GlobalFlowProperty(self.solver, cadence=100)
        flow.add_property(self.fields['u']@self.fields['u'], name='magsq_u')

        try:
            while self.solver.proceed:
                self.solver.step(timestep)
                if self.solver.iteration % 100 == 1:
                    u_rms = np.sqrt(flow.grid_average('magsq_u'))
                    logger.info(
                        f'Iteration {self.solver.iteration:d}\t\t'
                        f'Sim time {self.solver.sim_time:.3e}\t\t'
                        f'RMS velocity {u_rms:.3e}'
                    )
                    if np.isnan(u_rms):
                        logger.error('Model has crashed!')
                        break
        except Exception:
            logger.error('Exception raised, triggering end of main loop.')
            raise
        finally:
            self.solver.log_stats()

    def _log_checkpoints(self, sim_time, timestep):
        """Tells the solver to save a restart file at the end."""
        def restart_schedule(iteration, **_):
            last_save = (
                (sim_time//(self.save_iter*timestep))*self.save_iter
            )
            return iteration == last_save

        restarts = self.solver.evaluator.add_file_handler(
            os.path.join(self.dir, 'restart'), max_writes=1,
            custom_schedule=restart_schedule, mode='append',
        )
        restarts.add_tasks(self.solver.state, layout='g')


class SingleStepModel(ModelBase):
    """
    Model for timestepping existing data.

    Usage:
        1. Load the existing data using the
            SingleStepModel.load_initial_states method.
        2. Set up data logging using the
           SingleStepModel.log_final_states method.
        3. Run the model using the SingleStepModel.run method.
    """

    @wraps(ModelBase.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_states = None
        self.snapshots = None

    def load_initial_states(self, file):
        """
        Loads snapshots in preparation for timestepping.

        The data must be of the same resolution as this model.

        Args:
            file: NetCDF data file or glob pattern.
        """

        self.initial_states = xr.open_mfdataset(file)

    def log_final_states(self, dir, max_writes):
        """
        Tells the solver where to save the time-stepped output.

        Args:
            dir: Directory for data output.
            max_writes: Maximum number of writes per output file
                (default 1000).
        """

        # Omit iter argument to add_file_handler to disable auto evaluation
        self.snapshots = self.solver.evaluator.add_file_handler(
            dir, max_writes=max_writes)
        self.snapshots.add_tasks(
            [self.fields['u'], self.fields['theta']], layout='g')

    def run(self, timestep):
        """
        Performs the time-stepping.

        Args:
            timestep: Time step.
        """

        try:
            for i in range(self.initial_states.t.size):
                # Zero all fields just to be safe
                for field in self.fields:
                    self.fields[field]['g'] = 0

                # Load the next snapshot into the solver
                u = self.initial_states.u.isel(t=i).compute()
                w = self.initial_states.w.isel(t=i).compute()
                theta = self.initial_states.theta.isel(t=i).compute()
                self.fields['u'].load_from_global_grid_data(np.stack([u, w]))
                self.fields['theta'].load_from_global_grid_data(theta)

                # Set the sim time so it shows up correctly in the output
                self.solver.sim_time = self.initial_states.t[i].item()

                # Perform one timestep
                self.solver.step(timestep)

                # Save the model state after the timestep
                self.solver.evaluator.evaluate_handlers(
                    [self.snapshots], iteration=self.solver.iteration,
                    wall_time=self.solver.wall_time,
                    sim_time=self.solver.sim_time, timestep=self.solver.dt,
                )

                if i % 10 == 0:
                    logger.info(
                        f'Step {i+1:d} of {self.initial_states.t.size}')
        except Exception:
            logger.error('Exception raised, triggering end of main loop.')
            raise
        finally:
            self.solver.log_stats()


def zero_gauge(self):
    """Zero-mean pressure gauge condition."""
    lhs = d3.integ(self.model.fields['pi'])
    rhs = 0
    return lhs, rhs


def noslip_isothermal(self):
    """Classical no-slip, isothermal boundary conditions."""
    model = self.model
    conditions = (
        (model.fields['u'](z=0), 0),
        (model.fields['u'](z=1), 0),
        (model.fields['theta'](z=0), 1/2),
        (model.fields['theta'](z=1), -1/2),
    )
    return conditions


def divergence_free(self):
    """Continuity equation for incompressible fluids."""
    model = self.model
    lhs = d3.trace(model.substitutions['grad_u']) + model.fields['tau_pi']
    rhs = 0
    return lhs, rhs


class DNS:
    """Equation class for direct numerical simulations."""

    gauge_condition = property(zero_gauge)
    boundary_conditions = property(noslip_isothermal)
    continuity_equation = property(divergence_free)

    def __init__(self, model):
        """Instantiates the equation class."""
        self.model = model

    @property
    def momentum_equation(self):
        """Momentum equation."""
        model = self.model
        nu = np.sqrt(model.prandtl/model.rayleigh)
        lap_u = d3.div(model.substitutions['grad_u'])
        lhs = (
            d3.dt(model.fields['u'])
            - nu*lap_u
            + d3.grad(model.fields['pi'])
            - model.fields['theta']*model.unit_vectors['z_hat']
            + model.substitutions['lift'](model.fields['tau_u2'])
        )
        rhs = -model.fields['u']@model.substitutions['grad_u']
        return lhs, rhs

    @property
    def energy_equation(self):
        """Energy equation."""
        model = self.model
        kappa = 1/np.sqrt(model.rayleigh*model.prandtl)
        lap_theta = d3.div(model.substitutions['grad_theta'])
        lhs = (
            d3.dt(model.fields['theta'])
            - kappa*lap_theta
            + model.substitutions['lift'](model.fields['tau_theta2'])
        )
        rhs = -model.fields['u']@model.substitutions['grad_theta']
        return lhs, rhs


class SmagorinskyLES:
    """Equation class for LES using the Smagorinsky closure."""

    gauge_condition = property(zero_gauge)
    boundary_conditions = property(noslip_isothermal)
    continuity_equation = property(divergence_free)

    def __init__(self, model, delta_nu):
        """Instantiates the equation class."""
        self.model = model
        self.delta_nu = delta_nu

    @property
    def strain_tensor(self):
        """Strain tensor."""
        model = self.model
        return 0.5*(
            model.substitutions['grad_u']
            + d3.transpose(model.substitutions['grad_u'])
        )

    @property
    def damping(self):
        """van Driest damping function."""
        a_plus = 26.
        model = self.model
        z = model.local_grids['z']
        z_plus = np.minimum(z, 1 - z)/self.delta_nu
        field = model.dist.Field(name='damping', bases=model.zbasis)
        field['g'] = 1 - np.exp(-z_plus/a_plus)
        return field

    @property
    def eddy_viscosity(self):
        """Smagorinsky eddy viscosity."""
        model = self.model
        filter_width = 2
        delta_x = filter_width*model.xbasis.local_grid_spacing(axis=0)
        delta_z = filter_width*model.zbasis.local_grid_spacing(axis=1)
        delta = model.dist.Field(bases=(model.xbasis, model.zbasis))
        delta['g'] = np.sqrt(delta_x*delta_z)

        smagorinsky_coeff = 0.17
        return (
            (smagorinsky_coeff*delta*self.damping)**2
            * (2*d3.trace(self.strain_tensor@self.strain_tensor))**0.5
        )

    @property
    def momentum_equation(self):
        """Momentum equation."""
        model = self.model
        nu = np.sqrt(model.prandtl/model.rayleigh)
        lap_u = d3.div(model.substitutions['grad_u'])
        subgrid_stress = 2*self.eddy_viscosity*self.strain_tensor

        lhs = (
            d3.dt(model.fields['u'])
            - nu*lap_u
            + d3.grad(model.fields['pi'])
            - model.fields['theta']*model.unit_vectors['z_hat']
            + model.substitutions['lift'](model.fields['tau_u2'])
        )
        rhs = (
            -model.fields['u']@model.substitutions['grad_u']
            + d3.div(subgrid_stress)
        )
        return lhs, rhs

    @property
    def energy_equation(self):
        """Energy equation."""
        model = self.model
        kappa = 1/np.sqrt(model.rayleigh*model.prandtl)
        lap_theta = d3.div(model.substitutions['grad_theta'])
        subgrid_flux = self.eddy_viscosity*model.substitutions['grad_theta']

        lhs = (
            d3.dt(model.fields['theta'])
            - kappa*lap_theta
            + model.substitutions['lift'](model.fields['tau_theta2'])
        )
        rhs = (
            -model.fields['u']@model.substitutions['grad_theta']
            + d3.div(subgrid_flux)
        )
        return lhs, rhs


class ResolvedTendencyParametrisation(SmagorinskyLES):
    """Parametrisation scheme based on resolved tendencies."""

    gauge_condition = property(zero_gauge)
    boundary_conditions = property(noslip_isothermal)
    continuity_equation = property(divergence_free)

    def __init__(self, model, delta_nu, coef_file):
        """
        Initialises the parametrisation scheme.

        Args:
            model: models.Model instance.
            delta_nu: Viscous length scale for Smagorinsky scheme.
            coef_file: Path to csv file with parametrisation coefficients.
        """

        super().__init__(model, delta_nu)
        self.params = pd.read_csv(coef_file)

    @property
    def momentum_equation(self):
        """Momentum equation."""
        model = self.model
        nu = np.sqrt(model.prandtl/model.rayleigh)
        lap_u = d3.div(model.substitutions['grad_u'])
        subgrid_stress = 2*self.eddy_viscosity*self.strain_tensor

        coeffs_u = self.params['u'].to_numpy()
        coeffs_w = self.params['w'].to_numpy()
        # Add 1 to the zeroth-order-in-z coefficient because we are adding
        # the subgrid tendency to the resolved tendency
        coeffs_u[1] += 1
        coeffs_w[1] += 1

        z = model.dist.Field(name='z', bases=model.zbasis)
        z['g'] = model.local_grids['z']

        u_func = sum([
            coeffs_u[n+1]*(z - 1/2)**(2*n) for n in range(coeffs_u.size - 1)
        ])
        u_func = u_func.evaluate()
        w_func = sum([
            coeffs_w[n+1]*(z - 1/2)**(2*n) for n in range(coeffs_w.size - 1)
        ])
        w_func = w_func.evaluate()

        # Form a 2x2 matrix with u_func and w_func along the diagonal.
        # This can then be multiplied with the resolved tendency vector
        # to get the corrected tendency vector.
        uw_func = (
            u_func*model.unit_vectors['x_hat']*model.unit_vectors['x_hat']
            + w_func*model.unit_vectors['z_hat']*model.unit_vectors['z_hat']
        )

        lhs = (
            d3.dt(model.fields['u'])
            + uw_func @ (
                - nu*lap_u
                + d3.grad(model.fields['pi'])
                - model.fields['theta']*model.unit_vectors['z_hat']
                + model.substitutions['lift'](model.fields['tau_u2'])
            )
        )
        rhs = (
            # coeffs_u[0]*self.unit_vectors['x_hat']  # constant u term
            # + coeffs_w[0]*self.unit_vectors['z_hat']  # constant w term
            uw_func @ (
                -model.fields['u']@model.substitutions['grad_u']
                + d3.div(subgrid_stress)
            )
        )
        return lhs, rhs

    @property
    def energy_equation(self):
        """Energy equation."""
        model = self.model
        kappa = 1/np.sqrt(model.rayleigh*model.prandtl)
        lap_theta = d3.div(model.substitutions['grad_theta'])
        subgrid_flux = self.eddy_viscosity*model.substitutions['grad_theta']

        coeffs = self.params['theta'].to_numpy()
        # Add 1 to the zeroth-order-in-z coefficient because we are adding
        # the subgrid tendency to the resolved tendency
        coeffs[1] += 1

        z = model.dist.Field(name='z', bases=model.zbasis)
        z['g'] = model.local_grids['z']

        func = sum([
            coeffs[n+1]*(z - 1/2)**(2*n) for n in range(coeffs.size - 1)
        ])
        func = func.evaluate()

        lhs = (
            d3.dt(model.fields['theta'])
            + func * (
                - kappa*lap_theta
                + model.substitutions['lift'](model.fields['tau_theta2'])
            )
        )
        rhs = (
            # coeffs[0]
            func * (
                -model.fields['u']@model.substitutions['grad_theta']
                + d3.div(subgrid_flux)
            )
        )
        return lhs, rhs

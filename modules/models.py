"""Script to set up RBC simulations in Dedalus."""

import logging
import os

import dedalus.public as d3
import numpy as np
import xarray as xr
import dask

from modules.regridding import Regridder

logger = logging.getLogger(__name__)
# pylint: disable=no-member, too-many-arguments, too-many-instance-attributes


class BaseModel:
    """Base model for Rayleigh-Benard convection."""
    timestepper = d3.RK222

    def __init__(self, aspect, Nx, Nz, Rayleigh, Prandtl, hyper):
        """
        Builds the solver. This is the same for all simulations.

        Args:
            aspect: Domain aspect ratio.
            Nx: Number of horizontal modes.
            Nz: Number of vertical modes.
            Rayleigh: Rayleigh number.
            Prandtl: Prandtl number.
        """

        # Parameters
        self.Rayleigh = Rayleigh
        self.Prandtl = Prandtl
        self.hyper = hyper
        self.aspect = aspect
        self.restarted = False
        self.data_dir = ''
        self.save_interval = 1
        self.max_writes = 1000
        logger.info('Parameters:')
        logger.info(f'\tRa = {Rayleigh:.2e}')
        logger.info(f'\tPr = {Prandtl:.3f}')
        logger.info(f'\thyper = {hyper:.2e}')
        logger.info(f'\taspect = {aspect:.1f}')
        logger.info(f'\tNx = {Nx:d}')
        logger.info(f'\tNz = {Nz:d}')

        # Fundamental objects
        self.coords = d3.CartesianCoordinates('x', 'z')
        self.dist = d3.Distributor(self.coords, dtype=np.float64)
        self.xbasis = d3.RealFourier(
            self.coords['x'], size=Nx, bounds=(0, aspect), dealias=3/2)
        self.zbasis = d3.ChebyshevT(
            self.coords['z'], size=Nz, bounds=(0, 1), dealias=3/2)
        self.local_grids = dict(zip(
           ('x', 'z'),  self.dist.local_grids(self.xbasis, self.zbasis)
        ))
        self.unit_vectors = dict(zip(
           ('x_hat', 'z_hat'),  self.coords.unit_vector_fields(self.dist)
        ))

        # Fields
        self.fields = {
            'u': self.dist.VectorField(
                self.coords, name='u', bases=(self.xbasis, self.zbasis)),
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
                self.coords, name='tau_u1', bases=self.xbasis),
            'tau_u2': self.dist.VectorField(
                self.coords, name='tau_u2', bases=self.xbasis),
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
        self.problem.add_equation(self.momentum_equation)
        self.problem.add_equation(self.energy_equation)
        self.problem.add_equation(self.continuity_equation)
        self.problem.add_equation(self.gauge_condition)
        for condition in self.boundary_conditions:
            self.problem.add_equation(condition)

        # Solver
        self.solver = self.problem.build_solver(self.timestepper)

    @property
    def taper(self):
        """Field that tapers to zero at z = 0 and z = 1."""
        def taper_func(x):
            return 56*x**10 - 140*x**12 + 120*x**14 - 35*x**16

        field = self.dist.Field(bases=self.zbasis)
        field['g'] = 1 - taper_func(2*self.local_grids['z'] - 1)
        return field

    @property
    def momentum_equation(self):
        """Momentum equation."""
        nu = np.sqrt(self.Prandtl/self.Rayleigh)
        lap_u = d3.div(self.substitutions['grad_u'])
        lhs = (
            d3.dt(self.fields['u'])
            - nu*lap_u
            + d3.grad(self.fields['pi'])
            - self.fields['theta']*self.unit_vectors['z_hat']
            + self.substitutions['lift'](self.fields['tau_u2'])
        )
        rhs = (
            -self.fields['u']@self.substitutions['grad_u']
            + nu*self.hyper*self.taper*(lap_u@lap_u)**(1/2)*lap_u
        )
        return lhs, rhs

    @property
    def energy_equation(self):
        """Energy equation."""
        kappa = 1/np.sqrt(self.Rayleigh*self.Prandtl)
        lap_theta = d3.div(self.substitutions['grad_theta'])
        lhs = (
            d3.dt(self.fields['theta'])
            - kappa*lap_theta
            + self.substitutions['lift'](self.fields['tau_theta2'])
        )
        rhs = (
            -self.fields['u']@self.substitutions['grad_theta']
            + kappa*self.hyper*self.taper*abs(lap_theta)*lap_theta
        )
        return lhs, rhs

    @property
    def continuity_equation(self):
        """Continuity equation."""
        lhs = d3.trace(self.substitutions['grad_u']) + self.fields['tau_pi']
        rhs = 0
        return lhs, rhs

    @property
    def gauge_condition(self):
        """Gauge condition."""
        lhs = d3.integ(self.fields['pi'])
        rhs = 0
        return lhs, rhs

    @property
    def boundary_conditions(self):
        """Boundary conditions."""
        conditions = (
            (self.fields['u'](z=0), 0),
            (self.fields['u'](z=1), 0),
            (self.fields['theta'](z=0), 1/2),
            (self.fields['theta'](z=1), -1/2),
        )
        return conditions


class DNSModel(BaseModel):
    """Model for direct numerical simulations."""

    def set_initial_conditions(self):
        """Initialises the variables for a new run."""
        logger.info('Initial condition: wavy_theta')

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
        logger.info(f'Initial condition: restart from {file}')
        self.solver.load_state(file)
        self.restarted = True

    def load_from_existing(self, file, time):
        """
        Regrids and loads the state from an existing model run.

        Args:
            file: NetCDF data file or glob pattern.
            time: Time of state to be loaded.
        """

        data = xr.open_mfdataset(file)
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            data = data.drop_duplicates('t')
        data = data.sel(t=time, method='nearest').compute()
        actual_time = data.t.item()
        logger.info(
            f'Initial condition: state at t = {actual_time:.3f} from {file}')
        target = {
            'x': self.xbasis.global_grid().flatten(),
            'z': self.zbasis.global_grid().flatten(),
        }
        regridder = Regridder(
            data.theta, target, limits={'z': (0, 1)},
            periods={'x': self.aspect},
        )
        u = regridder(data.u).transpose('x', 'z').data
        w = regridder(data.w).transpose('x', 'z').data
        theta = regridder(data.theta).transpose('x', 'z').data
        self.fields['u'].load_from_global_grid_data(np.stack([u, w]))
        self.fields['theta'].load_from_global_grid_data(theta)

    def log_data(self, data_dir, save_interval, max_writes=1000):
        """
        Tells the solver to save output to a file.

        Args:
            data_dir: Directory for data output.
            save_interval: Number of time steps between saves.
            max_writes: Maximum number of writes per output file
                (default 1000).
        """

        self.data_dir = data_dir
        self.save_interval = save_interval
        self.max_writes = max_writes

        snapshots = self.solver.evaluator.add_file_handler(
            data_dir, iter=save_interval, max_writes=max_writes,
            mode=('append' if self.restarted else 'overwrite'),
        )
        snapshots.add_tasks(
            [self.fields['u'], self.fields['theta']], layout='g')
        logger.info('Output settings:')
        logger.info(f'\tDirectory: {data_dir:s}')
        logger.info(f'\tLogging interval: {save_interval:d}*dt')
        logger.info(f'\tMax writes per file: {max_writes:d}')

    def log_data_plusdt(self, data_dir):
        """
        Saves a second dataset with snapshots lagged by one time step.

        Args:
            data_dir: Directory for data output.
        """

        snapshots = self.solver.evaluator.add_file_handler(
            data_dir,
            custom_schedule=(
                lambda **kw: kw['iteration'] % self.save_interval == 1
            ),
            max_writes=self.max_writes,
            mode=('append' if self.restarted else 'overwrite'),
        )
        snapshots.add_tasks(
            [self.fields['u'], self.fields['theta']], layout='g')
        logger.info(f'\tSaving t+dt snapshots at: {data_dir:s}')

    def run(self, sim_time, timestep):
        """
        Runs the simulation.

        Args:
            sim_time: Total simulation time.
            timestep: Time step.
        """

        self._log_checkpoints(sim_time, timestep)
        self.solver.stop_sim_time = sim_time
        logger.info(f'Simulation length: t_max = {sim_time:.3g}')

        flow = d3.GlobalFlowProperty(self.solver, cadence=100)
        flow.add_property(self.fields['u']@self.fields['u'], name='magsq_u')

        try:
            logger.info(f'Starting main loop with dt = {timestep:.3g}')
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
                (sim_time//(self.save_interval*timestep))*self.save_interval
            )
            return iteration == last_save

        restarts = self.solver.evaluator.add_file_handler(
            os.path.join(self.data_dir, 'restart'), max_writes=1,
            custom_schedule=restart_schedule, mode='append',
        )
        restarts.add_tasks(self.solver.state, layout='g')


class SingleStepModel(BaseModel):
    """Model for coarse-graining and timestepping existing data."""

    def input_highres(self, file):
        """
        Prepares high-resolution input data for timestepping.

        Args:
            file: NetCDF data file or glob pattern.
        """
        logger.info(f'Using high-res data from {file}')
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # pylint: disable=W0201
            self.highres = xr.open_mfdataset(file)
            target = {
                'x': self.xbasis.global_grid().flatten(),
                'z': self.zbasis.global_grid().flatten(),
            }
            # pylint: disable=W0201
            self.regridder = Regridder(
                self.highres.theta, target, limits={'z': (0, 1)},
                periods={'x': self.aspect},
            )

    def log_data_t(self, data_dir, max_writes=1000):
        """
        Tells the solver to save the coarse-grained data before timestepping.

        Args:
            data_dir: Directory for data output.
            max_writes: Maximum number of writes per output file
                (default 1000).
        """

        self.max_writes = max_writes
        snapshots = self.solver.evaluator.add_file_handler(
            data_dir, iter=1, max_writes=max_writes)
        snapshots.add_tasks(
            [self.fields['u'], self.fields['theta']], layout='g')
        logger.info(f'\tSaving time t snapshots at: {data_dir:s}')

    def log_data_tplusdt(self, data_dir):
        """
        Tells the solver to save the coarse-grained data after timestepping.

        Args:
            data_dir: Directory for data output.
        """

        # Omit iter argument to add_file_handler to disable auto evaluation
        # pylint: disable=W0201
        self.snapshots_tplusdt = self.solver.evaluator.add_file_handler(
            data_dir, max_writes=self.max_writes)
        self.snapshots_tplusdt.add_tasks(
            [self.fields['u'], self.fields['theta']], layout='g')
        logger.info(f'\tSaving time t+dt snapshots at: {data_dir:s}')

    def run(self, timestep):
        """
        Runs the simulation.

        Args:
            sim_time: Total simulation time.
            timestep: Time step.
        """

        try:
            logger.info(f'Starting main loop with dt = {timestep:.3g}')
            for i in range(self.highres.t.size):
                # Load an initial state from the coarse-grained dataset
                u = self.regridder(
                    self.highres.u.isel(t=i)).transpose('x', 'z').data
                w = self.regridder(
                    self.highres.w.isel(t=i)).transpose('x', 'z').data
                theta = self.regridder(
                    self.highres.theta.isel(t=i)).transpose('x', 'z').data
                self.fields['u'].load_from_global_grid_data(np.stack([u, w]))
                self.fields['theta'].load_from_global_grid_data(theta)

                # Set the sim time so it shows up correctly in the output
                self.solver.sim_time = self.highres.t[i].item()

                # Perform one timestep (this automatically saves the initial
                # state)
                self.solver.step(timestep)

                # Save the model state after the timestep
                self.solver.evaluator.evaluate_handlers(
                    [self.snapshots_tplusdt], iteration=self.solver.iteration,
                    wall_time=self.solver.wall_time,
                    sim_time=self.solver.sim_time, timestep=self.solver.dt,
                )

                if i % 100 == 0:
                    logger.info(f'Step {i+1:d} of {self.highres.t.size}')
        except Exception:
            logger.error('Exception raised, triggering end of main loop.')
            raise
        finally:
            self.solver.log_stats()

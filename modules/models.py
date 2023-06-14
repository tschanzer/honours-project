"""Script to set up RBC simulations in Dedalus."""

import logging
import os

import dedalus.public as d3
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel:
    """Base model for Rayleigh-Benard convection."""
    timestepper = d3.RK222

    def __init__(self, aspect, Nx, Nz, Rayleigh, Prandtl):
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
        self.aspect = aspect
        logger.info('BaseModel parameters:')
        logger.info(f'\tRa = {Rayleigh:.2e}')
        logger.info(f'\tPr = {Prandtl:.3f}')
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
    def momentum_equation(self):
        """Momentum equation."""
        lhs = (
            self.Rayleigh/self.Prandtl*d3.dt(self.fields['u'])
            - d3.div(self.substitutions['grad_u'])
            + d3.grad(self.fields['pi'])
            - self.fields['theta']*self.unit_vectors['z_hat']
            + self.substitutions['lift'](self.fields['tau_u2'])
        )
        rhs = (
            -self.Rayleigh/self.Prandtl
            * self.fields['u']@self.substitutions['grad_u']
        )
        return lhs, rhs

    @property
    def energy_equation(self):
        """Energy equation."""
        lhs = (
            self.Rayleigh*d3.dt(self.fields['theta'])
            - d3.div(self.substitutions['grad_theta'])
            + self.substitutions['lift'](self.fields['tau_theta2'])
        )
        rhs = (
            -self.Rayleigh*self.fields['u']@self.substitutions['grad_theta']
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
        self.restart = False

    def restart(self, file):
        """Loads a model state from a restart file."""
        logger.info(f'Initial condition: restart from {file}')
        self.solver.load_state(file)
        self.restart = True

    def log_data(self, data_dir, save_interval):
        """
        Tells the solver to save output to a file.

        Args:
            data_dir: Directory for data output.
            save_interval: Number of time steps between saves.
        """

        self.data_dir = data_dir
        self.save_interval = save_interval
        if self.restart:
            def save_schedule(iteration, **_):
                save = (
                    (iteration % save_interval == 0)
                    and (iteration != self.solver.initial_iteration)
                )
                return save
        else:
            def save_schedule(iteration, **_):
                return iteration % save_interval == 0

        snapshots = self.solver.evaluator.add_file_handler(
            data_dir, custom_schedule=save_schedule, max_writes=1000,
            mode=('append' if self.restart else 'overwrite'),
        )
        snapshots.add_tasks(
            [self.fields['u'], self.fields['theta']], layout='g')
        logger.info('Output settings:')
        logger.info(f'\tDirectory: {data_dir:s}')
        logger.info(f'\tLogging interval: {save_interval:d}*dt')

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


class LinearHyperdiffusionModel(BaseModel):
    """Model with linear hyperdiffusion."""
    def __init__(self, *args, hyper_coef):
        """
        Builds the solver.

        Args:
            aspect: Domain aspect ratio.
            Nx: Number of horizontal modes.
            Nz: Number of vertical modes.
            Rayleigh: Rayleigh number.
            Prandtl: Prandtl number.
            hyper_coef: Dimensionless hyperviscosity coefficient.
        """

        logger.info('LinearHyperdiffusionModel parameters:')
        logger.info(f'\thyper_coef = {hyper_coef:.2e}')
        self.hyper_coef = hyper_coef
        super().__init__(*args)

    @property
    def taper(self):
        """Field that tapers to zero at z = 0 and z = 1."""
        field = self.dist.Field(bases=self.zbasis)
        field['g'] = 1 - (2*self.local_grids['z'] - 1)**10
        return field

    @property
    def momentum_equation(self):
        """Momentum equation."""
        lhs = (
            self.Rayleigh/self.Prandtl*d3.dt(self.fields['u'])
            - d3.div(self.substitutions['grad_u'])
            + self.hyper_coef*self.taper*d3.lap(d3.lap(self.fields['u']))
            + d3.grad(self.fields['pi'])
            - self.fields['theta']*self.unit_vectors['z_hat']
            + self.substitutions['lift'](self.fields['tau_u2'])
        )
        rhs = (
            -self.Rayleigh/self.Prandtl
            * self.fields['u']@self.substitutions['grad_u']
        )
        return lhs, rhs

    @property
    def energy_equation(self):
        """Energy equation."""
        lhs = (
            self.Rayleigh*d3.dt(self.fields['theta'])
            - d3.div(self.substitutions['grad_theta'])
            + self.hyper_coef*self.taper*d3.lap(d3.lap(self.fields['theta']))
            + self.substitutions['lift'](self.fields['tau_theta2'])
        )
        rhs = (
            -self.Rayleigh*self.fields['u']@self.substitutions['grad_theta']
        )
        return lhs, rhs

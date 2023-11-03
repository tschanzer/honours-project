"""Module for coarse-graining high-resolution data."""
# pylint: disable=no-member

import logging

import dedalus.public as d3
import numpy as np
import scipy as sp
import xarray as xr

logger = logging.getLogger(__name__)


class CoarseGrainer:
    """Diffusion model for coarse-graining high-resolution data."""
    def __init__(self, file, aspect):
        """
        Builds the solver.

        Args:
            file: Path or glob pattern for high-res input data.
            aspect: Domain aspect ratio.
        """

        self.input = xr.open_mfdataset(file)
        self.nx = self.input.x.size
        self.nz = self.input.z.size
        self.snapshots = None
        self.out_shape = None

        # Fundamental objects
        coords = d3.CartesianCoordinates('x', 'z')
        dist = d3.Distributor(coords, dtype=np.float64)
        xbasis = d3.RealFourier(coords['x'], size=self.nx, bounds=(0, aspect))
        zbasis = d3.ChebyshevT(coords['z'], size=self.nz, bounds=(0, 1))
        _, z_hat = coords.unit_vector_fields(dist)  # pylint: disable=W0632

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

    def configure_output(self, resolution, out_dir, max_writes):
        """
        Tells the solver where to save the output.

        Args:
            resolution: Output resolution.
            dir: Directory for data output.
            max_writes: Maximum number of writes per output file
                (default 1000).
        """

        self.out_shape = resolution
        # Omit iter argument to add_file_handler to disable auto evaluation
        self.snapshots = self.solver.evaluator.add_file_handler(
            out_dir, max_writes=max_writes,
        )
        new_scales = (resolution[0]/self.nx, resolution[1]/self.nz)
        self.snapshots.add_tasks(
            [self.fields['u'], self.fields['theta']],
            layout='g', scales=new_scales,
        )

    def run(self, time, dt):
        """
        Smooths the input data by running the solver.

        Args:
            time: Run time (i.e. amount of smoothing)
            dt: Time step.
        """

        for i in range(self.input.t.size):
            if i % 10 == 0:
                logger.info(f'Step {i+1:d} of {self.input.t.size}')

            # Zero all fields just to be safe
            for field in self.fields.values():
                field['g'] = 0

            # Load initial condition
            u = self.input.u.isel(t=i).compute()
            w = self.input.w.isel(t=i).compute()
            theta = self.input.theta.isel(t=i).compute()
            self.fields['u'].load_from_global_grid_data(np.stack([u, w]))
            self.fields['theta'].load_from_global_grid_data(theta)

            # Run until stop time is reached
            run_time = 0.
            while run_time < time:
                self.solver.step(dt)
                run_time += dt

            # Save the model state after the timestep, recording
            # iteration number, sim time and time step from *original*
            # data
            self.solver.evaluator.evaluate_handlers(
                [self.snapshots],
                iteration=self.input.iteration[i],
                wall_time=self.solver.wall_time,
                sim_time=self.input.sim_time[i],
                timestep=self.input.timestep[i],
            )


class ConservativeRegridder:
    """First-order conservative regridder for N-D rectilinear grids."""

    def __init__(self, source, target, limits=None, periods=None):
        """
        Creates the regridding object.

        Args:
            source: Fine-resolution xarray.Dataset or xarray.DataArray.
            target: Coarse-resolution xarray.Dataset or xarray.DataArray.
            limits: Dictionary mapping the name of each non-periodic
                regridding dimension to a 2-tuple of limits.
            periods: Dictionary mapping the name of each periodic regridding
                dimension to its period.

        Note:
            Choose the order of `dims` wisely as it can significantly
            affect performance.
        """

        source = self._check_input(source)
        target = self._check_input(target)

        self.dims = [*limits.keys(), *periods.keys()]
        self.source_shape = {dim: source[dim].size for dim in self.dims}
        self.target_shape = {dim: target[dim].size for dim in self.dims}
        self.target_coords = target.coords
        self.weights = {}
        if limits is None:
            limits = {}

        self.coarsen_dims = []
        self.interp_dims = []
        for dim in limits.keys():
            if source[dim].size > target[dim].size:
                self.coarsen_dims.append(dim)
                source_bounds = bounds_linear(source[dim].data, limits[dim])
                target_bounds = bounds_linear(target[dim].data, limits[dim])
                weight = compute_weights_linear(source_bounds, target_bounds)
                self.weights[dim] = sp.sparse.csr_array(weight)
            elif source[dim].size < target[dim].size:
                self.interp_dims.append(dim)

        for dim in periods.keys():
            if source[dim].size > target[dim].size:
                self.coarsen_dims.append(dim)
                source_bounds = bounds_periodic(source[dim].data, periods[dim])
                target_bounds = bounds_periodic(target[dim].data, periods[dim])
                weight = compute_weights_periodic(
                    source_bounds, target_bounds, periods[dim])
                self.weights[dim] = sp.sparse.csr_array(weight)
            elif source[dim].size < target[dim].size:
                self.interp_dims.append(dim)

    def __call__(self, grid):
        """
        Regrids data.

        Args:
            grid: xarray.Dataset or xarray.DataArray.

        Returns:
            Regridded version of the input.
        """

        # Regrid the data one dimension at a time
        for dim in self.coarsen_dims:
            # drop any non-index coordinates that depend on `dim`
            for coord in grid.coords:
                if coord != dim and dim in grid.coords[coord].dims:
                    grid = grid.reset_coords(coord, drop=True)

            # Regrid along `dim`, broadcasting over other dimensions
            grid = xr.apply_ufunc(
                apply_weights, grid,
                kwargs={'weights': self.weights[dim]},
                input_core_dims=[[dim]],
                output_core_dims=[[dim]],
                exclude_dims={dim},
                dask='allowed',
            )
            # insert the coarse coordinates
            grid = grid.assign_coords({dim: self.target_coords[dim].data})

        for dim in self.interp_dims:
            grid = grid.interp(
                {dim: self.target_coords[dim]},
                kwargs={'fill_value': 'extrapolate'},
            )

        return grid

    def __repr__(self):
        """Returns information about the regridder."""

        info = (
            'Conservative Regridder\n'
            f'  Input grid shape: {self.source_shape}\n'
            f'  Output grid shape: {self.target_shape}'
        )
        return info

    def _check_input(self, input_):
        if isinstance(input_, (xr.DataArray, xr.Dataset)):
            return input_
        if isinstance(input_, dict):
            return xr.DataArray(coords=input_, dims=list(input_.keys()))

        raise ValueError(f'Invalid input type: {type(input_)}')


def compute_weights_linear(source, target):
    """
    Regridding weight matrix for 1d grids with non-periodic boundaries.

    weight_ij = (
        (overlapping length between target interval i and source interval j)
        / (length of target interval i)
    )

    Args:
        source: Array of boundaries between points on the source grid.
        target: Array of boundaries between points on the target grid.

    Returns:
        Array of regridding weights.
    """

    # promote grids to 2d to allow broadcasting
    source = np.atleast_2d(source)  # source has index j
    target = np.atleast_2d(target).T  # target has index i

    # define shifted arrays for readability
    source_j = source[:, :-1]
    target_i = target[:-1, :]
    source_jplus1 = source[:, 1:]
    target_iplus1 = target[1:, :]

    # compute overlap length between every possible pair of intervals
    overlap = (
        np.minimum(source_jplus1, target_iplus1)
        - np.maximum(source_j, target_i)
    )
    overlap = np.maximum(overlap, 0)  # ensure it is non-negative

    # find the length of each target interval
    target_lengths = target_iplus1 - target_i

    # return the overlap fraction
    return overlap/target_lengths


def compute_weights_periodic(source, target, period):
    """
    Regridding weight matrix for 1d grids with periodic boundaries.

    weight_ij = (
        (overlapping length between target interval i and source interval j)
        / (length of target interval i)
    )

    Args:
        source: Array of boundaries between points on the source grid.
        target: Array of boundaries between points on the target grid.
        period: Period of the grid.

    Returns:
        Array of regridding weights.
    """

    if source[-1] > target[-1]:
        # If the rightmost source cell extends past the right edge of the
        # rightmost target cell, make a copy of the leftmost target cell
        # and put it at the right end of the target trid
        target = np.concatenate([target, [target[1] + period]])
        # Compute the weight matrix as usual
        weights_temp = compute_weights_linear(source, target)
        # Then remove the last row of the matrix (which corresponds to the
        # imaginary target cell) and add it to the first row
        weights = weights_temp[:-1, :]
        weights[0, :] += weights_temp[-1, :]
    elif source[0] < target[0]:
        # If the leftmost source cell extends past the left edge of the
        # leftmost target cell, make a copy of the rightmost target cell
        # and put it at the left end of the target trid
        target = np.concatenate([[target[-2] - period], target])
        # Compute the weight matrix as usual
        weights_temp = compute_weights_linear(source, target)
        # Then remove the first row of the matrix (which corresponds to the
        # imaginary target cell) and add it to the last row
        weights = weights_temp[1:, :]
        weights[-1, :] += weights_temp[0, :]
    else:
        # If the ends of the source and target grids match exactly, the
        # usual method works
        weights = compute_weights_linear(source, target)

    return weights


def apply_weights(data, weights):
    """
    Regrids an array along one dimension using a weight matrix.

    Args:
        data: numpy.ndarray, with the regridding dimension as the last
            dimension (i.e. `data.shape[-1] == weights.shape[1]`).
        weights: Regridding weight matrix (see `compute_weights_1d`).

    Returns:
        The regridded array, which has the same shape except in the last
        dimension, which has length `weights.shape[0]`.
    """

    # Collapse the non-regridding dimensions into one and transpose to
    # get a 2D array whose first dimension is the regridding dimension.
    # This way, the regridding can broadcast over an arbitrary number of
    # non-regridding dimensions.
    initial_shape = data.shape
    data = data.reshape((-1, initial_shape[-1])).T

    # Apply the weights (i.e. weights_ij*data_jk), then undo the
    # reshaping and transposition to get back to the input form.
    regridded_data = weights @ data
    regridded_data = regridded_data.T.reshape((*initial_shape[:-1], -1))
    return regridded_data


def bounds_linear(grid, limits):
    """
    Estimates cell boundaries for non-periodic 1D grids.

    Args:
        grid: 1D array of grid points.
        limits: Optional list specifying lower and upper boundaries.

    Returns:
        Array of estimated boundaries, with length len(grid) + 1.
        Boundaries are assumed to be the midpoints between grid points.
    """

    midpoints = (grid[:-1] + grid[1:])/2
    return np.concatenate([[limits[0]], midpoints, [limits[1]]])


def bounds_periodic(grid, period):
    """
    Estimates cell boundaries for periodic 1D grids.

    Args:
        grid: 1D array of grid points.
        period: Period of the grid.

    Returns:
        Array of estimated boundaries, with length len(grid) + 1.
        Boundaries are assumed to be the midpoints between grid points.
    """

    grid = np.concatenate([[grid[-1] - period], grid, [grid[0] + period]])
    return (grid[:-1] + grid[1:])/2

"""Utilities for conservative rectilinear regridding."""

import numpy as np
import scipy as sp
import xarray as xr


class Regridder:
    """Regridding object for N-D rectilinear grids."""

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

"""Utilities for conservative rectilinear regridding."""

import numpy as np
import scipy as sp
import xarray as xr


class Regridder:
    """Regridding object for N-D rectilinear grids."""

    def __init__(self, source, target, dims, limits=None):
        """
        Creates the regridding object.

        Args:
            source: Fine-resolution xarray.Dataset or xarray.DataArray.
            target: Coarse-resolution xarray.Dataset or xarray.DataArray.
            dims: List of names of dimensions to be regridded.
            limits: Dict of grid limits (see `bounds`).

        Note:
            Choose the order of `dims` wisely as it can significantly
            affect performance.
        """

        source = self._check_input(source)
        target = self._check_input(target)

        self.dims = dims
        source_shape = tuple(source[dim].size for dim in dims)
        target_shape = tuple(target[dim].size for dim in dims)
        self.source_shape = dict(zip(self.dims, source_shape))
        self.target_shape = dict(zip(self.dims, target_shape))
        self.target_coords = target.coords
        self.weights = {}
        if limits is None:
            limits = {}

        self.coarsen_dims = []
        self.interp_dims = []
        for dim in dims:
            if self.source_shape[dim] > self.target_shape[dim]:
                self.coarsen_dims.append(dim)
                # Compute the weight matrix along each dimension
                dim_limits = limits.get(dim)
                source_bounds = bounds(source[dim].data, limits=dim_limits)
                target_bounds = bounds(target[dim].data, limits=dim_limits)
                weight = compute_weights_1d(source_bounds, target_bounds)
                self.weights[dim] = weight
            elif self.source_shape[dim] < self.target_shape[dim]:
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


def compute_weights_1d(source, target):
    """
    Regridding weight matrix for 1d grids.

    weight_ij = (
        (overlapping length between target interval i and source interval j)
        / (length of target interval i)
    )

    Args:
        source: Array of boundaries between points on the source grid.
        target: Array of boundaries between points on the target grid.

    Returns:
        scipy.sparse.csr_array of regridding weights.
    """

    # promote grids to 2d to allow broadcasting
    source = np.atleast_2d(source)  # source has index j
    target = np.atleast_2d(target).T  # target has index i

    # define shifted arrays for code readability
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
    target_lengths = (
        np.minimum(target_iplus1, source[0, -1])
        - np.maximum(target_i, source[0, 0])
    )
    # if the target interval does not overlap with the source grid,
    # set its length to NaN
    target_lengths[target_lengths <= 0] = np.nan

    # return the overlap fraction as a sparse matrix
    return sp.sparse.csr_array(overlap/target_lengths)


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


def bounds(grid, limits=None):
    """
    Estimates boundaries between 1D grid points.

    Args:
        grid: 1D array of grid points.
        limits: Optional list specifying lower and upper boundaries.

    Returns:
        Array of estimated boundaries, with length len(grid) + 1.
        Boundaries are assumed to be the midpoints between grid points.
    """

    if limits is None:
        # choose first and last boundaries so the first and last grid
        # points are at the centres of their respective intervals
        limits = [2*grid[0] - grid[1], 2*grid[-1] - grid[-2]]

    midpoints = (grid[:-1] + grid[1:])/2
    return np.concatenate([[limits[0]], midpoints, [limits[1]]])

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
        """

        self.dims = dims
        self.source_shape = tuple(source[dim].size for dim in dims)
        self.target_shape = tuple(target[dim].size for dim in dims)
        self.target_coords = target.coords
        self.weights = {}
        if limits is None:
            limits = {}

        for dim in dims:
            # Compute the weight matrix along each dimension
            dim_limits = limits.get(dim)
            source_bounds = bounds(source[dim].data, limits=dim_limits)
            target_bounds = bounds(target[dim].data, limits=dim_limits)
            weight = compute_weights_1d(source_bounds, target_bounds)
            self.weights[dim] = weight

    def __call__(self, grid):
        """
        Regrids data.

        Args:
            grid: Fine-resolution xarray.Dataset or xarray.DataArray.

        Returns:
            Regridded version of the input.
        """

        # Regrid the data one dimension at a time
        for dim in self.dims:
            # drop any non-index coordinates that depend on `dim`
            for coord in grid.coords:
                if coord != dim and dim in grid.coords[coord].dims:
                    grid = grid.reset_coords(coord, drop=True)

            # Regrid along `dim`, broadcasting over other dimensions
            grid = xr.apply_ufunc(
                lambda x: self.weights[dim] @ x,  # pylint: disable=W0640
                grid,
                input_core_dims=[[dim]],
                output_core_dims=[[dim]],
                exclude_dims={dim},
                vectorize=True,
            )
            # insert the coarse coordinates
            grid = grid.assign_coords({dim: self.target_coords[dim].data})

        return grid

    def __repr__(self):
        """Returns information about the regridder."""

        source_shape = dict(zip(self.dims, self.source_shape))
        target_shape = dict(zip(self.dims, self.target_shape))
        info = (
            'Conservative Regridder\n'
            f'  Input grid shape: {source_shape}\n'
            f'  Output grid shape: {target_shape}'
        )
        return info


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

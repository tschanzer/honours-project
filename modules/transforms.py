"""Module for spectral transforms on Dedalus output."""

import dedalus.public as d3
import numpy as np
import xarray as xr

# pylint: disable=too-few-public-methods


class SpatialTransform:
    """Class for mathematical operations on Dedalus output."""

    def __init__(self, Nx, Nz):
        """
        Instantiates a Math object.

        Args:
            Nx: Horizontal resolution.
            Nz: Vertical resolution.
        """

        self.coords = d3.CartesianCoordinates('x', 'z')
        self.dist = d3.Distributor(self.coords, dtype=np.float64)
        self.xbasis = d3.RealFourier(
            self.coords['x'], size=Nx, dealias=3/2, bounds=(0, Nx/Nz))
        self.zbasis = d3.ChebyshevT(
            self.coords['z'], size=Nz, dealias=3/2, bounds=(0, 1))
        self.x, self.z = self.dist.local_grids(self.xbasis, self.zbasis)

    def __call__(self, data, to_layout='c'):
        """
        Transforms data between grid and coefficient space.

        Args:
            data: xarray.DataArray or xarray.Dataset to be transformed.
                Must contain dimensions 'x' and 'z' if
                `to_layout == 'c'`, or 'kx' and 'kz' if
                `to_layout == 'g'`.
            to_layout: Target layout ('g' for grid space, 'c' for
                coefficient space).

        Returns:
            Transformed xarray.DataArray or xarray.Dataset.
        """

        if to_layout == 'c':
            input_core_dims = [['x', 'z']]
            output_core_dims = [['kx', 'kz']]
        elif to_layout == 'g':
            input_core_dims = [['kx', 'kz']]
            output_core_dims = [['x', 'z']]
        else:
            raise ValueError(f'Invalid layout: {to_layout}.')

        transformed_data = xr.apply_ufunc(
            self._transform, data,
            kwargs={'to_layout': to_layout},
            input_core_dims=input_core_dims,
            output_core_dims=output_core_dims,
            vectorize=True,
        )
        return transformed_data

    def _transform(self, data, to_layout):
        """
        Transforms raw data between grid and coefficient space.

        Args:
            data: numpy.ndarray with shape `(self.x, self.z)`.
            to_layout: Target layout ('g' for grid space, 'c' for
                coefficient space).

        Returns:
            numpy.ndarray with shape `(self.x, self.z)` containing
            the transformed data.
        """

        if to_layout == 'c':
            from_layout = 'g'
        elif to_layout == 'g':
            from_layout = 'c'
        else:
            raise ValueError(f'Invalid layout: {to_layout}.')

        field = self.dist.Field(bases=(self.xbasis, self.zbasis))
        field[from_layout] = data
        return field[to_layout]

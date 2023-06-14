"""Module for mathematical operations on Dedalus output."""

import dedalus.public as d3
import numpy as np


class Math:
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

    def array_to_field(self, data):
        """
        Imports an xarray.DataArray into a dedalus Field.

        Args:
            data: xarray.DataArray with 'x' and 'z' dimensions.

        Returns:
            dedalus Field.
        """

        field = self.dist.Field(
            name=data.name, bases=(self.xbasis, self.zbasis))
        field['g'] = data.data
        return field

    def space_coefficients(self, data):
        """
        Transforms spatial data into Fourier/Chebyshev coefficient space.

        Args:
            data: xarray.DataArray with 'x' and 'z' dimensions.

        Returns:
            Fourier/Chebyshev coefficient array.
        """

        field = self.array_to_field(data)
        coef = data.copy()
        coef.data = field['c']
        coef.name = 'coef_' + coef.name
        coef = coef.rename({'x': 'kx', 'z': 'kz'})
        coef['kx'] = np.arange(coef.kx.size)
        coef['kz'] = np.arange(coef.kz.size)
        return coef

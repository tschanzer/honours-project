"""Module for spatial manipulations."""

import dedalus.public as d3
import numpy as np
import scipy as sp
import xarray as xr

# pylint: disable=too-few-public-methods


class Filter:
    """Class for filtering data onto coarser grids."""

    def __init__(self, from_shape, to_shape, aspect):
        """
        Constructs a Filter.

        Args:
            from_shape: (Nx, Nz) of the input data.
            to_shape: (Nx, Nz) of the output data.
            aspect: Domain aspect ratio (width/height).
        """

        self.from_shape = from_shape
        self.to_shape = to_shape
        self.aspect = aspect
        self.truncate = 4.0  # filter truncated at 4 standard deviations

        # construct the Field for z-interpolation
        coords = d3.CartesianCoordinates('x', 'z')
        dist = d3.Distributor(coords, dtype=np.float64)
        xbasis = d3.RealFourier(
            coords['x'], size=from_shape[0], bounds=(0, aspect))
        zbasis = d3.ChebyshevT(
            coords['z'], size=from_shape[1], bounds=(0, 1))
        self.field = dist.Field(bases=(xbasis, zbasis))

        # construct the upsampled grid
        self.x_fine = xbasis.global_grid().squeeze()
        self.z_fine = np.linspace(0, 1, 2*self.from_shape[1])
        self.dx_fine = self.x_fine[1] - self.x_fine[0]
        self.dz_fine = self.z_fine[1] - self.z_fine[0]

        # construct the coarse grid
        xbasis = d3.RealFourier(coords['x'], size=256, bounds=(0, aspect))
        zbasis = d3.ChebyshevT(coords['z'], size=32, bounds=(0, 1))
        self.x_coarse = xbasis.global_grid().squeeze()
        self.z_coarse = zbasis.global_grid().squeeze()

    def __call__(self, data, std, kind=None):
        """
        Filters data onto a coarser grid.

        Args:
            data: xarray.DataArray or xarray.Dataset.
            std: Standard deviation of the Gaussian filter to be used.
            kind: Optional, used to override the extension method.
                The variable name(s) are used if this is not supplied.
                Options are:
                    'u': Ensure zero value at z=0,1
                    'w': Ensure zero vertical derivative at z=0,1
                    'theta': Ensure 1/2 at z=0 and -1/2 at z=1

        Returns:
            Filtered xarray.DataArray or xarray.Dataset.
        """

        if isinstance(data, xr.DataArray):
            return self._process_dataarray(
                data, std, data.name if kind is None else kind)
        elif isinstance(data, xr.Dataset):
            output = {}
            for var in data.data_vars:
                output[var] = self._process_dataarray(
                    data[var], std, var if kind is None else kind)

            return xr.Dataset(output)
        else:
            raise ValueError(f'Invalid input type {type(data)}.')

    def _upsample(self, data):
        """
        Vertically upsamples the input field onto a regular grid.

        Upsampling is performed by Chebyshev interpolation.

        Args:
            data: numpy.ndarray.

        Returns:
            numpy.ndarray with shape (data.shape[0], 2*data.shape[1])
        """

        data_interp = np.zeros((self.from_shape[0], self.z_fine.size))

        self.field['g'] = data
        for i, pos in enumerate(self.z_fine):
            data_interp[:,i] = self.field(z=pos).evaluate()['g'].squeeze()

        return data_interp

    def _extend(self, data, std, var):
        """
        Extends the upsampled array, following Bae and Lozano-Durán (2017).

        Args:
            data: Upsampled numpy.ndarray.
            std: Standard deviation of the Gaussian filter to be used.
            var: Variable being regridded ('u', 'w' or 'theta').

        Returns:
            Extended numpy.ndarray.
        """

        # number of grid points corresponding to 1 standard deviation
        std_x = std/self.dx_fine
        std_z = std/self.dz_fine

        # truncation lengths
        truncate_x = self.truncate*std_x
        truncate_z = self.truncate*std_z

        # extension lengths
        extend_x = int(np.ceil(truncate_x))
        extend_z = int(np.ceil(truncate_z))

        # perform periodic x extension
        data = np.concatenate([
            data[-extend_x:,:],
            data,
            data[:extend_x,:],
        ], axis=0)

        # perform z extension
        if var == 'theta':
            data_before = 1 - data[:, extend_z:0:-1]
            data_after = -1 - data[:, -2:-(extend_z+2):-1]
        elif var == 'u':
            data_before = -data[:, extend_z:0:-1]
            data_after = -data[:, -2:-(extend_z+2):-1]
        elif var == 'w':
            data_before = data[:, extend_z:0:-1]
            data_after = data[:, -2:-(extend_z+2):-1]
        else:
            raise ValueError(f'Invalid variable name {var}.')

        data = np.concatenate([data_before, data, data_after], axis=1)
        return data

    def _filter(self, data, std):
        """
        Gaussian filters the extended array.

        Args:
            data: Extended numpy.ndarray.
            std: Standard deviation of the filter.

        Returns:
            Filtered numpy.ndarray.
        """

        std_x = std/self.dx_fine
        std_z = std/self.dz_fine
        return sp.ndimage.gaussian_filter(
            data, sigma=(std_x, std_z), truncate=self.truncate)

    def _truncate(self, data, std):
        """
        Truncates the extended array after filtering.

        Args:
            data: Filtered numpy.ndarray.
            std: Standard deviation of the filter.

        Returns:
            Truncated numpy.ndarray.
        """

        extend_x = int(np.ceil(self.truncate*std/self.dx_fine))
        extend_z = int(np.ceil(self.truncate*std/self.dz_fine))
        return data[extend_x:-extend_x, extend_z:-extend_z]

    def _downsample(self, data):
        """
        Downsamples the truncated array.

        Args:
            data: Truncated numpy.ndarray.

        Returns:
            Downsampled numpy.ndarray.
        """

        x_sampler = sp.interpolate.interp1d(self.x_fine, data, axis=0)
        data = x_sampler(self.x_coarse)
        z_sampler = sp.interpolate.interp1d(self.z_fine, data, axis=1)
        data = z_sampler(self.z_coarse)
        return data

    def _process(self, data, std, var):
        """
        Performs all processing steps.

        Args:
            data: numpy.ndarray.
            std: Standard deviation of the Gaussian filter to be used.
            var: Variable being regridded ('u', 'w' or 'theta').

        Returns:
            numpy.ndarray.
        """

        data = self._upsample(data)
        data = self._extend(data, std, var)
        data = self._filter(data, std)
        data = self._truncate(data, std)
        data = self._downsample(data)
        return data

    def _process_dataarray(self, data, std, var):
        """
        Processes an xarray.DataArray.

        Args:
            data: xarray.DataArray.
            std: Standard deviation of the Gaussian filter to be used.
            var: Variable being regridded ('u', 'w' or 'theta').

        Returns:
            xarray.DataArray.
        """

        data = xr.apply_ufunc(
            self._process, data,
            kwargs={'std': std, 'var': var},
            input_core_dims=[['x', 'z']],
            output_core_dims=[['x', 'z']],
            exclude_dims={'x', 'z'},
            dask='allowed',
        )
        return data.assign_coords({'x': self.x_coarse, 'z': self.z_coarse})

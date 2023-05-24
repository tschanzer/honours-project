"""Wrappers of xesmf.Regridder for arbitrary data."""

import numpy as np
import xesmf as xe
import xarray as xr
import warnings


def bounds(grid):
    """
    Calculates cell bounds for a 1D coordinate grid.

    Assumes that points on the coordinate grid are the midpoints of
    the cells and that the first and last cells have equal width.
    Note: this is not, in general, the same as assuming that the cell
    bounds are halfway between points on the coordinate grid.

    Args:
        grid: 1D coordinate grid.

    Returns:
        Array of cell edges with length x.size + 1.
    """

    mat = np.eye(grid.size + 1) + np.eye(grid.size + 1, k=1)
    mat[-1, 0] = 1
    rhs = np.concatenate([2*grid, [grid[0] + grid[-1]]])
    return np.linalg.solve(mat, rhs)


class Regridder2D(xe.Regridder):
    """Wrapper of xesmf.Regridder for data with different coordinate names."""

    def __init__(self, highres, lowres, coords=('x', 'z')):
        """
        Creates a 1st order conservative regridding object.

        Args:
            highres: High-resolution xarray.Dataset or DataArray.
            lowres: Low-resolution xarray.Dataset or DataArray.
            coords: 2-tuple of regridding coordinate names.

        Returns:
            Modified xesmf.Regridder object that regrids data from the
            `highres` grid onto the `lowres` grid.
        """

        self.coords = coords
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message=r'Latitude is outside of \[-90, 90\]')
            super().__init__(
                self._prep_regrid(highres),
                self._prep_regrid(lowres),
                'conservative',
            )

    def __call__(self, data):
        """
        Coarse-grains high-resolution data.

        Args:
            data: High-resolution xarray.Dataset or DataArray.
                to be coarse-grained.

        Returns:
            Coarse-grained version of the input.
        """

        data = self._coords_to_latlon(data)
        data = self._check_dim_order(data)
        result = super().__call__(data)
        return self._latlon_to_coords(result)

    def _coords_to_latlon(self, data):
        """
        Renames coordinates for regridding by xESMF.

        Args:
            data: xarray.Dataset or DataArray.

        Returns:
            xarray.Dataset or DataArray with renamed coordinates.
        """

        return data.rename({self.coords[0]: 'lat', self.coords[1]: 'lon'})

    def _latlon_to_coords(self, data):
        """
        Undoes the action of _coords_to_latlon.

        Args:
            data: xarray.Dataset or DataArray with renamed coordinates.

        Returns:
            xarray.Dataset or DataArray with original coordinate names.
        """

        return data.rename({'lat': self.coords[0], 'lon': self.coords[1]})

    def _prep_regrid(self, data):
        """
        Prepares an array for regridding by xESMF.

        Args:
            data: xarray.Dataset or DataArray.

        Returns:
            xarray.Dataset or DataArray with renamed coordinates and
            added coordinate bounds.
        """

        data = self._coords_to_latlon(data)
        data = data.assign_coords({
            'lat_b': bounds(data.lat.data),
            'lon_b': bounds(data.lon.data),
        })
        return data

    def _check_dim_order(self, data):
        """
        Ensures the dimensions are in the right order.

        Args:
            data: xarray.Dataset or DataArray with 'lat', and 'lon'
                coordinates.

        Returns:
            C-contiguous copy of the input with 'lat' and 'lon' as the
            last two dimensions.
        """
        dims = list(data.dims)
        if dims[-2:] == ['lat', 'lon']:
            return data

        other_coords = [dim for dim in dims if dim not in ['lat', 'lon']]
        data = data.transpose(*other_coords, 'lat', 'lon')
        # The data must be made C-contiguous to optimise performance
        data = data.astype(
            data.dtypes if isinstance(data, xr.Dataset) else data.dtype,
            order='C',
        )
        return data


class Regridder1D(Regridder2D):
    """Wrapper of xesmf.Regridder for 1D regridding."""

    def __init__(self, highres, lowres, coord='t'):
        """
        Creates a 1st order conservative regridding object.

        Args:
            highres: High-resolution xarray.Dataset or DataArray.
            lowres: Low-resolution xarray.Dataset or DataArray.
            coord: Name of regridding coordinate.

        Returns:
            Modified xesmf.Regridder object that regrids data from the
            `highres` grid onto the `lowres` grid.
        """

        self.coord = coord
        self.dummy_coord = 'dummy_' + coord
        super().__init__(highres, lowres, (self.coord, self.dummy_coord))

    def _add_dummy_coord(self, data):
        """Adds a length-1 dummy coordinate."""
        return data.expand_dims({self.dummy_coord: [0]}, axis=-1)

    def _remove_dummy_coord(self, data):
        """Removes the length-1 dummy coordinate."""
        return data.squeeze(self.dummy_coord, drop=True)

    def _coords_to_latlon(self, data):
        """Wrapper of Regridder2D._coords_to_latlon for 1D regridding."""
        data = self._add_dummy_coord(data)
        return super()._coords_to_latlon(data)

    def _latlon_to_coords(self, data):
        """Wrapper of Regridder2D._latlon_to_coords for 1D regridding."""
        data = super()._latlon_to_coords(data)
        return self._remove_dummy_coord(data)

    def _prep_regrid(self, data):
        """Equivalent of Regridder2D._prep_regrid for 1D regridding."""
        data = self._coords_to_latlon(data)
        data = data.assign_coords({
            'lat_b': bounds(data.lat.data),
            'lon_b': [-1/2, 1/2],
        })
        return data


class Regridder3D:
    """Wrapper of xesmf.Regridder for 3D regridding."""

    def __init__(self, highres, lowres, coords=('x', 'z', 't')):
        """
        Creates a 1st order conservative regridding object.

        Args:
            highres: High-resolution xarray.Dataset or DataArray.
            lowres: Low-resolution xarray.Dataset or DataArray.
            coord: Name of regridding coordinates. The first two will
                be regridded first, followed by the third.

        Returns:
            Modified xesmf.Regridder object that regrids data from the
            `highres` grid onto the `lowres` grid.
        """
        self.regridder2d = Regridder2D(highres, lowres, coords[:2])
        self.regridder1d = Regridder1D(highres, lowres, coords[-1])

    def __call__(self, data):
        """
        Coarse-grains high-resolution data.

        Args:
            data: High-resolution xarray.Dataset or DataArray.
                to be coarse-grained.

        Returns:
            Coarse-grained version of the input.
        """

        return self.regridder1d(self.regridder2d(data))

    def __repr__(self):
        """Returns information about the regridder."""
        repr_2d = self.regridder2d.__repr__()
        repr_1d = self.regridder1d.__repr__()
        return repr_2d + '\n\n' + repr_1d

"""Analysis tools for data-driven parametrisation."""

import numpy as np
import scipy as sp
import xarray as xr


def tendency(state_t, state_tplusdt):
    """
    Forward difference approximation to time derivative.

    Args:
        state_t: xr.DataArray or xr.Dataset of initial data.
        state_tplusdt: xr.DataArray or xr.Dataset of data at a later time.

    Returns:
        xr.DataArray or xr.Dataset corresponding to
        (state_tplusdt - state_t)/dt.
    """

    tend = (state_tplusdt.drop('t') - state_t.drop('t'))/state_t.timestep
    return tend.assign_coords({'t': state_t.t})


def insert_bc(data, bottom, top, aspect=8):
    """
    Inserts the boundary values into a DataArray.

    Args:
        data: xr.DataArray.
        bottom: z=0 boundary condition.
        top: z=1 boundary condition.
        aspect: Domain aspect ratio.

    Returns:
        xr.DataArray with z=0,1 and periodic x boundary values inserted.
    """

    # z=0 value
    z_bc = xr.DataArray([bottom], coords={'z': [0.]})
    _, z_bc = xr.broadcast(data.isel(z=0), z_bc)
    data = xr.concat([z_bc, data], dim='z')

    # z=1 value
    z_bc = xr.DataArray([top], coords={'z': [1.]})
    _, z_bc = xr.broadcast(data.isel(z=0), z_bc)
    data = xr.concat([data, z_bc], dim='z')

    # Periodic x value
    x_bc = data.sel(x=0).assign_coords({'x': aspect})
    data = xr.concat([data, x_bc], dim='x')
    return data


def sample(data, x, z, bottom, top, aspect=8):
    """
    Samples a field at given points using linear interpolation.

    Args:
        data: xr.DataArray.
        x, z: 1D arrays of coordinates for the sampling points.
        bottom, top: z=0,1 boundary values for the field.
        aspect: Domain aspect ratio.

    Returns:
        1D np.ndarray of field values at sampling points.
    """

    data = insert_bc(data, bottom, top, aspect)
    return sp.interpolate.interpn(
        (data.x.data, data.z.data), data.transpose('x', 'z', 't').data,
        np.stack([x, z], axis=1),
    ).ravel()

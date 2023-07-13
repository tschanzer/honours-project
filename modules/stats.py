"""Statistics for Rayleigh-Benard convection."""

import numpy as np
import xarray as xr


def level_quantile(data, level, quantile, t_range, regridder):
    """
    Calculates a quantile of a variable at a given z level.

    Args:
        data: xarray.DataArray containing the variable of interest.
        level: Vertical position at which to compute the quantile.
        quantile: Desired quantile.
        t_range: List of lower and upper time limits to consider.
        regridder: Regridder object for coarsening the data.

    Returns:
        The requested quantile of the coarsened data for `z == level`,
        `t_range[0] <= t <= t_range[1]`.
    """

    mask = (data.t >= t_range[0]) & (data.t >= t_range[1])
    level_data = data.isel(t=mask).interp(z=level)
    level_data = level_data.chunk({'t': -1, 'x': -1})
    level_data = regridder(level_data)
    return np.abs(level_data).quantile(quantile)


def nusselt_number(data, rayleigh, prandtl):
    """
    Calculates the (instantaneous) vertically averaged Nusselt number.

    Args:
        data: xarray.Dataset.
        rayleigh: Rayleigh number.
        prandtl: Prandtl number.

    Returns:
        xarray.DataArray
    """

    convection = np.sqrt(rayleigh*prandtl)*(data.w*data.theta).mean(['x'])
    return 1 + convection.integrate('z')/(data.z.max() - data.z.min())


def rms_speed(data):
    """
    Calculates the instantaneous RMS speed.

    Args:
        data: xarray.Dataset.

    Returns:
        xarray.DataArray
    """

    mean_square = (data.u**2 + data.w**2).mean('x')
    mean_square = mean_square.integrate('z')/(data.z.max() - data.z.min())
    return np.sqrt(mean_square)


def kinetic_dissipation(data, rayleigh, prandtl):
    """
    Calculates the spatially averaged kinetic energy dissipation rate.

    Args:
        data: xarray.Dataset.
        rayleigh: Rayleigh number.
        prandtl: Prandtl number.

    Returns:
        xarray.DataArray
    """

    # Compute the Jacobian du_i/dx_j
    jacobian = [[None, None], [None, None]]
    coords = ['x', 'z']
    components = [data.u, data.w]
    for i in range(2):
        for j in range(2):
            jacobian[i][j] = components[i].differentiate(coords[j])

    # Compute the sum
    dissipation = 0
    for i in range(2):
        for j in range(2):
            dissipation += (jacobian[i][j] + jacobian[j][i])**2

    # average over space and time
    dissipation = dissipation.mean('x')
    dissipation = dissipation.integrate('z')/(data.z.max() - data.z.min())
    dissipation *= 0.5*(prandtl/rayleigh)**0.5
    return dissipation


def thermal_dissipation(data, rayleigh, prandtl):
    """
    Calculates the spatially averaged thermal energy dissipation rate.

    Args:
        data: xarray.Dataset.
        rayleigh: Rayleigh number.
        prandtl: Prandtl number.

    Returns:
        xarray.DataArray
    """

    dissipation = (
        data.theta.differentiate('x')**2
        + data.theta.differentiate('z')**2
    )

    # average over space and time
    dissipation = dissipation.mean('x')
    dissipation = dissipation.integrate('z')/(data.z.max() - data.z.min())
    dissipation *= (rayleigh*prandtl)**(-0.5)
    return dissipation


def _running_mean(data):
    mean = np.zeros_like(data)
    mean[..., 0] = data[..., 0]
    for i in range(1, data.shape[-1]):
        mean[..., i] = mean[..., i-1]*i/(i + 1) + data[..., i]/(i + 1)
    return mean


def running_mean(data, dim):
    """
    Calculates the running mean along a dimension.

    Args:
        data: xarray.Dataset or xarray.Dataset.
        dim: Dimension name.

    Returns:
        xarray.Dataset or xarray.Dataset
    """

    return xr.apply_ufunc(
        _running_mean, data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
    )


def reverse_running_mean(data, dim):
    """
    Calculates the running mean along a dimension, starting from the end.

    Args:
        data: xarray.Dataset or xarray.Dataset.
        dim: Dimension name.

    Returns:
        xarray.Dataset or xarray.Dataset
    """

    return xr.apply_ufunc(
        lambda x: _running_mean(x[..., ::-1]), data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
    )

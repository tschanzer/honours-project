"""Statistics for Rayleigh-Benard convection."""

import numpy as np
import xarray as xr

from modules import tools


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


def thermal_bl_thickness(data):
    """
    Calculate the mean thermal boundary layer thickness.

    Mean is taken over time and x, and between the top and bottom
    boundary layers.

    Args:
        data: xarray.Dataset.
    """

    profile = data.theta.mean(['x']).compute()
    profile = tools.insert_bc(profile, (1/2, -1/2), aspect=None)
    lapse_rate = -profile.differentiate('z', edge_order=2)
    return (0.5/lapse_rate.sel(z=0.) + 0.5/lapse_rate.sel(z=1.))/2


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
        data: xarray.DataArray or xarray.Dataset.
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
        data: xarray.DataArray or xarray.Dataset.
        dim: Dimension name.

    Returns:
        xarray.Dataset or xarray.Dataset
    """

    return xr.apply_ufunc(
        lambda x: _running_mean(x[..., ::-1]), data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
    )


def rolling_mean_half_range(data, dim, width):
    """
    Calculate half the range of the rolling mean of an array.

    Args:
        data: xarray.DataArray.
        dim: Dimension name.
        width: Width of the rolling window in *data units*.

    Returns:
        (max - min)/2 of the rolling mean of the array.
    """

    delta = (data[dim][1] - data[dim][0]).item()
    width_n = round(width/delta) + 1
    mean = data.rolling({dim: width_n}).mean()
    return (np.nanmax(mean) - np.nanmin(mean))/2


def running_mean_half_range(data, dim, width):
    """
    Calculate half the range of the running mean of an array.

    Args:
        data: xarray.DataArray.
        dim: Dimension name.
        width: Minimum width of the window in *data units*.

    Returns:
        (max - min)/2 of the running mean of the array,
        limited by the minimum window width.
    """

    mean = running_mean(data, dim)
    mean = mean.isel({dim: mean[dim] >= mean[dim][0] + width})
    return (mean.max() - mean.min())/2


def mean_and_uncertainty(data, dim, width):
    """
    Calculate the mean of a time series with uncertainty.

    Args:
        data: xarray.DataArray.
        dim: Dimension name.
        width: Width of the window in *data units*.

    Returns:
        Mean (over the last <width> data units) and uncertainty.
    """

    mean = data.mean(dim)
    uncertainty = np.maximum(
        rolling_mean_half_range(data, dim, width),
        running_mean_half_range(data, dim, width),
    )
    return mean, uncertainty


def time_autocorrelation(array, max_lag, lag_step):
    """
    Calculate the spatially-averaged temporal autocorrelation.

    Args:
        data: xarray.DataArray.
        max_lag: Maximum lag for the autocorrelation.
        lag_step: Lag step size for the autocorrelation.

    Returns:
        xarray.DataArray.
    """

    step = array.t.diff('t')
    if not np.allclose(step, step[0]):
        raise ValueError('Coordinate must have regular steps.')
    step = step[0].item()

    # Number of grid points corresponding to one lag step
    n_lag_step = round(lag_step/step)
    # Round lag step to the nearest multiple of grid spacing
    lag_step = n_lag_step*step
    # Number of lag steps needed to reach max_lag
    n_max_lag = int(max_lag//lag_step)

    result = []
    for n in range(n_max_lag + 1):
        array1 = array.isel(t=slice(n*n_lag_step, None))
        array2 = array.isel(t=slice(None, array['t'].size - n*n_lag_step))
        corr = xr.corr(array1.drop('t'), array2.drop('t'), 't')
        corr = corr.mean('x').integrate('z')/(array.z.max() - array.z.min())
        result.append(corr.compute())

    result = xr.concat(result, 't_lag')
    result = result.assign_coords(
        {'t_lag': np.linspace(0, n_max_lag*lag_step, n_max_lag + 1)}
    )
    return result

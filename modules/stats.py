"""Statistics for Rayleigh-Benard convection."""

import numpy as np


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

"""Analysis tools for data-driven parametrisation."""

import numpy as np
import scipy as sp
import xarray as xr
import string


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

    tend = (
        (state_tplusdt.drop('t') - state_t.drop('t'))
        / (state_tplusdt.t.drop('t') - state_t.t.drop('t'))
    )
    return tend.assign_coords({'t': state_t.t})


def insert_bc(data, bc, aspect):
    """
    Inserts the boundary values into a DataArray.

    Args:
        data: xarray.DataArray or xarray.Dataset.
        bc: If data is a DataArray, a tuple (bottom, top) of bottom/top
            boundary values. If data is a Dataset, a dict mapping
            variable names to tuples of the above form.
        aspect: Domain aspect ratio.

    Returns:
        xr.DataArray with z=0,1 and periodic x boundary values inserted.
    """

    # z=0 value
    if isinstance(data, xr.DataArray):
        bottom = xr.DataArray([bc[0]], coords={'z': [0.]})
    elif isinstance(data, xr.Dataset):
        bottom = xr.Dataset({
            var: xr.DataArray([val[0]], coords={'z': [0.]})
            for var, val in bc.items()
        })
    else:
        raise ValueError('Invalid data type.')

    _, bottom = xr.broadcast(data.isel(z=0), bottom)
    data = xr.concat([bottom, data], dim='z')

    # z=1 value
    if isinstance(data, xr.DataArray):
        top = xr.DataArray([bc[1]], coords={'z': [1.]})
    else:
        top = xr.Dataset({
            var: xr.DataArray([val[1]], coords={'z': [1.]})
            for var, val in bc.items()
        })
    _, top = xr.broadcast(data.isel(z=0), top)
    data = xr.concat([data, top], dim='z')

    # Periodic x value
    x_bc = data.sel(x=0).assign_coords({'x': aspect})
    data = xr.concat([data, x_bc], dim='x')
    return data


def sample(data, x, z):
    """
    Samples a field at given points using linear interpolation.

    Args:
        data: xr.DataArray.
        x, z: 1D arrays of coordinates for the sampling points.

    Returns:
        1D np.ndarray of field values at sampling points.
    """

    return sp.interpolate.interpn(
        (data.x.data, data.z.data),
        data.transpose('x', 'z', 't').data,
        np.stack([x, z], axis=1),
    )


def label_subplots(axes):
    """Label subplots alphabetically."""
    if axes.ndim > 1: axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.set_title(f'({string.ascii_lowercase[i]})', loc='left')


def plot_z_dependence(
        x_data, y_data, axes, z_ranges, n_samples,
        aspect, sigma, **hist2d_kwargs):
    """
    Plot a series of histograms conditioned on z.

    Args:
        x_data, y_data: xr.DataArray.
        axes: Array of axes.
        ranges: List of (z_min, z_max) with length equal to axes.size.
        n_samples: Number of samples to take in each range.
        aspect: Domain aspect ratio.
        sigma: Standard deviation of the Gaussian filter used to smooth
            the histogram array for plotting contours.
        hist2d_kwargs: Additional keyword arguments to pass to hist2d.
    """

    rng = np.random.default_rng(seed=0)
    meshes = np.empty_like(axes)
    for i, (ax, (z_min, z_max)) in enumerate(zip(axes.ravel(), z_ranges)):
        x_sample = rng.uniform(0, aspect, n_samples)
        z_sample = rng.uniform(z_min, z_max, n_samples)
        x_data_sample = sample(x_data, x_sample, z_sample).ravel()
        y_data_sample = sample(y_data, x_sample, z_sample).ravel()
        hist, x, y, meshes.flat[i] = ax.hist2d(
            x_data_sample, y_data_sample, norm='log', rasterized=True,
            **hist2d_kwargs,
        )
        hist_contour(ax, x, y, hist, sigma)
        ax.set(title=f'$z \\in [{z_min:.3f}, {z_max:.3f}]$')

    return meshes


def hist_contour(axes, x, y, hist, sigma):
    """
    Plot smooth contours on a 2D histogram.

    Args:
        axes: Axes on which to draw the contours.
        x, y: Bin edges of the histogram.
        hist: Histogram array.
        sigma: Standard deviation of the Gaussian filter used to smooth
            the histogram array.

    Returns:
        QuadContourSet.
    """

    hist = sp.ndimage.gaussian_filter(hist, sigma=sigma)
    hist[hist >= 1] = np.log10(hist[hist >= 1])
    contour = axes.contour(
        (x[:-1] + x[1:])/2, (y[:-1] + y[1:])/2, hist.T,
        colors='k', linewidths=0.5, levels=np.arange(1, hist.max(), 0.5)
    )
    return contour
